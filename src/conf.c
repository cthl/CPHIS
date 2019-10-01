// conf.c - implementation of the CPHIS configuration class

#include <cphis.h>
#include <conf.h>
#include <stdlib.h>

CphisError CphisConfCreate(
             CphisConf *conf,
             int numLocalDOF,
             int numScales,
             const int *scales
           )
{
  // Check input arguments.
  if (numLocalDOF < 1 || numScales < 1 || numScales > numLocalDOF) {
    CPHISCHECK(CPHIS_INVALID_ARG);
  }
  for (int s = 0; s < numScales; s++) {
    if (scales[2*s] < 0 || scales[2*s + 1] < 0) {
      CPHISCHECK(CPHIS_INVALID_ARG);
    }
    if (scales[2*s] >= numLocalDOF || scales[2*s + 1] >= numLocalDOF) {
      CPHISCHECK(CPHIS_INVALID_ARG);
    }
    if (scales[2*s] > scales[2*s + 1]) {
      CPHISCHECK(CPHIS_INVALID_ARG);
    }
  }

  *conf = malloc(sizeof(struct _CphisConf));
  if (!(*conf)) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  (*conf)->numLocalDOF = numLocalDOF;
  (*conf)->numScales = numScales;
  // Set up scales.
  (*conf)->scales = malloc(numScales*sizeof(int));
  if (!(*conf)->scales) {
    free(*conf);
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }
  for (int s = 0; s < numScales; s++) {
    (*conf)->scales[2*s] = scales[2*s];
    (*conf)->scales[2*s + 1] = scales[2*s + 1];
  }
  // Set up solvers.
  (*conf)->solvers = malloc(numScales*sizeof(CphisScaleSolver));
  if (!(*conf)->solvers) {
    free((*conf)->scales);
    free(*conf);
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }
  (*conf)->solversOwned = malloc(numScales*sizeof(int));
  if (!(*conf)->solversOwned) {
    free((*conf)->solversOwned);
    free((*conf)->scales);
    free(*conf);
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }
  for (int s = 0; s < numScales; s++) {
    (*conf)->solvers[s] = NULL;
    (*conf)->solversOwned[s] = 0;
  }

  // Set default parameters.
  (*conf)->rtol = 1.0e-6;
  (*conf)->atol = 0.0;
  (*conf)->maxIter = 1e6;
  (*conf)->nu1 = 1;
  (*conf)->nu2 = 0;
  (*conf)->cycle = CPHIS_CYCLE_V;
  (*conf)->fmgCycle = 0;
  (*conf)->verbosity = CPHIS_VERBOSITY_ALL;

  return CPHIS_SUCCESS;
}

CphisError CphisConfDestroy(CphisConf conf)
{
  // Destroy scale solvers if necessary.
  for (int s = 0; s < conf->numScales; s++) {
    if (conf->solversOwned[s]) {
      CphisScaleSolverDestroy(conf->solvers[s]);
    }
  }

  free(conf->scales);
  free(conf->solvers);
  free(conf->solversOwned);
  free(conf);
  return CPHIS_SUCCESS;
}

CphisError CphisConfSetTolRel(CphisConf conf, CphisReal tol)
{
  conf->rtol = tol;
  return CPHIS_SUCCESS;
}

CphisError CphisConfSetTolAbs(CphisConf conf, CphisReal tol)
{
  conf->atol = tol;
  return CPHIS_SUCCESS;
}

CphisError CphisConfSetMaxIter(CphisConf conf, int maxIter)
{
  if (maxIter < 0) {
    CPHISCHECK(CPHIS_INVALID_ARG);
  }
  conf->maxIter = maxIter;
  return CPHIS_SUCCESS;
}

CphisError CphisConfSetCycleType(CphisConf conf, CphisCycleType cycle)
{
  conf->cycle = cycle;
  return CPHIS_SUCCESS;
}

CphisError CphisConfSetFMGCycle(CphisConf conf, int fmgCycle)
{
  // FMG has not been implemented, so we don't allow the user to enable it.
  if (fmgCycle) {
    CPHISCHECK(CPHIS_NOT_IMPLEMENTED);
  }
  //conf->fmgCycle = fmgCycle;
  return CPHIS_SUCCESS;
}

CphisError CphisConfSetVerbosity(CphisConf conf, CphisVerbosityLevel verbosity)
{
  if (verbosity < 0) {
    CPHISCHECK(CPHIS_INVALID_ARG);
  }
  conf->verbosity = verbosity;
  return CPHIS_SUCCESS;
}

CphisError CphisConfSetNu(CphisConf conf, int nu1, int nu2)
{
  if (nu1 < 0 || nu2 < 0) {
    CPHISCHECK(CPHIS_INVALID_ARG);
  }
  
  conf->nu1 = nu1;
  conf->nu2 = nu2;
  return CPHIS_SUCCESS;
}

CphisError CphisConfSetScaleSolver(
             CphisConf conf,
             int scale,
             CphisScaleSolver solver
           )
{
  if (scale < 0 || scale >= conf->numScales) {
    return CPHIS_OUT_OF_BOUNDS;
  }

  conf->solvers[scale] = solver;

  return CPHIS_SUCCESS;
}

CphisError CphisConfSetSmoothers(CphisConf conf, CphisScaleSolverType type)
{
  CphisError err;

  // Create and set smoother for all scales except the coarsest one.
  for (int s = conf->numScales - 1; s > 0; s--) {
    err = CphisScaleSolverCreate(&conf->solvers[s], type);
    if (err) {
      // Clean up and abort.
      for (s++; s < conf->numScales; s++) {
        CphisScaleSolverDestroy(conf->solvers[s]);
      }
      CPHISCHECK(err);
    }
    // Since we created the solver, we need to make sure we destroy it later on.
    conf->solversOwned[s] = 1;
  }

  return CPHIS_SUCCESS;
}
