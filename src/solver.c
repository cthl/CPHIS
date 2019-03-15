// solver.c - CPHIS solver implementation

#include <cphis.h>
#include <solver.h>
#include <conf.h>
#include <aux.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static CphisError CphisSolverCleanupScale(CphisSolver solver, int scale)
{
  int skipAscale = 0;
  if (
    scale == solver->conf->numScales - 1 &&
    solver->conf->scales[2*scale] == 0
  ) {
    skipAscale = 1;
  }
  if (!skipAscale && solver->Ascale[scale]) {
    CphisMatDestroy(solver->Ascale[scale]);
  }
  if (solver->Aresidual[scale]) CphisMatDestroy(solver->Aresidual[scale]);
  if (solver->Arhs[scale]) CphisMatDestroy(solver->Arhs[scale]);
  if (solver->r[scale]) CphisVecDestroy(solver->r[scale]);
  if (solver->e[scale]) CphisVecDestroy(solver->e[scale]);
  if (solver->fscale[scale]) CphisVecDestroy(solver->fscale[scale]);
  if (solver->uscale[scale]) CphisVecDestroy(solver->uscale[scale]);
  if (solver->urhs[scale]) CphisVecDestroy(solver->urhs[scale]);

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverSetupScale(CphisSolver solver, int scale)
{
  CphisError err;

  // Make sure that it is always safe to abort.
  solver->Ascale[scale] = NULL;
  solver->Aresidual[scale] = NULL;
  solver->Arhs[scale] = NULL;
  solver->r[scale] = NULL;
  solver->e[scale] = NULL;
  solver->fscale[scale] = NULL;
  solver->uscale[scale] = NULL;
  solver->urhs[scale] = NULL;

  // Prepare creation of matrices and vectors.
  const CphisBackendType type = solver->A->type;
  const CphisIndex numElements = solver->A->numElements;
  const CphisIndex *elements = solver->A->elements;
  // Gather some information about the scale.
  // Number of local d.o.f. in the system
  const int numLocalDOF = solver->conf->numLocalDOF;
  // Number of matrix rows
  const CphisIndex numRows = numElements*numLocalDOF;
  // Number of local d.o.f. that are smoothed on this scale
  const int numLocalDOFScale = solver->conf->scales[2*scale + 1]
                             - solver->conf->scales[2*scale]
                             + 1;
  // Minimum local d.o.f. that is smoothed on this scale
  const int minLocalDOFScale = solver->conf->scales[2*scale];
  // Maximum local d.o.f. that is smoothed on this scale
  const int maxLocalDOFScale = solver->conf->scales[2*scale + 1];
  // Number of local d.o.f. in the residual, i.e., in all lower scales combined
  int numLocalDOFResidual = -1;
  if (scale > 0) {
    numLocalDOFResidual = solver->conf->scales[2*(scale - 1) + 1] + 1;
  }
  // Maximum local d.o.f. in the residual, i.e., in any of the lower scales
  int maxLocalDOFResidual = -1;
  if (scale > 0) {
    maxLocalDOFResidual = solver->conf->scales[2*(scale - 1) + 1];
  }
  // Is the scale of HSS-type, i.e., does it smooth only a subset of its local
  // d.o.f.?
  const int isHSSType = minLocalDOFScale != 0;
  // Is this an error correction scale, i.e., one that starts with an initial
  // guess of zero?
  const int isErrorCorrectionScale = scale < solver->conf->numScales - 1;
  // Number of local d.o.f. in this scale that will affect the residual
  int numLocalDOFScaleToResidual;
  if (isErrorCorrectionScale) {
    // Only the smoothed local d.o.f. affect the residual.
    numLocalDOFScaleToResidual = numLocalDOFScale;
  }
  else {
    // All local d.o.f. affect the residual.
    numLocalDOFScaleToResidual = maxLocalDOFScale + 1;
  }
  // Minimum local d.o.f. that will affect the residual
  int minLocalDOFScaleToResidual;
  if (isErrorCorrectionScale) {
    minLocalDOFScaleToResidual = minLocalDOFScale;
  }
  else {
    minLocalDOFScaleToResidual = 0;
  }

  // If the highest scale includes all lower d.o.f. (as in standard
  // p-multigrid), Ascale[numScales - 1] is equal to the original system matrix,
  // so we do not have to create a copy.
  int skipAscale = 0;
  if (scale == solver->conf->numScales - 1 && !isHSSType) {
    solver->Ascale[scale] = solver->A;
    skipAscale = 1;
  }
  if (!skipAscale) {
    err = CphisMatCreate(
            &solver->Ascale[scale],
            numElements,
            elements,
            numLocalDOFScale,
            type
          );
    if (err) {
      CphisSolverCleanupScale(solver, scale);
      CPHISCHECK(err);
    }
  }

  // Aresidual, r, and e are only needed on higher scales, and their size is
  // determined by the maximum local d.o.f. in the next lower scale.
  if (scale > 0) {
    err = CphisVecCreate(
            &solver->r[scale],
            numElements,
            elements,
            numLocalDOFResidual,
            type
          );
    if (err) {
      CphisSolverCleanupScale(solver, scale);
      CPHISCHECK(err);
    }
    err = CphisVecCreate(
            &solver->e[scale],
            numElements,
            elements,
            numLocalDOFResidual,
            type
          );
    if (err) {
      CphisSolverCleanupScale(solver, scale);
      CPHISCHECK(err);
    }
    err = CphisMatCreate(
            &solver->Aresidual[scale],
            numElements,
            elements,
            numLocalDOFResidual,
            type
          );
    if (err) {
      CphisSolverCleanupScale(solver, scale);
      CPHISCHECK(err);
    }
  }

  // Arhs, fscale, and uscale are only needed on HSS-type scales. Their size is
  // determined by the number of local d.o.f. that are smoothed on the current
  // scale.
  if (isHSSType) {
    err = CphisVecCreate(
            &solver->fscale[scale],
            numElements,
            elements,
            numLocalDOFScale,
            type
          );
    if (err) {
      CphisSolverCleanupScale(solver, scale);
      CPHISCHECK(err);
    }
    err = CphisVecCreate(
            &solver->uscale[scale],
            numElements,
            elements,
            numLocalDOFScale,
            type
          );
    if (err) {
      CphisSolverCleanupScale(solver, scale);
      CPHISCHECK(err);
    }
    err = CphisVecCreate(
            &solver->urhs[scale],
            numElements,
            elements,
            minLocalDOFScale,
            type
          );
    if (err) {
      CphisSolverCleanupScale(solver, scale);
      CPHISCHECK(err);
    }
    err = CphisMatCreate(
            &solver->Arhs[scale],
            numElements,
            elements,
            numLocalDOFScale,
            type
          );
    if (err) {
      CphisSolverCleanupScale(solver, scale);
      CPHISCHECK(err);
    }
  }

  // Extract the matrix blocks.
  int numErrors = 0;
  #pragma omp parallel for reduction(+:numErrors)
  for (CphisIndex i = 0; i < numRows; i++) {
    const int rowLocalDOF = i%numLocalDOF;
    if (rowLocalDOF > maxLocalDOFScale) {
      // Matrix row belongs to a higher scale, so we skip it.
      continue;
    }
    
    // Get matrix entries in this row.
    CphisIndex *cols;
    CphisScalar *vals;
    CphisIndex numEntries;
    err = CphisMatGetData(
            solver->A,
            i,
            &cols,
            &vals,
            &numEntries
          );
    if (err) {
      numErrors++;
      CPHISCHECKTHREAD(err);
    }

    // Check if the row belongs to the current scale and/or to lower scales.
    const int isRowInScale = rowLocalDOF >= minLocalDOFScale;
    const int isRowInResidual = rowLocalDOF <= maxLocalDOFResidual;

    for (CphisIndex j = 0; j < numEntries; j++) {
      const int colLocalDOF = cols[j]%numLocalDOF;
      if (colLocalDOF > maxLocalDOFScale) {
        // Column belongs to a higher scale, so we skip it.
        continue;
      }

      // Check if the row belongs to the current scale.
      const int isColInScale = colLocalDOF >= minLocalDOFScale;

      CphisIndex iBlock, jBlock;
      if (!skipAscale && isRowInScale && isColInScale) {
        // Entry goes to Ascale.
        iBlock = (i/numLocalDOF)*numLocalDOFScale
               + rowLocalDOF
               - minLocalDOFScale;
        jBlock = (cols[j]/numLocalDOF)*numLocalDOFScale
               + colLocalDOF
               - minLocalDOFScale;
        err = CphisMatSet(
                solver->Ascale[scale],
                iBlock,
                jBlock,
                vals[j]
              );
        if (err) {
          numErrors++;
          CPHISCHECKTHREAD(err);
        }
      }
      if (isRowInResidual && (!isErrorCorrectionScale || isColInScale)) {
        // Entry goes to Aresidual.
        iBlock = (i/numLocalDOF)*numLocalDOFResidual + rowLocalDOF;
        jBlock = (cols[j]/numLocalDOF)*numLocalDOFScaleToResidual
               + colLocalDOF
               - minLocalDOFScaleToResidual;
        err = CphisMatSet(
                solver->Aresidual[scale],
                iBlock,
                jBlock,
                vals[j]
              );
        if (err) {
          numErrors++;
          CPHISCHECKTHREAD(err);
        }
      }
      if (isRowInScale && !isColInScale) {
        // Entry goes to Arhs.
        iBlock = (i/numLocalDOF)*numLocalDOFScale
               + rowLocalDOF
               - minLocalDOFScale;
        jBlock = (cols[j]/numLocalDOF)*minLocalDOFScale + colLocalDOF;
        err = CphisMatSet(
                solver->Arhs[scale],
                iBlock,
                jBlock,
                vals[j]
              );
        if (err) {
          numErrors++;
          CPHISCHECKTHREAD(err);
        }
      }
    }
  }
  if (numErrors > 0) {
    CphisSolverCleanupScale(solver, scale);
    CPHISCHECK(CPHIS_ERROR_IN_THREAD);
  }

  // Finalize matrices.
  if (!skipAscale) {
    err = CphisMatFinalize(solver->Ascale[scale]);
    if (err) {
      CphisSolverCleanupScale(solver, scale);
      CPHISCHECK(err);
    }
  }
  if (scale > 0) {
    err = CphisMatFinalize(solver->Aresidual[scale]);
    if (err) {
        CphisSolverCleanupScale(solver, scale);
        CPHISCHECK(err);
    }
  }
  if (isHSSType) {
    err = CphisMatFinalize(solver->Arhs[scale]);
    if (err) {
      CphisSolverCleanupScale(solver, scale);
      CPHISCHECK(err);
    }
  }

  // Set up scale solver.
  err = CphisScaleSolverSetup(
          solver->conf->solvers[scale],
          solver->Ascale[scale]
        );
  if (err) {
    CphisSolverCleanupScale(solver, scale);
    CPHISCHECK(err);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisSolverCreate(
             CphisSolver *solver,
             const CphisConf conf,
             const CphisMat A
           )
{
  CphisError err;

  #ifdef _OPENMP
  const double tStart = omp_get_wtime();
  #endif

  *solver = malloc(sizeof(struct _CphisSolver));
  if (!(*solver)) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  // Make sure that it is always safe to clean up.
  (*solver)->rfull = NULL;
  (*solver)->Ascale = NULL;
  (*solver)->Aresidual = NULL;
  (*solver)->Arhs = NULL;
  (*solver)->r = NULL;
  (*solver)->e = NULL;
  (*solver)->fscale = NULL;
  (*solver)->uscale = NULL;
  (*solver)->urhs = NULL;
  (*solver)->iterCoarseScale = 0;

  err = CphisVecCreate(
          &(*solver)->rfull,
          A->numElements,
          A->elements,
          A->numLocalDOF,
          A->type
        );CPHISCHECK(err);

  // Allocate arrays for matrix and vector handles.
  // The only possible errors here are failed allocations.
  err = CPHIS_FAILED_ALLOC;
  (*solver)->Ascale = malloc(conf->numScales*sizeof(CphisMat));
  if (!(*solver)->Ascale) goto cphis_solver_create_cleanup;
  (*solver)->Aresidual = malloc(conf->numScales*sizeof(CphisMat));
  if (!(*solver)->Aresidual) goto cphis_solver_create_cleanup;
  (*solver)->Arhs = malloc(conf->numScales*sizeof(CphisMat));
  if (!(*solver)->Arhs) goto cphis_solver_create_cleanup;
  (*solver)->r = malloc(conf->numScales*sizeof(CphisVec));
  if (!(*solver)->r) goto cphis_solver_create_cleanup;
  (*solver)->e = malloc(conf->numScales*sizeof(CphisVec));
  if (!(*solver)->e) goto cphis_solver_create_cleanup;
  (*solver)->fscale = malloc(conf->numScales*sizeof(CphisVec));
  if (!(*solver)->fscale) goto cphis_solver_create_cleanup;
  (*solver)->uscale = malloc(conf->numScales*sizeof(CphisVec));
  if (!(*solver)->uscale) goto cphis_solver_create_cleanup;
  (*solver)->urhs = malloc(conf->numScales*sizeof(CphisVec));
  if (!(*solver)->urhs) goto cphis_solver_create_cleanup;

  (*solver)->conf = conf;
  (*solver)->A = A;

  // Set up scales.
  for (int s = 0; s < conf->numScales; s++) {
    err = CphisSolverSetupScale(*solver, s);
    if (err) {
      // Clean up and abort.
      for (s--; s >= 0; s--) {
        CphisSolverCleanupScale(*solver, s);
      }
      goto cphis_solver_create_cleanup;
    }
  }

  // Initialize timers.
  memset((*solver)->timers, 0, CPHIS_NUM_TIMERS*sizeof(double));

  #ifdef _OPENMP
  const double tEnd = omp_get_wtime();
  (*solver)->timers[CPHIS_TIMER_SETUP] += tEnd - tStart;
  #endif

  return CPHIS_SUCCESS;

cphis_solver_create_cleanup:
  CphisVecDestroy((*solver)->rfull);
  free((*solver)->Ascale);
  free((*solver)->Aresidual);
  free((*solver)->Arhs);
  free((*solver)->r);
  free((*solver)->e);
  free((*solver)->fscale);
  free((*solver)->uscale);
  free((*solver)->urhs);
  free(*solver);
  CPHISCHECK(err);
  return err; // Suppress compiler warning (missing return statement).
}

CphisError CphisSolverDestroy(CphisSolver solver)
{
  // Clean up scales.
  for (int s = 0; s < solver->conf->numScales; s++) {
    CphisSolverCleanupScale(solver, s);
  }
  CphisVecDestroy(solver->rfull);
  free(solver);

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverSolveCoarseScale(
                    const CphisSolver solver,
                    const CphisVec f,
                    const CphisVec u
                  )
{
  CphisError err;

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#0: Beginning coarse scale solve\n");
  }

  #ifdef _OPENMP
  const double tStart = omp_get_wtime();
  #endif

  CphisConvergenceFlag flag;
  CphisReal residual;
  int iter;
  err = CphisScaleSolverSolve(
          solver->conf->solvers[0],
          f,
          u,
          &flag,
          &residual,
          &iter
        );CPHISCHECK(err);
  solver->iterCoarseScale += iter;

  #ifdef _OPENMP
  const double tEnd = omp_get_wtime();
  solver->timers[CPHIS_TIMER_COARSE_SOLVE] += tEnd - tStart;
  #endif

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf(
      "#0: Finished coarse scale solve (residual = %e, %d iter.)\n",
      residual,
      iter
    );
    if (flag != CPHIS_CONVERGED) {
      CphisPrintf("#0: The coarse scale solver did not converge!\n");
    }
  }

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverSmoothHSS(
                    const CphisSolver solver,
                    const CphisVec f,
                    CphisVec u,
                    int scale,
                    int isZeroInitialGuess
                  )
{
  CphisError err;

  CphisScalar *fData, *fscaleData, *uData, *uscaleData, *urhsData;
  err = CphisVecGetData(f, &fData);CPHISCHECK(err);
  err = CphisVecGetData(solver->fscale[scale], &fscaleData);CPHISCHECK(err);
  err = CphisVecGetData(u, &uData);CPHISCHECK(err);
  err = CphisVecGetData(solver->uscale[scale], &uscaleData);CPHISCHECK(err);
  err = CphisVecGetData(solver->urhs[scale], &urhsData);CPHISCHECK(err);
  CphisIndex numElements;
  err = CphisVecGetNumElements(f, &numElements);CPHISCHECK(err);
  int numLocalDOFfu, numLocalDOFfuscale, numLocalDOFurhs;
  err = CphisVecGetNumLocalDOF(f, &numLocalDOFfu);CPHISCHECK(err);
  err = CphisVecGetNumLocalDOF(
          solver->fscale[scale],
          &numLocalDOFfuscale
        );CPHISCHECK(err);
  err = CphisVecGetNumLocalDOF(
          solver->urhs[scale],
          &numLocalDOFurhs
        );CPHISCHECK(err);

  if (!isZeroInitialGuess) {
    // We need to move the d.o.f. that are not smoothed on this scale to the
    // right-hand side.
    // First, split u into uscale and urhs.
    #pragma omp parallel for
    for (CphisIndex k = 0; k < numElements; k++) {
      for (int l = 0; l < numLocalDOFfu; l++) {
        const CphisIndex row = k*numLocalDOFfu + l;
        if (l < numLocalDOFurhs) {
          // Entry goes to urhs.
          urhsData[k*numLocalDOFurhs + l] = uData[row];
        }
        else {
          // Entry goes to uscale.
          uscaleData[k*numLocalDOFfuscale + l - numLocalDOFurhs] = uData[row];
        }
      }
    }
    // Compute right-hand side.
    err = CphisMatVec(
            solver->Arhs[scale],
            solver->urhs[scale],
            solver->fscale[scale]
          );CPHISCHECK(err);
    for (CphisIndex k = 0; k < numElements; k++) {
      for (int l = numLocalDOFurhs; l < numLocalDOFfu; l++) {
        fscaleData[k*numLocalDOFfuscale + l - numLocalDOFurhs]
          = fData[k*numLocalDOFfu + l]
          - fscaleData[k*numLocalDOFfuscale + l - numLocalDOFurhs];
      }
    }
  }
  else {
    err = CphisVecSetAll(solver->uscale[scale], 0.0);CPHISCHECK(err);
    // Get fscale.
    for (CphisIndex k = 0; k < numElements; k++) {
      for (int l = numLocalDOFurhs; l < numLocalDOFfu; l++) {
        fscaleData[k*numLocalDOFfuscale + l - numLocalDOFurhs]
          = fData[k*numLocalDOFfu + l];
      }
    }
  }

  // Apply the smoother to the smaller system.
  err = CphisScaleSolverSolve(
          solver->conf->solvers[scale],
          solver->fscale[scale],
          solver->uscale[scale],
          NULL,
          NULL,
          NULL
        );CPHISCHECK(err);

  // Use values from uscale to update u.
  #pragma omp parallel for
  for (CphisIndex k = 0; k < numElements; k++) {
    for (int l = numLocalDOFurhs; l < numLocalDOFfu; l++) {
      uData[k*numLocalDOFfu + l]
        = uscaleData[k*numLocalDOFfuscale + l - numLocalDOFurhs];
    }
  }

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverPresmooth(
                    const CphisSolver solver,
                    const CphisVec f,
                    CphisVec u,
                    int scale
                  )
{
  CphisError err;

  if (solver->conf->nu1 == 0) return CPHIS_SUCCESS;

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#%d: Presmoothing\n", scale);
  }

  #ifdef _OPENMP
  const double tStart = omp_get_wtime();
  #endif

  // Prepare smoother.
  err = CphisScaleSolverSetTol(
          solver->conf->solvers[scale],
          0.0
        );CPHISCHECK(err);
  err = CphisScaleSolverSetMaxIter(
          solver->conf->solvers[scale],
          solver->conf->nu1
        );CPHISCHECK(err);

  const int isHSSType = solver->conf->scales[2*scale] != 0;
  if (isHSSType) {
    const int isErrorCorrectionScale = scale < solver->conf->numScales - 1;
    err = CphisSolverSmoothHSS(
            solver,
            f,
            u,
            scale,
            isErrorCorrectionScale
          );CPHISCHECK(err);
  }
  else {
    // On standard p-multigrid scales, all we have to do is smooth call the
    // smoother.
    err = CphisScaleSolverSolve(
            solver->conf->solvers[scale],
            f,
            u,
            NULL,
            NULL,
            NULL
          );CPHISCHECK(err);
  }

  #ifdef _OPENMP
  const double tEnd = omp_get_wtime();
  solver->timers[CPHIS_TIMER_PRESMOOTH] += tEnd - tStart;
  #endif

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverComputeRestrictedResidual(
                    const CphisSolver solver,
                    const CphisVec f,
                    const CphisVec u,
                    int scale
                  )
{
  CphisError err;

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#%d: Computing restricted residual\n", scale);
  }

  #ifdef _OPENMP
  const double tStart = omp_get_wtime();
  #endif

  const int isHSSType = solver->conf->scales[2*scale] != 0;
  const int isErrorCorrectionScale = scale < solver->conf->numScales - 1;
  if (isHSSType && isErrorCorrectionScale) {
    // Compute residual directly from uscale.
    err = CphisMatVec(
            solver->Aresidual[scale],
            solver->uscale[scale],
            solver->r[scale]
          );CPHISCHECK(err);
  }
  else {
    // We need to incorporate all of u in the residual computation.
    err = CphisMatVec(
            solver->Aresidual[scale],
            u,
            solver->r[scale]
          );CPHISCHECK(err);
  }

  CphisScalar *rData, *fData;
  err = CphisVecGetData(solver->r[scale], &rData);CPHISCHECK(err);
  err = CphisVecGetData(f, &fData);CPHISCHECK(err);
  CphisIndex numElements;
  err = CphisVecGetNumElements(f, &numElements);CPHISCHECK(err);
  int numLocalDOFr, numLocalDOFf;
  err = CphisVecGetNumLocalDOF(solver->r[scale], &numLocalDOFr);CPHISCHECK(err);
  err = CphisVecGetNumLocalDOF(f, &numLocalDOFf);CPHISCHECK(err);
  #pragma omp parallel for
  for (CphisIndex k = 0; k < numElements; k++) {
    for (int l = 0; l < numLocalDOFr; l++) {
      rData[k*numLocalDOFr + l] = fData[k*numLocalDOFf + l]
                                - rData[k*numLocalDOFr + l];
    }
  }

  #ifdef _OPENMP
  const double tEnd = omp_get_wtime();
  solver->timers[CPHIS_TIMER_RESIDUAL] += tEnd - tStart;
  #endif

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverProlongate(
                    const CphisSolver solver,
                    CphisVec u, 
                    int scale
                  )
{
  CphisError err;

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#%d: Prolongation\n", scale);
  }

  #ifdef _OPENMP
  const double tStart = omp_get_wtime();
  #endif

  CphisScalar *eData, *uData;
  err = CphisVecGetData(solver->e[scale], &eData);CPHISCHECK(err);
  err = CphisVecGetData(u, &uData);CPHISCHECK(err);
  CphisIndex numElements;
  err = CphisVecGetNumElements(u, &numElements);CPHISCHECK(err);
  int numLocalDOFe, numLocalDOFu;
  err = CphisVecGetNumLocalDOF(solver->e[scale], &numLocalDOFe);CPHISCHECK(err);
  err = CphisVecGetNumLocalDOF(u, &numLocalDOFu);CPHISCHECK(err);
  #pragma omp parallel for
  for (CphisIndex k = 0; k < numElements; k++) {
    for (int l = 0; l < numLocalDOFe; l++) {
      uData[k*numLocalDOFu + l] += eData[k*numLocalDOFe + l];
    }
  }

  #ifdef _OPENMP
  const double tEnd = omp_get_wtime();
  solver->timers[CPHIS_TIMER_PROLONGATION] += tEnd - tStart;
  #endif

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverPostsmooth(
                    const CphisSolver solver,
                    const CphisVec f,
                    CphisVec u,
                    int scale
                  )
{
  CphisError err;

  if (solver->conf->nu2 == 0) return CPHIS_SUCCESS;

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#%d: Postsmoothing\n", scale);
  }

  #ifdef _OPENMP
  const double tStart = omp_get_wtime();
  #endif

  // Prepare smoother.
  err = CphisScaleSolverSetMaxIter(
          solver->conf->solvers[scale],
          solver->conf->nu2
        );CPHISCHECK(err);

  const int isHSSType = solver->conf->scales[2*scale] != 0;
  if (isHSSType) {
    err = CphisSolverSmoothHSS(solver, f, u, scale, 0);CPHISCHECK(err);
  }
  else {
    // On standard p-multigrid scales, all we have to do is smooth call the
    // smoother.
    err = CphisScaleSolverSolve(
            solver->conf->solvers[scale],
            f,
            u,
            NULL,
            NULL,
            NULL
          );CPHISCHECK(err);
  }

  #ifdef _OPENMP
  const double tEnd = omp_get_wtime();
  solver->timers[CPHIS_TIMER_POSTSMOOTH] += tEnd - tStart;
  #endif

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverCycleV(
                    const CphisSolver solver,
                    const CphisVec f,
                    CphisVec u,
                    int scale
                  )
{
  CphisError err;

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#%d: Entering\n", scale);
  }

  if (scale == 0) {
    err = CphisSolverSolveCoarseScale(solver, f, u);CPHISCHECK(err);
    if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
      CphisPrintf("#0: Leaving\n");
    }
    return CPHIS_SUCCESS;
  }

  err = CphisSolverPresmooth(solver, f, u, scale);CPHISCHECK(err);
  err = CphisSolverComputeRestrictedResidual(
          solver,
          f,
          u,
          scale
        );CPHISCHECK(err);

  // A V-cycle is just a simple recursion.
  err = CphisVecSetAll(solver->e[scale], 0.0);CPHISCHECK(err);
  err = CphisSolverCycleV(
          solver,
          solver->r[scale],
          solver->e[scale],
          scale - 1
        );CPHISCHECK(err);

  err = CphisSolverProlongate(solver, u, scale);CPHISCHECK(err);
  err = CphisSolverPostsmooth(solver, f, u, scale);CPHISCHECK(err);

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#%d: Leaving\n", scale);
  }

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverCycleW(
                    const CphisSolver solver,
                    const CphisVec f,
                    CphisVec u,
                    int scale
                  )
{
  CphisError err;

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#%d: Entering\n", scale);
  }

  if (scale == 0) {
    err = CphisSolverSolveCoarseScale(solver, f, u);CPHISCHECK(err);
    if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
      CphisPrintf("#0: Leaving\n");
    }
    return CPHIS_SUCCESS;
  }

  err = CphisSolverPresmooth(solver, f, u, scale);CPHISCHECK(err);
  err = CphisSolverComputeRestrictedResidual(
          solver,
          f,
          u,
          scale
        );CPHISCHECK(err);

  // A W-cycle recurses twice.
  err = CphisVecSetAll(solver->e[scale], 0.0);CPHISCHECK(err);
  err = CphisSolverCycleW(
          solver,
          solver->r[scale],
          solver->e[scale],
          scale - 1
        );CPHISCHECK(err);
  err = CphisSolverCycleW(
          solver,
          solver->r[scale],
          solver->e[scale],
          scale - 1
        );CPHISCHECK(err);

  err = CphisSolverProlongate(solver, u, scale);CPHISCHECK(err);
  err = CphisSolverPostsmooth(solver, f, u, scale);CPHISCHECK(err);

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#%d: Leaving\n", scale);
  }

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverCycleF(
                    const CphisSolver solver,
                    const CphisVec f,
                    CphisVec u,
                    int scale
                  )
{
    CphisError err;

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#%d: Entering\n", scale);
  }

  if (scale == 0) {
    err = CphisSolverSolveCoarseScale(solver, f, u);CPHISCHECK(err);
    if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
      CphisPrintf("#0: Leaving\n");
    }
    return CPHIS_SUCCESS;
  }

  err = CphisSolverPresmooth(solver, f, u, scale);CPHISCHECK(err);
  err = CphisSolverComputeRestrictedResidual(
          solver,
          f,
          u,
          scale
        );CPHISCHECK(err);

  // Like the W-cycle, the F-cycle recurses twice.
  // However, the second recursion is a simple V-cycle.
  err = CphisVecSetAll(solver->e[scale], 0.0);CPHISCHECK(err);
  err = CphisSolverCycleF(
          solver,
          solver->r[scale],
          solver->e[scale],
          scale - 1
        );CPHISCHECK(err);
  err = CphisSolverCycleV(
          solver,
          solver->r[scale],
          solver->e[scale],
          scale - 1
        );CPHISCHECK(err);

  err = CphisSolverProlongate(solver, u, scale);CPHISCHECK(err);
  err = CphisSolverPostsmooth(solver, f, u, scale);CPHISCHECK(err);

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintf("#%d: Leaving\n", scale);
  }

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverCycle(
                    const CphisSolver solver,
                    const CphisVec f,
                    CphisVec u
                  )
{
  CphisError err;

  // Enter recursion based on cycle type.
  switch (solver->conf->cycle) {
  case CPHIS_CYCLE_V:
    err = CphisSolverCycleV(
            solver,
            f,
            u,
            solver->conf->numScales - 1
          );CPHISCHECK(err);
    break;
  case CPHIS_CYCLE_W:
    err = CphisSolverCycleW(
            solver,
            f,
            u,
            solver->conf->numScales - 1
          );CPHISCHECK(err);
    break;
  case CPHIS_CYCLE_F:
    err = CphisSolverCycleF(
            solver,
            f,
            u,
            solver->conf->numScales - 1
          );CPHISCHECK(err);
    break;
  default:
    CPHISCHECK(CPHIS_UNKNOWN_TYPE);
    break;
  }

  return CPHIS_SUCCESS;
}

static CphisError CphisSolverPrintTimers(const CphisSolver solver)
{
  CphisPrintf("CPHIS Timers:\n");
  #ifdef _OPENMP
  CphisPrintf(
    "       Setup:                              %.3f\n",
    solver->timers[CPHIS_TIMER_SETUP]
  );
  CphisPrintf(
    "       Solve:                              %.3f\n",
    solver->timers[CPHIS_TIMER_SOLVER]
  );
  CphisPrintf(
    "       Presmooth:                          %.3f\n",
    solver->timers[CPHIS_TIMER_PRESMOOTH]
  );
  CphisPrintf(
    "       Postsmooth:                         %.3f\n",
    solver->timers[CPHIS_TIMER_POSTSMOOTH]
  );
  CphisPrintf(
    "       Coarse scale solves:                %.3f\n",
    solver->timers[CPHIS_TIMER_COARSE_SOLVE]
  );
  CphisPrintf(
    "       Multigrid residual & restriction:   %.3f\n",
    solver->timers[CPHIS_TIMER_RESIDUAL]
  );
  CphisPrintf(
    "       Prolongation:                       %.3f\n",
    solver->timers[CPHIS_TIMER_PROLONGATION]
  );
  CphisPrintf(
    "       System residual check:              %.3f\n",
    solver->timers[CPHIS_TIMER_SYSTEM_RESIDUAL]
  );

  // Compute and print time delta, i.e., the difference between the total solver
  // time and the sum of the individual timers. This time difference is caused
  // by terminal output etc.
  double tDelta = 0.0;
  for (int i = 0; i < CPHIS_NUM_TIMERS; i++) {
    if (i == CPHIS_TIMER_SETUP || i == CPHIS_TIMER_SOLVER) continue;
    tDelta += solver->timers[i];
  }
  tDelta = solver->timers[CPHIS_TIMER_SOLVER] - tDelta;
  CphisPrintf("       Time delta:                         %.3f\n", tDelta);
  #else
  CphisPrintf("       Timers require OpenMP to be enabled!\n");
  #endif

  return CPHIS_SUCCESS;
}

CphisError CphisSolverSolve(
             const CphisSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           )
{
  CphisError err;
  int k = 0;
  CphisReal rNorm, r0Norm;

  #ifdef _OPENMP
  const double tStart = omp_get_wtime();
  #endif

  // Compute initial residual norm.
  err = CphisMatVec(solver->A, x, solver->rfull);CPHISCHECK(err);
  err = CphisVecAXPY(-1.0, b, solver->rfull);CPHISCHECK(err);
  err = CphisVecNorm2(solver->rfull, &r0Norm);CPHISCHECK(err);
  rNorm = r0Norm;

  #ifdef _OPENMP
  solver->timers[CPHIS_TIMER_SYSTEM_RESIDUAL] += omp_get_wtime() - tStart;
  #endif

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
    CphisPrintHline(1);
    CphisPrintf("CPHIS: Beginning solve\n");
    CphisPrintHline(1);
  }

  while (1) {
    // Check residual norm.
    if (rNorm/r0Norm < solver->conf->tol) {
      if (flag) *flag = CPHIS_CONVERGED;
      break;
    }

    // Check for maximum number of iterations.
    if (k >= solver->conf->maxIter) {
      if (flag) *flag = CPHIS_MAX_ITER;
      break;
    }

    if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
      if (k > 0) CphisPrintHline(0);
      CphisPrintf("Beginning iteration #%d\n", k + 1);
    }

    // Perform one full cycle.
    err = CphisSolverCycle(solver, b, x);CPHISCHECK(err);

    // Compute residual norm.
    #ifdef _OPENMP
    const double tStartResidual = omp_get_wtime();
    #endif
    err = CphisMatVec(solver->A, x, solver->rfull);CPHISCHECK(err);
    err = CphisVecAXPY(-1.0, b, solver->rfull);CPHISCHECK(err);
    err = CphisVecNorm2(solver->rfull, &rNorm);CPHISCHECK(err);
    #ifdef _OPENMP
    const double tEndResidual = omp_get_wtime();
    solver->timers[CPHIS_TIMER_SYSTEM_RESIDUAL]
      += tEndResidual - tStartResidual;
    #endif

    if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
      CphisPrintf(
        "End of iteration #%d (residual = %e)\n",
        k + 1,
        rNorm/r0Norm
      );
    }

    k++;
  }

  #ifdef _OPENMP
  const double tEnd = omp_get_wtime();
  solver->timers[CPHIS_TIMER_SOLVER] += tEnd - tStart;
  #endif

  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
      CphisPrintHline(1);
  }
  if (solver->conf->verbosity >= CPHIS_VERBOSITY_SUMMARY) {
    if (rNorm/r0Norm < solver->conf->tol) {
      CphisPrintf(
        "CPHIS: Solver converged to desired tolerance of %e!\n",
        solver->conf->tol
      );
    }
    if (k >= solver->conf->maxIter) {
      CphisPrintf("CPHIS: Solver reached the maximum number of iterations!\n");
    }
    CphisPrintf("       Rel. residual:                      %e\n", rNorm/r0Norm);
    CphisPrintf("       #Iterations:                        %d\n", k);
    CphisPrintf(
      "       #Iterations (coarse scale solver):  %d\n",
      solver->iterCoarseScale
    );
    CphisPrintf(
      "       #Iterations (smoothers):            %d\n",
      k*(solver->conf->numScales - 1)*(solver->conf->nu1 + solver->conf->nu2)
    );

    CphisPrintHline(0);

    err = CphisSolverPrintTimers(solver);CPHISCHECK(err);
  }
  if (solver->conf->verbosity >= CPHIS_VERBOSITY_DETAILED) {
      CphisPrintHline(1);
  }

  if (residual) *residual = rNorm/r0Norm;
  if (iter) *iter = k;

  return CPHIS_SUCCESS;
}
