// scale_solver.c - scale solver interface implementation

#include <cphis.h>
#include <scale_solver.h>
#include <linalg.h>
#include <stdlib.h>

CphisError CphisScaleSolverCreate(
             CphisScaleSolver *solver,
             CphisScaleSolverType type
           )
{
  *solver = malloc(sizeof(struct _CphisScaleSolver));
  if (!(*solver)) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  (*solver)->type = type;
  (*solver)->tol = 1.0e-6;
  (*solver)->maxIter = 1e6;
  (*solver)->omega = 1.0;
  (*solver)->A = NULL;
  (*solver)->data = NULL;

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverDestroy(CphisScaleSolver solver)
{
  CphisError err;
  switch (solver->type) {
  case CPHIS_SCALE_SOLVER_LU:
    err = CphisScaleSolverDestroy_LU(solver);
    break;
  case CPHIS_SCALE_SOLVER_JACOBI:
    err = CphisScaleSolverDestroy_Jacobi(solver);
    break;
  case CPHIS_SCALE_SOLVER_BLOCK_JACOBI:
    err = CphisScaleSolverDestroy_BlockJacobi(solver);
    break;
  case CPHIS_SCALE_SOLVER_CG:
    err = CphisScaleSolverDestroy_CG(solver);
    break;
  case CPHIS_SCALE_SOLVER_BICGSTAB:
    err = CphisScaleSolverDestroy_BiCGStab(solver);
    break;
  case CPHIS_SCALE_SOLVER_EXTERNAL:
    err = CphisScaleSolverDestroy_External(solver);
    break;
  default:
    err = CPHIS_UNKNOWN_TYPE;
    break;
  }

  free(solver);

  CPHISCHECK(err);
  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSetMaxIter(CphisScaleSolver solver, int maxIter)
{
  solver->maxIter = maxIter;
  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSetTol(CphisScaleSolver solver, CphisReal tol)
{
  solver->tol = tol;
  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSetOmega(CphisScaleSolver solver, CphisReal omega)
{
  solver->omega = omega;
  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSetup(CphisScaleSolver solver, const CphisMat A)
{
  if (A->numLocalDOFRange != A->numLocalDOFDomain) {
    CPHISCHECK(CPHIS_INCOMPATIBLE);
  }
  if (solver->A) {
    CPHISCHECK(CPHIS_INVALID_STATE);
  }
  solver->A = A;

  CphisError err;
  switch (solver->type) {
  case CPHIS_SCALE_SOLVER_LU:
    err = CphisScaleSolverSetup_LU(solver);CPHISCHECK(err);
    break;
  case CPHIS_SCALE_SOLVER_JACOBI:
    err = CphisScaleSolverSetup_Jacobi(solver);CPHISCHECK(err);
    break;
  case CPHIS_SCALE_SOLVER_BLOCK_JACOBI:
    err = CphisScaleSolverSetup_BlockJacobi(solver);CPHISCHECK(err);
    break;
  case CPHIS_SCALE_SOLVER_CG:
    err = CphisScaleSolverSetup_CG(solver);CPHISCHECK(err);
    break;
  case CPHIS_SCALE_SOLVER_BICGSTAB:
    err = CphisScaleSolverSetup_BiCGStab(solver);CPHISCHECK(err);
    break;
  case CPHIS_SCALE_SOLVER_EXTERNAL:
    err = CphisScaleSolverSetup_External(solver);CPHISCHECK(err);
    break;
  default:
    CPHISCHECK(CPHIS_UNKNOWN_TYPE);
    break;
  }

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSolve(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           )
{
  if (!solver->A) {
    CPHISCHECK(CPHIS_UNINITIALIZED);
  }

  CphisError err;
  switch (solver->type) {
  case CPHIS_SCALE_SOLVER_LU:
    err = CphisScaleSolverSolve_LU(
            solver,
            b,
            x,
            flag,
            residual,
            iter
          );CPHISCHECK(err);
    break;
  case CPHIS_SCALE_SOLVER_JACOBI:
    err = CphisScaleSolverSolve_Jacobi(
            solver,
            b,
            x,
            flag,
            residual,
            iter
          );CPHISCHECK(err);
    break;
  case CPHIS_SCALE_SOLVER_BLOCK_JACOBI:
    err = CphisScaleSolverSolve_BlockJacobi(
            solver,
            b,
            x,
            flag,
            residual,
            iter
          );CPHISCHECK(err);
    break;
  case CPHIS_SCALE_SOLVER_CG:
    err = CphisScaleSolverSolve_CG(
            solver,
            b,
            x,
            flag,
            residual,
            iter
          );CPHISCHECK(err);
    break;
  case CPHIS_SCALE_SOLVER_BICGSTAB:
    err = CphisScaleSolverSolve_BiCGStab(
            solver,
            b,
            x,
            flag,
            residual,
            iter
          );CPHISCHECK(err);
    break;
  case CPHIS_SCALE_SOLVER_EXTERNAL:
    err = CphisScaleSolverSolve_External(
            solver,
            b,
            x,
            flag,
            residual,
            iter
          );CPHISCHECK(err);
    break;
  default:
    CPHISCHECK(CPHIS_UNKNOWN_TYPE);
    break;
  }

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSetExternal(
             CphisScaleSolver solver,
             CphisError (*setupFunc)(const CphisMat, void*),
             CphisError (*solveFunc)(
                          const CphisVec,
                          CphisVec,
                          CphisConvergenceFlag*,
                          CphisReal*,
                          int*,
                          void*
                        ),
             void *context
           )
{
  solver->data = malloc(sizeof(struct _CphisScaleSolverData_External));
  if (!solver->data) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  struct _CphisScaleSolverData_External *data = solver->data;

  data->setupFunc = setupFunc;
  data->solveFunc = solveFunc;
  data->context = context;

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverDestroy_External(CphisScaleSolver solver)
{
  struct _CphisScaleSolverData_External *data = solver->data;

  if (data) {
    free(data);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSetup_External(CphisScaleSolver solver)
{
  CphisError err;

  struct _CphisScaleSolverData_External *data = solver->data;

  err = data->setupFunc(solver->A, data->context);

  return err;
}

CphisError CphisScaleSolverSolve_External(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           )
{
  CphisError err;

  struct _CphisScaleSolverData_External *data = solver->data;

  err = data->solveFunc(b, x, flag, residual, iter, data->context);

  return err;
}
