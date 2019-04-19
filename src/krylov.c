// krylov.c - Krylov solver implementation

#include <cphis.h>
#include <linalg.h>
#include <scale_solver.h>
#include <stdlib.h>
#include <math.h>

CphisError CphisScaleSolverDestroy_CG(CphisScaleSolver solver)
{
  struct _CphisScaleSolverData_CG *data = solver->data;

  if (data) {
    CphisVecDestroy(data->r);
    CphisVecDestroy(data->p);
    CphisVecDestroy(data->z);
    free(data);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverDestroy_BiCGStab(CphisScaleSolver solver)
{
  struct _CphisScaleSolverData_BiCGStab *data = solver->data;

  if (data) {
    CphisVecDestroy(data->r0);
    CphisVecDestroy(data->r);
    CphisVecDestroy(data->p);
    CphisVecDestroy(data->v);
    CphisVecDestroy(data->h);
    CphisVecDestroy(data->s);
    CphisVecDestroy(data->t);
    free(data);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSetup_CG(CphisScaleSolver solver)
{
  solver->data = malloc(sizeof(struct _CphisScaleSolverData_CG));
  if (!solver->data) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  struct _CphisScaleSolverData_CG *data = solver->data;

  // Make sure that cleanup is safe at all times.
  data->r = NULL;
  data->p = NULL;
  data->z = NULL;

  CphisError err;
  err = CphisVecCreate(
          &data->r,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          solver->A->type,
          NULL
        );
  if (err) goto cphis_scale_solver_setup_cg_cleanup;
  err = CphisVecCreate(
          &data->p,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          solver->A->type,
          NULL
        );
  if (err) goto cphis_scale_solver_setup_cg_cleanup;
  err = CphisVecCreate(
          &data->z,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          solver->A->type,
          NULL
        );
  if (err) goto cphis_scale_solver_setup_cg_cleanup;

  return CPHIS_SUCCESS;

cphis_scale_solver_setup_cg_cleanup:
  if (data->r) CphisVecDestroy(data->r);
  if (data->p) CphisVecDestroy(data->p);
  if (data->z) CphisVecDestroy(data->z);
  free(solver->data);
  CPHISCHECK(err);
  return err; // Suppress compiler warning (missing return statement).
}

CphisError CphisScaleSolverSetup_BiCGStab(CphisScaleSolver solver)
{
  if (solver->A->type != CPHIS_BACKEND_DEFAULT) {
    CPHISCHECK(CPHIS_INCOMPATIBLE);
  }

  solver->data = malloc(sizeof(struct _CphisScaleSolverData_BiCGStab));
  if (!solver->data) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  struct _CphisScaleSolverData_BiCGStab *data = solver->data;

  // Make sure that cleanup is safe at all times.
  data->r0 = NULL;
  data->r = NULL;
  data->p = NULL;
  data->v = NULL;
  data->h = NULL;
  data->s = NULL;
  data->t = NULL;

  CphisError err;
  err = CphisVecCreate(
          &data->r0,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          solver->A->type,
          NULL
        );
  if (err) goto cphis_scale_solver_setup_bicgstab_cleanup;
  err = CphisVecCreate(
          &data->r,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          solver->A->type,
          NULL
        );
  if (err) goto cphis_scale_solver_setup_bicgstab_cleanup;
  err = CphisVecCreate(
          &data->p,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          solver->A->type,
          NULL
        );
  if (err) goto cphis_scale_solver_setup_bicgstab_cleanup;
  err = CphisVecCreate(
          &data->v,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          solver->A->type,
          NULL
        );
  if (err) goto cphis_scale_solver_setup_bicgstab_cleanup;
  err = CphisVecCreate(
          &data->h,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          solver->A->type,
          NULL
        );
  if (err) goto cphis_scale_solver_setup_bicgstab_cleanup;
  err = CphisVecCreate(
          &data->s,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          solver->A->type,
          NULL
        );
  if (err) goto cphis_scale_solver_setup_bicgstab_cleanup;
  err = CphisVecCreate(
          &data->t,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          solver->A->type,
          NULL
        );
  if (err) goto cphis_scale_solver_setup_bicgstab_cleanup;

  return CPHIS_SUCCESS;

cphis_scale_solver_setup_bicgstab_cleanup:
  if (data->r0) CphisVecDestroy(data->r0);
  if (data->r) CphisVecDestroy(data->r);
  if (data->p) CphisVecDestroy(data->p);
  if (data->v) CphisVecDestroy(data->v);
  if (data->h) CphisVecDestroy(data->h);
  if (data->s) CphisVecDestroy(data->s);
  if (data->t) CphisVecDestroy(data->t);
  free(solver->data);
  CPHISCHECK(err);
  return err; // Suppress compiler warning (missing return statement).
}

CphisError CphisScaleSolverSolve_CG(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           )
{
  CphisError err;
  CphisScalar alpha, beta, rTr, pTAp;
  CphisReal r0Norm;
  int k;
  struct _CphisScaleSolverData_CG *data = solver->data;

  // Compute initial residual and its norm.
  err = CphisMatVec(solver->A, x, data->z);CPHISCHECK(err);
  err = CphisVecAssign(b, data->r);CPHISCHECK(err);
  err = CphisVecAXPY(-1.0, data->z, data->r);CPHISCHECK(err);
  err = CphisVecNorm2(data->r, &r0Norm);
  err = CphisVecAssign(data->r, data->p);CPHISCHECK(err);
  err = CphisVecDot(data->r, data->r, &rTr);CPHISCHECK(err);

  k = 0;
  while (1) {
    // Check residual norm.
    if (sqrt(rTr)/r0Norm < solver->tol) {
      if (flag) *flag = CPHIS_CONVERGED;
      break;
    }

    // Check for maximum number of iterations.
    if (k >= solver->maxIter) {
      if (flag) *flag = CPHIS_MAX_ITER;
      break;
    }

    // Perform CG iteration.
    err = CphisMatVec(solver->A, data->p, data->z);CPHISCHECK(err);
    err = CphisVecDot(data->p, data->z, &pTAp);CPHISCHECK(err);
    alpha = rTr/pTAp;
    err = CphisVecAXPY(alpha, data->p, x);CPHISCHECK(err);
    err = CphisVecAXPY(-alpha, data->z, data->r);CPHISCHECK(err);
    beta = 1.0/rTr;
    err = CphisVecDot(data->r, data->r, &rTr);CPHISCHECK(err);
    beta *= rTr;
    err = CphisVecScale(data->p, beta);CPHISCHECK(err);
    err = CphisVecAXPY(1.0, data->r, data->p);CPHISCHECK(err);

    k++;
  }

  if (residual) *residual = sqrt(rTr)/r0Norm;
  if (iter) *iter = k;

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSolve_BiCGStab(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           )
{
  CphisError err;
  CphisScalar rho, omega, alpha, beta, r0Tv, tTs, tTt;
  CphisReal rNorm, r0Norm;
  int k;
  struct _CphisScaleSolverData_BiCGStab *data = solver->data;

  // Compute initial residual and prepare solve.
  err = CphisMatVec(solver->A, x, data->r0);CPHISCHECK(err);
  err = CphisVecAssign(b, data->r);CPHISCHECK(err);
  err = CphisVecAXPY(-1.0, data->r0, data->r);CPHISCHECK(err);
  err = CphisVecAssign(data->r, data->r0);CPHISCHECK(err);
  err = CphisVecNorm2(data->r0, &r0Norm);CPHISCHECK(err);
  rNorm = r0Norm;
  rho = alpha = omega = 1.0;
  err = CphisVecSetAll(data->p, 0.0);CPHISCHECK(err);
  err = CphisVecSetAll(data->v, 0.0);CPHISCHECK(err);

  k = 0;
  while (1) {
    // Check residual norm.
    if (rNorm/r0Norm < solver->tol) {
      if (flag) *flag = CPHIS_CONVERGED;
      break;
    }

    // Check for maximum number of iterations.
    if (k >= solver->maxIter) {
      if (flag) *flag = CPHIS_MAX_ITER;
      break;
    }

    beta = alpha/(rho*omega);
    err = CphisVecDot(data->r0, data->r, &rho);CPHISCHECK(err);
    beta *= rho;

    err = CphisVecScale(data->p, beta);CPHISCHECK(err);
    err = CphisVecAXPY(1.0, data->r, data->p);CPHISCHECK(err);
    err = CphisVecAXPY(-beta*omega, data->v, data->p);CPHISCHECK(err);

    err = CphisMatVec(solver->A, data->p, data->v);CPHISCHECK(err);

    err = CphisVecDot(data->r0, data->v, &r0Tv);CPHISCHECK(err);
    alpha = rho/r0Tv;

    err = CphisVecAssign(x, data->h);CPHISCHECK(err);
    err = CphisVecAXPY(alpha, data->p, data->h);CPHISCHECK(err);

    err = CphisVecAssign(data->r, data->s);CPHISCHECK(err);
    err = CphisVecAXPY(-alpha, data->v, data->s);CPHISCHECK(err);

    err = CphisMatVec(solver->A, data->s, data->t);CPHISCHECK(err);

    err = CphisVecDot(data->t, data->s, &tTs);CPHISCHECK(err);
    err = CphisVecDot(data->t, data->t, &tTt);CPHISCHECK(err);
    omega = tTs/tTt;

    err = CphisVecAssign(data->h, x);CPHISCHECK(err);
    err = CphisVecAXPY(omega, data->s, x);CPHISCHECK(err);

    err = CphisVecAssign(data->s, data->r);CPHISCHECK(err);
    err = CphisVecAXPY(-omega, data->t, data->r);CPHISCHECK(err);

    // Compute new residual norm.
    err = CphisVecNorm2(data->r, &rNorm);CPHISCHECK(err);

    k++;
  }

  if (residual) *residual = rNorm/r0Norm;
  if (iter) *iter = k;

  return CPHIS_SUCCESS;
}
