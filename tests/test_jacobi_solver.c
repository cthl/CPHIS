#include <cphis.h>
#include "test_poisson2d.h"
#include <stdlib.h>

int main()
{
  CphisError err;

  const CphisReal tol = 1.0e-3;

  // Get linear system for the test.
  const CphisIndex n = 100;
  const CphisIndex n2 = n*n;
  CphisMat A;
  CphisVec u, ux, f, r;
  err = CphisMatCreate(
          &A,
          n2,
          NULL,
          1,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);
  err = CphisVecCreate(
          &u,
          n2,
          NULL,
          1,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);
  err = CphisVecCreate(
          &ux,
          n2,
          NULL,
          1,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);
  err = CphisVecCreate(
          &f,
          n2,
          NULL,
          1,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);
  err = CphisVecCreate(
          &r,
          n2,
          NULL,
          1,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);
  err = getPoisson2dSystem(n, A, f, u, ux);

  // Compute initial residual norm.
  CphisScalar r0Norm;
  err = CphisMatVec(A, u, r);CPHISCHECK(err);
  err = CphisVecAXPY(-1.0, f, r);CPHISCHECK(err);
  err = CphisVecNorm2(r, &r0Norm);CPHISCHECK(err);

  // Create solver and solve the system.
  CphisScaleSolver solver;
  err = CphisScaleSolverCreate(
          &solver,
          CPHIS_SCALE_SOLVER_JACOBI
        );CPHISCHECK(err);
  err = CphisScaleSolverSetTol(solver, tol);
  err = CphisScaleSolverSetup(solver, A);CPHISCHECK(err);
  CphisConvergenceFlag flag;
  err = CphisScaleSolverSolve(solver, f, u, &flag, NULL, NULL);CPHISCHECK(err);
  if (flag != CPHIS_CONVERGED) {
    return CPHIS_TEST_FAILED;
  }

  // Check error and residual norm.
  CphisReal eNorm, uxNorm, rNorm;
  err = CphisVecNorm2(ux, &uxNorm);CPHISCHECK(err);
  err = CphisVecAXPY(-1.0, u, ux);CPHISCHECK(err);
  err = CphisVecNorm2(ux, &eNorm);CPHISCHECK(err);
  if (eNorm/uxNorm > 5.0e-1) {
    return CPHIS_TEST_FAILED;
  }
  err = CphisMatVec(A, u, r);CPHISCHECK(err);
  err = CphisVecAXPY(-1.0, f, r);CPHISCHECK(err);
  err = CphisVecNorm2(r, &rNorm);CPHISCHECK(err);
  if (rNorm/r0Norm > tol) {
    return CPHIS_TEST_FAILED;
  }

  // Clean up.
  err = CphisScaleSolverDestroy(solver);CPHISCHECK(err);
  err = CphisMatDestroy(A);CPHISCHECK(err);
  err = CphisVecDestroy(u);CPHISCHECK(err);
  err = CphisVecDestroy(ux);CPHISCHECK(err);
  err = CphisVecDestroy(f);CPHISCHECK(err);
  err = CphisVecDestroy(r);CPHISCHECK(err);

  return CPHIS_SUCCESS;
}
