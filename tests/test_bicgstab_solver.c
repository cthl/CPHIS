#include <cphis.h>
#include <stdlib.h>

int main()
{
  CphisError err;

  // The desired tolerance and the expected results
  const CphisReal tol = 1.0e-8;
  const CphisReal refResidual = 9.583e-9;
  const int refIter = 133;

  // Get linear system for the test.
  const CphisIndex n = 640;
  const int numLocalDOF = 10;
  CphisMat A;
  CphisVec u, f, r;
  err = CphisMatFromMatrixMarket(
          &A,
          numLocalDOF,
          "../Atest.mtx"
        );CPHISCHECK(err);
  err = CphisVecFromMatrixMarket(
          &f,
          numLocalDOF,
          "../ftest.mtx"
        );CPHISCHECK(err);
  err = CphisVecCreate(
          &u,
          n,
          NULL,
          numLocalDOF,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);
  err = CphisVecCreate(
          &r,
          n,
          NULL,
          numLocalDOF,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);

  // Compute initial residual norm.
  CphisScalar r0Norm;
  err = CphisMatVec(A, u, r);CPHISCHECK(err);
  err = CphisVecAXPY(-1.0, f, r);CPHISCHECK(err);
  err = CphisVecNorm2(r, &r0Norm);CPHISCHECK(err);

  // Create solver and solve the system.
  CphisScaleSolver solver;
  err = CphisScaleSolverCreate(
          &solver,
          CPHIS_SCALE_SOLVER_BICGSTAB
        );CPHISCHECK(err);
  err = CphisScaleSolverSetTol(solver, tol);CPHISCHECK(err);
  err = CphisScaleSolverSetup(solver, A);CPHISCHECK(err);

  CphisConvergenceFlag flag;
  int iter;
  err = CphisScaleSolverSolve(solver, f, u, &flag, NULL, &iter);CPHISCHECK(err);

  // Check convergence flag.
  if (flag != CPHIS_CONVERGED) {
    CPHISCHECK(CPHIS_TEST_FAILED);
  }

  // Check number of iterations.
  if (iter != refIter) {
    CPHISCHECK(CPHIS_TEST_FAILED);
  }

  // Explicit residual check
  CphisReal rNorm;
  err = CphisMatVec(A, u, r);CPHISCHECK(err);
  err = CphisVecAXPY(-1.0, f, r);CPHISCHECK(err);
  err = CphisVecNorm2(r, &rNorm);CPHISCHECK(err);
  if (rNorm/r0Norm > refResidual) {
    CPHISCHECK(CPHIS_TEST_FAILED);
  }

  // Clean up.
  err = CphisScaleSolverDestroy(solver);CPHISCHECK(err);
  err = CphisMatDestroy(A);CPHISCHECK(err);
  err = CphisVecDestroy(u);CPHISCHECK(err);
  err = CphisVecDestroy(f);CPHISCHECK(err);
  err = CphisVecDestroy(r);CPHISCHECK(err);

  return CPHIS_SUCCESS;
}
