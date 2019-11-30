#include <cphis.h>
#include <stdlib.h>

int main()
{
  CphisError err;

  // The desired tolerance and the expected results
  const CphisReal tol = 1.0e-8;
  const CphisReal refResidual = 9.352e-9;
  const int refIter = 124;

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

  // Create the coarse scale solver.
  CphisScaleSolver coarseScaleSolver;
  err = CphisScaleSolverCreate(
          &coarseScaleSolver,
          CPHIS_SCALE_SOLVER_LU
        );CPHISCHECK(err);

  // Create CPHIS solver and solve the system.
  CphisConf conf;
  CphisSolver solver;
  const int scales[] = {0, 0, 0, 3, 4, 9};
  err = CphisConfCreate(&conf, numLocalDOF, 3, scales);CPHISCHECK(err);
  err = CphisConfSetVerbosity(conf, CPHIS_VERBOSITY_ERRORS);CPHISCHECK(err);
  err = CphisConfSetTolRel(conf, tol);CPHISCHECK(err);
  err = CphisConfSetNu(conf, 4, 4);CPHISCHECK(err);
  err = CphisConfSetSmoothers(
          conf,
          CPHIS_SCALE_SOLVER_BLOCK_JACOBI
        );CPHISCHECK(err);
  err = CphisConfSetScaleSolver(conf, 0, coarseScaleSolver);CPHISCHECK(err);
  err = CphisSolverCreate(&solver, conf, A);CPHISCHECK(err);

  CphisConvergenceFlag flag;
  int iter;
  err = CphisSolverSolve(solver, f, u, &flag, NULL, &iter);CPHISCHECK(err);

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
  err = CphisSolverDestroy(solver);CPHISCHECK(err);
  err = CphisConfDestroy(conf);CPHISCHECK(err);
  err = CphisScaleSolverDestroy(coarseScaleSolver);CPHISCHECK(err);
  err = CphisMatDestroy(A);CPHISCHECK(err);
  err = CphisVecDestroy(u);CPHISCHECK(err);
  err = CphisVecDestroy(f);CPHISCHECK(err);
  err = CphisVecDestroy(r);CPHISCHECK(err);

  return CPHIS_SUCCESS;
}
