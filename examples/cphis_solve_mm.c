// cphis_solve_mm.c - import and solve a system in matrix market format

#include <cphis.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  if (argc < 10 || (argc - 8)%2 != 0) {
    fprintf(
      stderr,
      "Usage: ./cphis_solve_mm mat_file vec_file tol maxiter "
      "nu1 nu2 delta scales...\n"
    );
    return -1;
  }

  // Determine the scales.
  int numScales = (argc - 8)/2;
  int *scales = malloc(2*numScales*sizeof(int));
  if (!scales) CPHISCHECK(CPHIS_FAILED_ALLOC);
  for (int s = 0; s < numScales; s++) {
    scales[2*s] = atoi(argv[8 + 2*s]);
    scales[2*s + 1] = atoi(argv[8 + 2*s + 1]);
  }
  const int numLocalDOF = scales[2*(numScales - 1) + 1] + 1;

  // Get tolerance and maximum number of iterations.
  const CphisReal tol = atof(argv[3]);
  const int maxIter = atoi(argv[4]);

  // Get smoother iterations and coarse scale tolerance.
  const int nu1 = atoi(argv[5]);
  const int nu2 = atoi(argv[6]);
  const CphisReal delta = atof(argv[7]);

  // Load linear system.
  CphisError err;
  CphisMat A;
  CphisVec x, b;
  CphisIndex numElements;
  err = CphisMatFromMatrixMarket(&A, numLocalDOF, argv[1]);CPHISCHECK(err);
  err = CphisVecFromMatrixMarket(&b, numLocalDOF, argv[2]);CPHISCHECK(err);
  err = CphisMatGetNumElements(A, &numElements);CPHISCHECK(err);
  err = CphisVecCreate(
          &x,
          numElements,
          NULL,
          numLocalDOF,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);
  err = CphisVecSetAll(x, 0.0);CPHISCHECK(err);

  // Create solver and solve linear system.
  CphisScaleSolver coarseScaleSolver;
  err = CphisScaleSolverCreate(
          &coarseScaleSolver,
          CPHIS_SCALE_SOLVER_BICGSTAB
        );CPHISCHECK(err);
  err = CphisScaleSolverSetTol(coarseScaleSolver, delta);CPHISCHECK(err);
  err = CphisScaleSolverSetMaxIter(coarseScaleSolver, 1000);CPHISCHECK(err);

  // Create CPHIS solver and solve the system.
  CphisConf conf;
  CphisSolver solver;
  err = CphisConfCreate(&conf, numLocalDOF, numScales, scales);CPHISCHECK(err);
  err = CphisConfSetTol(conf, tol);CPHISCHECK(err);
  err = CphisConfSetMaxIter(conf, maxIter);CPHISCHECK(err);
  err = CphisConfSetNu(conf, nu1, nu2);CPHISCHECK(err);
  err = CphisConfSetSmoothers(
          conf,
          CPHIS_SCALE_SOLVER_BLOCK_JACOBI
        );CPHISCHECK(err);
  err = CphisConfSetScaleSolver(conf, 0, coarseScaleSolver);CPHISCHECK(err);
  err = CphisSolverCreate(&solver, conf, A);CPHISCHECK(err);

  err = CphisSolverSolve(solver, b, x, NULL, NULL, NULL);CPHISCHECK(err);

  // Clean up and exit.
  CphisSolverDestroy(solver);
  CphisConfDestroy(conf);
  free(scales);
  CphisScaleSolverDestroy(coarseScaleSolver);
  CphisMatDestroy(A);
  CphisVecDestroy(x);
  CphisVecDestroy(b);
  return 0;
}
