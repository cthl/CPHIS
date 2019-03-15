// lu.c - implementation of LU factorization and LU solver

#include <cphis.h>
#include <linalg.h>
#include <scale_solver.h>
#include <string.h>
#include <stdlib.h>

#define IDX(N, I, J) ((N)*(I) + (J))

void CphisLUFactorize(CphisIndex n, CphisScalar *A)
{
  for (CphisIndex i = 0; i < n; i++) {
    for (CphisIndex j = i + 1; j < n; j++) {
      const CphisScalar coeff = A[IDX(n, j, i)]/A[IDX(n, i, i)];
      A[IDX(n, j, i)] = coeff;
      for (CphisIndex k = i + 1; k < n; k++) {
        A[IDX(n, j, k)] -= coeff*A[IDX(n, i, k)];
      }
    }
  }
}

void CphisLUSolve(
       CphisIndex n,
       const CphisScalar *LU,
       const CphisScalar *b,
       CphisScalar *x
     )
{
  // Forward substitution
  for (CphisIndex i = 0; i < n; i++) {
    x[i] = b[i];
    for (CphisIndex j = 0; j < i; j++) {
      x[i] -= LU[IDX(n, i, j)]*x[j];
    }
  }

  // Backward substitution
  for (CphisIndex i = n - 1; i >= 0; i--) {
    for (CphisIndex j = i + 1; j < n; j++) {
      x[i] -= LU[IDX(n, i, j)]*x[j];
    }
    x[i] /= LU[IDX(n, i, i)];
  }
}

CphisError CphisScaleSolverDestroy_LU(CphisScaleSolver solver)
{
  if (solver->data) {
    free(solver->data);
  }
  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSetup_LU(CphisScaleSolver solver)
{
  // Create a dense copy of the system matrix.
  const CphisIndex numRows = solver->A->numElements*solver->A->numLocalDOF;
  solver->data = calloc(numRows*numRows, sizeof(CphisScalar));
  if (!solver->data) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  CphisError err;
  CphisIndex *cols;
  CphisScalar *vals;
  CphisIndex numEntries;
  CphisScalar *denseA = solver->data;
  for (CphisIndex i = 0; i < numRows; i++) {
    err = CphisMatGetData(solver->A, i, &cols, &vals, &numEntries);
    if (err) {
      free(denseA);
      CPHISCHECK(err);
    }
    for (CphisIndex j = 0; j < numEntries; j++) {
      denseA[numRows*i + cols[j]] = vals[j];
    }
  }

  // Compute LU factorization.
  CphisLUFactorize(numRows, denseA);

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSolve_LU(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           )
{
  CphisError err;
  const CphisIndex numRows = solver->A->numElements*solver->A->numLocalDOF;
  const CphisScalar *LU = solver->data;
  CphisScalar *bData;
  CphisScalar *xData;

  err = CphisVecGetData(b, &bData);CPHISCHECK(err);
  err = CphisVecGetData(x, &xData);CPHISCHECK(err);

  CphisLUSolve(numRows, LU, bData, xData);

  if (flag) *flag = CPHIS_CONVERGED;
  if (residual) *residual = 0.0;
  if (iter) *iter = 1;

  return CPHIS_SUCCESS;
}
