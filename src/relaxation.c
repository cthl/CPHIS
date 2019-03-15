// relaxation.c - implementation of relaxation methods

#include <cphis.h>
#include <linalg.h>
#include <scale_solver.h>
#include <stdlib.h>

CphisError CphisScaleSolverDestroy_Jacobi(CphisScaleSolver solver)
{
  struct _CphisScaleSolverData_Jacobi *data = solver->data;

  if (data) {
    free(data->invD);
    CphisVecDestroy(data->r);
    free(data);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverDestroy_BlockJacobi(CphisScaleSolver solver)
{
  struct _CphisScaleSolverData_BlockJacobi *data = solver->data;

  if (data) {
    free(data->blocks);
    CphisVecDestroy(data->r);
    CphisVecDestroy(data->z);
    free(data);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSetup_Jacobi(CphisScaleSolver solver)
{
  // Smoother has not been implemented for distributed setting yet.
  if (solver->A->type != CPHIS_BACKEND_DEFAULT) {
    CPHISCHECK(CPHIS_INCOMPATIBLE);
  }

  solver->data = malloc(sizeof(struct _CphisScaleSolverData_Jacobi));
  if (!solver->data) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  struct _CphisScaleSolverData_Jacobi *data = solver->data;

  const CphisIndex numRows = solver->A->numElements*solver->A->numLocalDOF;
  data->invD = malloc(numRows*sizeof(CphisScalar));
  if (!data->invD) {
    free(solver->data);
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  CphisError err;
  err = CphisVecCreate(
          &data->r,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          CPHIS_BACKEND_DEFAULT
        );
  if (err) {
    free(data->invD);
    free(solver->data);;
    CPHISCHECK(err);
  }

  // Extract diagonal.
  CphisIndex *cols;
  CphisScalar *vals;
  CphisIndex numEntries;
  for (CphisIndex i = 0; i < numRows; i++) {
    err = CphisMatGetData(solver->A, i, &cols, &vals, &numEntries);
    if (err) {
      free(data->invD);
      CphisVecDestroy(data->r);
      free(solver->data);
      CPHISCHECK(err);
    }
    // Find diagonal entry.
    for (CphisIndex j = 0; j < numEntries; j++) {
      if (i == cols[j]) {
        data->invD[i] = 1.0/vals[j];
        break;
      }
    }
  }

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSetup_BlockJacobi(CphisScaleSolver solver)
{
  // Smoother has not been implemented for distributed setting yet.
  if (solver->A->type != CPHIS_BACKEND_DEFAULT) {
    CPHISCHECK(CPHIS_INCOMPATIBLE);
  }

  solver->data = malloc(sizeof(struct _CphisScaleSolverData_BlockJacobi));
  if (!solver->data) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  struct _CphisScaleSolverData_BlockJacobi *data = solver->data;

  const int numLocalDOF = solver->A->numLocalDOF;
  const int blockSize = numLocalDOF*numLocalDOF;
  const CphisIndex numRows = solver->A->numElements*numLocalDOF;

  // Allocate memory for LU factorizations of diagonal blocks.
  // This must be initialized with zero!
  data->blocks = calloc(solver->A->numElements*blockSize, sizeof(CphisScalar));
  if (!data->blocks) {
    free(solver->data);
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  // Extract block diagonal.
  CphisError err;
  CphisIndex *cols;
  CphisScalar *vals;
  CphisIndex numEntries;
  for (CphisIndex i = 0; i < numRows; i++) {
    err = CphisMatGetData(solver->A, i, &cols, &vals, &numEntries);
    if (err) {
      free(data->blocks);
      free(solver->data);
      CPHISCHECK(err);
    }
    const CphisIndex blockIndex = i/numLocalDOF;
    // Find block diagonal entries.
    for (CphisIndex j = 0; j < numEntries; j++) {
      if (blockIndex == cols[j]/numLocalDOF) {
        // Compute row and column index within block.
        const CphisIndex iBlock = i%numLocalDOF;
        const CphisIndex jBlock = cols[j]%numLocalDOF;
        // Write entry to (dense) block matrix.
        data->blocks[blockIndex*blockSize + iBlock*numLocalDOF + jBlock]
          = vals[j];
      }
    }
  }

  // Compute LU factorizations.
  for (CphisIndex i = 0; i < solver->A->numElements; i++) {
    CphisLUFactorize(numLocalDOF, &data->blocks[i*numLocalDOF*numLocalDOF]);
  }

  err = CphisVecCreate(
          &data->r,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          CPHIS_BACKEND_DEFAULT
        );
  if (err) {
    free(data->blocks);
    free(solver->data);;
    CPHISCHECK(err);
  }
  err = CphisVecCreate(
          &data->z,
          solver->A->numElements,
          solver->A->elements,
          solver->A->numLocalDOF,
          CPHIS_BACKEND_DEFAULT
        );
  if (err) {
    CphisVecDestroy(data->r);
    free(data->blocks);
    free(solver->data);;
    CPHISCHECK(err);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSolve_Jacobi(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           )
{
  CphisError err;
  CphisScalar rNorm, r0Norm;
  int k;
  const CphisIndex numRows = x->numElements*x->numLocalDOF;
  struct _CphisScaleSolverData_Jacobi *data = solver->data;

  CphisScalar *xData, *rData;
  err = CphisVecGetData(x, &xData);CPHISCHECK(err);
  err = CphisVecGetData(data->r, &rData);CPHISCHECK(err);

  // Compute initial residual norm.
  err = CphisMatVec(solver->A, x, data->r);CPHISCHECK(err);
  err = CphisVecAXPBY(1.0, b, -1.0, data->r);CPHISCHECK(err);
  err = CphisVecNorm2(data->r, &r0Norm);CPHISCHECK(err);
  rNorm = r0Norm;

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

    // Perform Jacobi iteration.
    #pragma omp parallel for
    for (CphisIndex i = 0; i < numRows; i++) {
      xData[i] += data->invD[i]*rData[i];
    }

    // Update residual and residual norm.
    err = CphisMatVec(solver->A, x, data->r);CPHISCHECK(err);
    err = CphisVecAXPBY(1.0, b, -1.0, data->r);CPHISCHECK(err);
    err = CphisVecNorm2(data->r, &rNorm);CPHISCHECK(err);

    k++;
  }

  if (residual) *residual = rNorm/r0Norm;
  if (iter) *iter = k;

  return CPHIS_SUCCESS;
}

CphisError CphisScaleSolverSolve_BlockJacobi(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           )
{
  CphisError err;
  CphisScalar rNorm, r0Norm;
  int k;
  const int numLocalDOF = x->numLocalDOF;
  const int blockSize = numLocalDOF*numLocalDOF;
  struct _CphisScaleSolverData_BlockJacobi *data = solver->data;

  CphisScalar *xData, *rData, *zData;
  err = CphisVecGetData(x, &xData);CPHISCHECK(err);
  err = CphisVecGetData(data->r, &rData);CPHISCHECK(err);
  err = CphisVecGetData(data->z, &zData);CPHISCHECK(err);

  // Compute initial residual norm.
  err = CphisMatVec(solver->A, x, data->r);CPHISCHECK(err);
  err = CphisVecAXPBY(1.0, b, -1.0, data->r);CPHISCHECK(err);
  err = CphisVecNorm2(data->r, &r0Norm);CPHISCHECK(err);
  rNorm = r0Norm;

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

    // Perform Jacobi iteration.
    #pragma omp parallel for
    for (CphisIndex i = 0; i < x->numElements; i++) {
      CphisLUSolve(
        numLocalDOF,
        &data->blocks[i*blockSize],
        &rData[i*numLocalDOF],
        &zData[i*numLocalDOF]
      );
      for (int j = 0; j < numLocalDOF; j++) {
        xData[i*numLocalDOF + j] += zData[i*numLocalDOF + j];
      }
    }

    // Update residual and residual norm.
    err = CphisMatVec(solver->A, x, data->r);CPHISCHECK(err);
    err = CphisVecAXPBY(1.0, b, -1.0, data->r);CPHISCHECK(err);
    err = CphisVecNorm2(data->r, &rNorm);CPHISCHECK(err);

    k++;
  }

  if (residual) *residual = rNorm/r0Norm;
  if (iter) *iter = k;

  return CPHIS_SUCCESS;
}
