// mat_default.c - default matrix implementation

#include <cphis.h>
#include <linalg.h>
#include <stdlib.h>
#include <string.h>

CphisError CphisMatCreate_default(
             CphisMat *mat,
             CphisIndex numElements,
             int numLocalDOF
           )
{
  (*mat)->mat = malloc(sizeof(struct _CphisMat_default));
  if (!(*mat)->mat) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  struct _CphisMat_default *matInternal = (*mat)->mat;

  // Create buffers.
  const CphisIndex numRows = numElements*numLocalDOF;
  matInternal->bufferCapacities = malloc(numRows*sizeof(CphisIndex));
  matInternal->bufferSizes = malloc(numRows*sizeof(CphisIndex));
  matInternal->colBuffers = malloc(numRows*sizeof(CphisIndex*));
  matInternal->valBuffers = malloc(numRows*sizeof(CphisScalar*));
  if (
    !matInternal->bufferCapacities ||
    !matInternal->bufferSizes ||
    !matInternal->colBuffers ||
    !matInternal->valBuffers
  ) {
    free(matInternal->bufferCapacities);
    free(matInternal->bufferSizes);
    free(matInternal->colBuffers);
    free(matInternal->valBuffers);
    free(matInternal);
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }
  for (CphisIndex i = 0; i < numRows; i++) {
    matInternal->bufferCapacities[i] = 0;
    matInternal->bufferSizes[i] = 0;
    matInternal->colBuffers[i] = NULL;
    matInternal->valBuffers[i] = NULL;
  }

  // Set the CRS pointers to NULL so it is always safe to call free().
  matInternal->rows = NULL;
  matInternal->cols = NULL;
  matInternal->vals = NULL;

  return CPHIS_SUCCESS;
}

CphisError CphisMatDestroy_default(CphisMat mat)
{
  struct _CphisMat_default *matInternal = mat->mat;

  // Clean up buffers if necessary.
  const CphisIndex numRows = mat->numElements*mat->numLocalDOF;
  if (matInternal->bufferSizes) {
    for (CphisIndex i = 0; i < numRows; i++) {
      free(matInternal->colBuffers[i]);
      free(matInternal->valBuffers[i]);
    }
    free(matInternal->bufferCapacities);
    free(matInternal->bufferSizes);
    free(matInternal->colBuffers);
    free(matInternal->valBuffers);
  }

  // Clean up CRS data if necessary.
  if (matInternal->rows) {
    free(matInternal->rows);
    free(matInternal->cols);
    free(matInternal->vals);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatVec_default(const CphisMat A, const CphisVec x, CphisVec y)
{
  CPHISCHECKMATVECCOMPAT(A, x, y);

  const struct _CphisMat_default *matInternal = A->mat;
  const CphisScalar *xData = x->vec;
  CphisScalar *yData = y->vec;

  const CphisIndex numRows = A->numElements*A->numLocalDOF;
  #pragma omp parallel for
  for (CphisIndex i = 0; i < numRows; i++) {
    CphisScalar result = 0.0;
    for (
      size_t j = matInternal->rows[i];
      j < matInternal->rows[i + 1];
      j++
    ) {
      result += matInternal->vals[j]*xData[matInternal->cols[j]];
    }
    yData[i] = result;
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatGetData_default(
             const CphisMat mat,
             CphisIndex row,
             CphisIndex **cols,
             CphisScalar **vals,
             CphisIndex *numEntries
           )
{
  const struct _CphisMat_default *matInternal = mat->mat;

  const size_t j = matInternal->rows[row];
  *cols = &(matInternal->cols[j]);
  *vals = &(matInternal->vals[j]);
  *numEntries = matInternal->rows[row + 1] - j;

  return CPHIS_SUCCESS;
}

CphisError CphisMatSet_default(
             CphisMat mat,
             CphisIndex i,
             CphisIndex j,
             CphisScalar aij
           )
{
  struct _CphisMat_default *matInternal = mat->mat;

  // Create buffer if it does not exist.
  if (matInternal->bufferCapacities[i] == 0) {
    matInternal->colBuffers[i] = malloc(sizeof(CphisIndex));
    matInternal->valBuffers[i] = malloc(sizeof(CphisScalar));
    if (!matInternal->colBuffers[i] || !matInternal->valBuffers[i]) {
      free(matInternal->colBuffers[i]);
      free(matInternal->valBuffers[i]);
      CPHISCHECK(CPHIS_FAILED_ALLOC);
    }
    matInternal->bufferCapacities[i] = 1;
  }

  // Increase buffer capacity if necessary.
  const CphisIndex size = matInternal->bufferSizes[i];
  const CphisIndex capacity = matInternal->bufferCapacities[i];
  if (size == capacity) {
    void *ptr;
    // Increase column buffer.
    ptr = realloc(matInternal->colBuffers[i], 2*capacity*sizeof(CphisIndex));
    if (!ptr) {
      CPHISCHECK(CPHIS_FAILED_ALLOC);
    }
    matInternal->colBuffers[i] = ptr;
    // Increase value buffer.
    ptr = realloc(matInternal->valBuffers[i], 2*capacity*sizeof(CphisScalar));
    if (!ptr) {
      CPHISCHECK(CPHIS_FAILED_ALLOC);
    }
    matInternal->valBuffers[i] = ptr;

    matInternal->bufferCapacities[i] *= 2;
  }

  // Add matrix entry.
  matInternal->colBuffers[i][size] = j;
  matInternal->valBuffers[i][size] = aij;
  matInternal->bufferSizes[i]++;

  return CPHIS_SUCCESS;
}

CphisError CphisMatFinalize_default(CphisMat mat)
{
  struct _CphisMat_default *matInternal = mat->mat;

  // We begin by counting the matrix entries.
  size_t nnz = 0;
  const CphisIndex numRows = mat->numElements*mat->numLocalDOF;
  #pragma omp parallel for reduction(+:nnz)
  for (CphisIndex i = 0; i < numRows; i++) {
    nnz += matInternal->bufferSizes[i];
  }

  // Allocate memory for CRS data.
  matInternal->rows = malloc((numRows + 1)*sizeof(size_t));
  matInternal->cols = malloc(nnz*sizeof(CphisIndex));
  matInternal->vals = malloc(nnz*sizeof(CphisScalar));
  if (
    !matInternal->rows ||
    !matInternal->cols ||
    !matInternal->vals
  ) {
    free(matInternal->rows);
    free(matInternal->cols);
    free(matInternal->vals);
  }

  // Populate row pointers.
  matInternal->rows[0] = 0;
  for (CphisIndex i = 1; i < numRows; i++) {
    matInternal->rows[i] = matInternal->rows[i - 1]
                         + matInternal->bufferSizes[i - 1];
  }
  matInternal->rows[numRows] = nnz;

  // Populate CRS arrays and clean up buffers.
  #pragma omp parallel for
  for (CphisIndex i = 0; i < numRows; i++) {
    const size_t offset = matInternal->rows[i];
    memcpy(
      &(matInternal->cols[offset]),
      matInternal->colBuffers[i],
      matInternal->bufferSizes[i]*sizeof(CphisIndex)
    );
    memcpy(
      &(matInternal->vals[offset]),
      matInternal->valBuffers[i],
      matInternal->bufferSizes[i]*sizeof(CphisScalar)
    );
    free(matInternal->colBuffers[i]);
    free(matInternal->valBuffers[i]);
  }
  free(matInternal->bufferCapacities);
  free(matInternal->bufferSizes);
  free(matInternal->colBuffers);
  free(matInternal->valBuffers);
  // Prevent double free.
  matInternal->bufferCapacities = NULL;
  matInternal->bufferSizes = NULL;
  matInternal->colBuffers = NULL;
  matInternal->valBuffers = NULL;

  return CPHIS_SUCCESS;
}
