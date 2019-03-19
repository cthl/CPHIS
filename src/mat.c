// mat.c - generic matrix interface implementation

#include <cphis.h>
#include <linalg.h>
#include <stdlib.h>

CphisError CphisMatCreate(
             CphisMat *mat,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOF,
             CphisBackendType type
           )
{
  *mat = malloc(sizeof(struct _CphisMat));
  if (!(*mat)) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  (*mat)->type = type;
  (*mat)->numElements = numElements;
  (*mat)->elements = NULL;
  (*mat)->numLocalDOF = numLocalDOF;
  (*mat)->finalized = 0;

  CphisError err;
  switch ((*mat)->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisMatCreate_default(mat, numElements, numLocalDOF);
    break;
  default:
    err = CPHIS_UNKNOWN_TYPE;
    break;
  }

  if (err) {
    free(*mat);
  }

  CPHISCHECK(err);
  return CPHIS_SUCCESS;
}

CphisError CphisMatDestroy(CphisMat mat)
{
  CphisError err;
  switch (mat->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisMatDestroy_default(mat);
    break;
  default:
    err = CPHIS_UNKNOWN_TYPE;
    break;
  }

  free(mat);

  CPHISCHECK(err);
  return CPHIS_SUCCESS;
}

CphisError CphisMatGetNumElements(const CphisMat mat, CphisIndex *numElements)
{
  *numElements = mat->numElements;
  return CPHIS_SUCCESS;
}

CphisError CphisMatGetNumLocalDOF(const CphisMat mat, int *numLocalDOF)
{
  *numLocalDOF = mat->numLocalDOF;
  return CPHIS_SUCCESS;
}

CphisError CphisMatVec(const CphisMat A, const CphisVec x, CphisVec y)
{
  if (!A->finalized) {
    CPHISCHECK(CPHIS_INVALID_STATE);
  }

  CphisError err;
  switch (A->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisMatVec_default(A, x, y);CPHISCHECK(err);
    break;
  default:
    CPHISCHECK(CPHIS_UNKNOWN_TYPE);
    break;
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatGetData(
             const CphisMat mat,
             CphisIndex row,
             CphisIndex **cols,
             CphisScalar **vals,
             CphisIndex *numEntries
           )
{
  if (!mat->finalized) {
    CPHISCHECK(CPHIS_INVALID_STATE);
  }

  CphisError err;
  switch (mat->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisMatGetData_default(
            mat,
            row,
            cols,
            vals,
            numEntries
          );CPHISCHECK(err);
    break;
  default:
    CPHISCHECK(CPHIS_UNKNOWN_TYPE);
    break;
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatSet(
             CphisMat mat,
             CphisIndex i,
             CphisIndex j,
             CphisScalar aij
           )
{
  if (mat->finalized) {
    CPHISCHECK(CPHIS_INVALID_STATE);
  }

  CphisError err;
  switch (mat->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisMatSet_default(mat, i, j, aij);CPHISCHECK(err);
    break;
  default:
    CPHISCHECK(CPHIS_UNKNOWN_TYPE);
    break;
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatFinalize(CphisMat mat)
{
  CphisError err;
  switch (mat->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisMatFinalize_default(mat);CPHISCHECK(err);
    break;
  default:
    CPHISCHECK(CPHIS_UNKNOWN_TYPE);
    break;
  }

  mat->finalized = 1;

  return CPHIS_SUCCESS;
}