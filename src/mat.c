// mat.c - generic matrix interface implementation

#include <cphis.h>
#include <linalg.h>
#ifdef CPHIS_HAVE_TPETRA
#include <cphis_tpetra.h>
#endif
#include <stdlib.h>

CphisError CphisMatCreate(
             CphisMat *mat,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOFRange,
             int numLocalDOFDomain,
             CphisBackendType type,
             void *from
           )
{
  *mat = malloc(sizeof(struct _CphisMat));
  if (!(*mat)) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  (*mat)->type = type;
  (*mat)->numElements = numElements;
  (*mat)->elements = elements;
  (*mat)->numLocalDOFRange = numLocalDOFRange;
  (*mat)->numLocalDOFDomain = numLocalDOFDomain;
  (*mat)->finalized = 0;
  (*mat)->owned = (from == NULL);
  (*mat)->bufferSize = 0;
  (*mat)->colBuffer = NULL;
  (*mat)->valBuffer = NULL;

  CphisError err;
  if (from) {
    (*mat)->mat = from;
    err = CphisMatFinalize(*mat);
  }
  else {
    switch ((*mat)->type) {
    case CPHIS_BACKEND_DEFAULT:
      err = CphisMatCreate_default(mat, numElements, numLocalDOFRange);
      break;
    case CPHIS_BACKEND_TPETRA:
      #ifdef CPHIS_HAVE_TPETRA
      err = CphisMatCreate_Tpetra(
              mat,
              numElements,
              elements,
              numLocalDOFRange
            );
      #else
      err = CPHIS_MISSING_BACKEND;
      #endif
      break;
    default:
      err = CPHIS_UNKNOWN_TYPE;
      break;
    }
  }

  if (err) {
    free(*mat);
    CPHISCHECK(err);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatDestroy(CphisMat mat)
{
  CphisError err = CPHIS_SUCCESS;

  if (mat->owned) {
    switch (mat->type) {
    case CPHIS_BACKEND_DEFAULT:
      err = CphisMatDestroy_default(mat);
      break;
    case CPHIS_BACKEND_TPETRA:
      #ifdef CPHIS_HAVE_TPETRA
      err = CphisMatDestroy_Tpetra(mat);
      #else
      err = CPHIS_MISSING_BACKEND;
      #endif
      break;
    default:
      err = CPHIS_UNKNOWN_TYPE;
      break;
    }
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
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisMatVec_Tpetra(A, x, y);CPHISCHECK(err);
    #else
    CPHISCHECK(CPHIS_MISSING_BACKEND);
    #endif
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
             const CphisIndex **cols,
             const CphisScalar **vals,
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
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisMatGetData_Tpetra(
            mat,
            row,
            cols,
            vals,
            numEntries
          );CPHISCHECK(err);
    #else
    CPHISCHECK(CPHIS_MISSING_BACKEND);
    #endif
    break;
  default:
    CPHISCHECK(CPHIS_UNKNOWN_TYPE);
    break;
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatGetInternal(const CphisMat mat, void **matInternal)
{
  *matInternal = mat->mat;
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
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisMatSet_Tpetra(mat, i, j, aij);CPHISCHECK(err);
    #else
    CPHISCHECK(CPHIS_MISSING_BACKEND);
    #endif
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
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisMatFinalize_Tpetra(mat);CPHISCHECK(err);
    #else
    CPHISCHECK(CPHIS_MISSING_BACKEND);
    #endif
    break;
  default:
    CPHISCHECK(CPHIS_UNKNOWN_TYPE);
    break;
  }

  mat->finalized = 1;

  return CPHIS_SUCCESS;
}
