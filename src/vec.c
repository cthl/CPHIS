// vec.c - generic vector interface implementation

#include <cphis.h>
#include <linalg.h>
#ifdef CPHIS_HAVE_TPETRA
#include <cphis_tpetra.h>
#endif
#include <stdlib.h>

CphisError CphisVecCreate(
             CphisVec *vec,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOF,
             CphisBackendType type,
             void *from
           )
{
  *vec = malloc(sizeof(struct _CphisVec));
  if (!(*vec)) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  (*vec)->type = type;
  (*vec)->numElements = numElements;
  (*vec)->elements = NULL;
  (*vec)->numLocalDOF = numLocalDOF;
  (*vec)->owned = (from == NULL);

  if (from) {
    (*vec)->vec = from;
  }
  else {
    CphisError err;
    switch (type) {
    case CPHIS_BACKEND_DEFAULT:
      err = CphisVecCreate_default(vec, numElements, numLocalDOF);
      break;
    case CPHIS_BACKEND_TPETRA:
      #ifdef CPHIS_HAVE_TPETRA
      err = CphisVecCreate_Tpetra(vec, numElements, elements, numLocalDOF);
      #else
      err = CPHIS_MISSING_BACKEND;
      #endif
      break;
    default:
      err = CPHIS_UNKNOWN_TYPE;
      break;
    }
    if (err) {
      free(*vec);
      CPHISCHECK(err);
    }
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecDestroy(CphisVec vec)
{
  CphisError err = CPHIS_SUCCESS;

  if (vec->owned) {
    switch (vec->type) {
    case CPHIS_BACKEND_DEFAULT:
      err = CphisVecDestroy_default(vec);
      break;
    case CPHIS_BACKEND_TPETRA:
      #ifdef CPHIS_HAVE_TPETRA
      err = CphisVecDestroy_Tpetra(vec);
      #else
      err = CPHIS_MISSING_BACKEND;
      #endif
      break;
    default:
      err = CPHIS_UNKNOWN_TYPE;
      break;
    }
  }

  free(vec);

  CPHISCHECK(err);
  return CPHIS_SUCCESS;
}

CphisError CphisVecGetNumElements(const CphisVec vec, CphisIndex *numElements)
{
  *numElements = vec->numElements;
  return CPHIS_SUCCESS;
}

CphisError CphisVecGetNumLocalDOF(const CphisVec vec, int *numLocalDOF)
{
  *numLocalDOF = vec->numLocalDOF;
  return CPHIS_SUCCESS;
}

CphisError CphisVecNorm2(const CphisVec x, CphisReal *norm2)
{
  CphisError err;
  switch (x->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisVecNorm2_default(x, norm2);CPHISCHECK(err);
    break;
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisVecNorm2_Tpetra(x, norm2);CPHISCHECK(err);
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

CphisError CphisVecDot(const CphisVec x, const CphisVec y, CphisScalar *dot)
{
  CPHISCHECKVECCOMPAT(x, y);

  CphisError err;
  switch (x->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisVecDot_default(x, y, dot);CPHISCHECK(err);
    break;
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisVecDot_Tpetra(x, y, dot);CPHISCHECK(err);
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

CphisError CphisVecAXPY(CphisScalar a, const CphisVec x, CphisVec y)
{
  CPHISCHECKVECCOMPAT(x, y);

  CphisError err;
  switch (x->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisVecAXPY_default(a, x, y);CPHISCHECK(err);
    break;
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisVecAXPY_Tpetra(a, x, y);CPHISCHECK(err);
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

CphisError CphisVecAXPBY(
             CphisScalar a,
             const CphisVec x,
             CphisScalar b,
             CphisVec y
           )
{
  CPHISCHECKVECCOMPAT(x, y);

  CphisError err;
  switch (x->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisVecAXPBY_default(a, x, b, y);CPHISCHECK(err);
    break;
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisVecAXPBY_Tpetra(a, x, b, y);CPHISCHECK(err);
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

CphisError CphisVecScale(CphisVec x, CphisScalar a)
{
  CphisError err;
  switch (x->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisVecScale_default(x, a);CPHISCHECK(err);
    break;
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisVecScale_Tpetra(x, a);CPHISCHECK(err);
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

CphisError CphisVecSetAll(CphisVec vec, CphisScalar val)
{
  CphisError err;
  switch (vec->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisVecSetAll_default(vec, val);CPHISCHECK(err);
    break;
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisVecSetAll_Tpetra(vec, val);CPHISCHECK(err);
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

CphisError CphisVecAssign(const CphisVec x, CphisVec y)
{
  CPHISCHECKVECCOMPAT(x, y);

  CphisError err;
  switch (x->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisVecAssign_default(x, y);CPHISCHECK(err);
    break;
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisVecAssign_Tpetra(x, y);CPHISCHECK(err);
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

CphisError CphisVecGetData(const CphisVec vec, CphisScalar **data)
{
  CphisError err;
  switch (vec->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisVecGetData_default(vec, data);CPHISCHECK(err);
    break;
  case CPHIS_BACKEND_TPETRA:
    #ifdef CPHIS_HAVE_TPETRA
    err = CphisVecGetData_Tpetra(vec, data);CPHISCHECK(err);
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

CphisError CphisVecGetInternal(const CphisVec vec, void **vecInternal)
{
  *vecInternal = vec->vec;
  return CPHIS_SUCCESS;
}
