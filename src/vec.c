// vec.c - generic vector interface implementation

#include <cphis.h>
#include <linalg.h>
#include <stdlib.h>

CphisError CphisVecCreate(
             CphisVec *vec,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOF,
             CphisBackendType type
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

  CphisError err;
  switch (type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisVecCreate_default(vec, numElements, numLocalDOF);
    break;
  default:
    err = CPHIS_UNKNOWN_TYPE;
    break;
  }

  if (err) {
    free(*vec);
  }

  CPHISCHECK(err);
  return CPHIS_SUCCESS;
}

CphisError CphisVecDestroy(CphisVec vec)
{
  CphisError err;
  switch (vec->type) {
  case CPHIS_BACKEND_DEFAULT:
    err = CphisVecDestroy_default(vec);
    break;
  default:
    err = CPHIS_UNKNOWN_TYPE;
    break;
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
  default:
    CPHISCHECK(CPHIS_UNKNOWN_TYPE);
    break;
  }

  return CPHIS_SUCCESS;
}
