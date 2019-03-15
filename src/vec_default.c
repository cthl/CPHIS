// vec_default.c - default vector implementation

#include <cphis.h>
#include <linalg.h>
#include <stdlib.h>
#include <math.h>

CphisError CphisVecCreate_default(
             CphisVec *vec,
             CphisIndex numElements,
             int numLocalDOF
           )
{
  (*vec)->vec = malloc(numElements*numLocalDOF*sizeof(CphisScalar));
  if (!(*vec)->vec) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }
  return CPHIS_SUCCESS;
}

CphisError CphisVecDestroy_default(CphisVec vec)
{
  free(vec->vec);
  return CPHIS_SUCCESS;
}

CphisError CphisVecNorm2_default(const CphisVec x, CphisReal *norm2)
{
  const CphisScalar *xData = x->vec;
  const CphisIndex numRows = x->numElements*x->numLocalDOF;
  CphisReal sum = 0.0;

  #pragma omp parallel for reduction(+:sum)
  for (CphisIndex i = 0; i < numRows; i++) {
    sum += (CphisReal) xData[i]*xData[i];
  }
  *norm2 = sqrt(sum);

  return CPHIS_SUCCESS;
}

CphisError CphisVecDot_default(
             const CphisVec x,
             const CphisVec y,
             CphisScalar *dot
           )
{
  const CphisScalar *xData = x->vec;
  const CphisScalar *yData = y->vec;
  const CphisIndex numRows = x->numElements*x->numLocalDOF;
  CphisScalar sum = 0.0;

  #pragma omp parallel for reduction(+:sum)
  for (CphisIndex i = 0; i < numRows; i++) {
    sum += xData[i]*yData[i];
  }
  *dot = sum;

  return CPHIS_SUCCESS;
}

CphisError CphisVecAXPY_default(CphisScalar a, const CphisVec x, CphisVec y)
{
  const CphisScalar *xData = x->vec;
  const CphisIndex numRows = x->numElements*x->numLocalDOF;
  CphisScalar *yData = y->vec;

  #pragma omp parallel for
  for (CphisIndex i = 0; i < numRows; i++) {
    yData[i] += a*xData[i];
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecAXPBY_default(
             CphisScalar a,
             const CphisVec x,
             CphisScalar b,
             CphisVec y
           )
{
  const CphisScalar *xData = x->vec;
  const CphisIndex numRows = x->numElements*x->numLocalDOF;
  CphisScalar *yData = y->vec;

  #pragma omp parallel for
  for (CphisIndex i = 0; i < numRows; i++) {
    yData[i] = a*xData[i] + b*yData[i];
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecScale_default(CphisVec x, CphisScalar a)
{
  CphisScalar *xData = x->vec;
  const CphisIndex numRows = x->numElements*x->numLocalDOF;

  #pragma omp parallel for
  for (CphisIndex i = 0; i < numRows; i++) {
    xData[i] *= a;
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecSetAll_default(CphisVec vec, CphisScalar val)
{
  CphisScalar *vecData = vec->vec;
  const CphisIndex numRows = vec->numElements*vec->numLocalDOF;

  #pragma omp parallel for
  for (CphisIndex i = 0; i < numRows; i++) {
    vecData[i] = val;
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecAssign_default(const CphisVec x, CphisVec y)
{
  const CphisScalar *xData = x->vec;
  CphisScalar *yData = y->vec;
  const CphisIndex numRows = x->numElements*x->numLocalDOF;

  #pragma omp parallel for
  for (CphisIndex i = 0; i < numRows; i++) {
    yData[i] = xData[i];
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecGetData_default(const CphisVec vec, CphisScalar **data)
{
  *data = vec->vec;
  return CPHIS_SUCCESS;
}
