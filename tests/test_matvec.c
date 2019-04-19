#include <cphis.h>
#include <linalg.h>
#include "test_poisson2d.h"
#include <math.h>
#include <stdlib.h>

int main()
{
  CphisError err;

  // Problem size, number of matrix-vector products, and reference result.
  const CphisIndex n = 100;
  const CphisIndex n2 = n*n;
  const int m = 3;
  const CphisReal reference = 1.356171080653171e2;

  // Create matrix and vectors.
  CphisMat A;
  CphisVec x, y;
  err = CphisMatCreate(
          &A,
          n2,
          NULL,
          1,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);
  err = CphisVecCreate(
          &x,
          n2,
          NULL,
          1,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);
  err = CphisVecCreate(
          &y,
          n2,
          NULL,
          1,
          CPHIS_BACKEND_DEFAULT,
          NULL
        );CPHISCHECK(err);

  err = getPoisson2dSystem(n, A, NULL, x, NULL);

  // Perform matrix-vector products and check the results.
  for (int i = 0; i < m; i++) {
    err = CphisMatVec(A, x, y);CPHISCHECK(err);
    err = CphisVecAssign(y, x);CPHISCHECK(err);
  }
  CphisReal norm2;
  err = CphisVecNorm2(x, &norm2);CPHISCHECK(err);
  if (fabs(norm2 - reference)/reference > 5e-15) {
    return CPHIS_TEST_FAILED;
  }

  // Clean up.
  err = CphisVecDestroy(x);CPHISCHECK(err);
  err = CphisVecDestroy(y);CPHISCHECK(err);
  err = CphisMatDestroy(A);CPHISCHECK(err);

  return CPHIS_SUCCESS;
}
