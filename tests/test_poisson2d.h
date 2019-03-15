#ifndef __TEST_POISSON2D_H__
#define __TEST_POISSON2D_H__

#include <cphis.h>
#include <math.h>

#define PI 3.14159265358979323846264338327950288

// Create a linear system for testing purposes.
// NULL can be passed for any of the vectors if they are not desired.
static CphisError getPoisson2dSystem(
                    CphisIndex n,
                    CphisMat A,
                    CphisVec f,
                    CphisVec u0,
                    CphisVec u
                  )
{
  CphisError err;
  const CphisScalar h = 1.0/(n + 1);
  const CphisScalar h2 = h*h;
  CphisScalar *uData, *u0Data, *fData;
  if (f) {
    err = CphisVecGetData(u, &uData);CPHISCHECK(err);
  }
  if (u0) {
    err = CphisVecGetData(u0, &u0Data);CPHISCHECK(err);
  }
  if (u) {
    err = CphisVecGetData(f, &fData);CPHISCHECK(err);
  }

  // Populate the system.
  for (CphisIndex i = 0; i < n; i++) {
    for (CphisIndex j = 0; j < n; j++) {
      const CphisIndex row = n*i + j;
      CphisIndex col;

      // Matrix entries
      col = row;
      err = CphisMatSet(A, row, col, 4.0);CPHISCHECK(err);

      col = row - 1;
      if (j > 0) {
        err = CphisMatSet(A, row, col, -1.0);CPHISCHECK(err);
      }

      col = row + 1;
      if (j < n - 1) {
        err = CphisMatSet(A, row, col, -1.0);CPHISCHECK(err);
      }

      col = row - n;
      if (i > 0) {
        err = CphisMatSet(A, row, col, -1.0);CPHISCHECK(err);
      }

      col = row + n;
      if (i < n - 1) {
        err = CphisMatSet(A, row, col, -1.0);CPHISCHECK(err);
      }

      // Populate vectors.
      const CphisScalar gridX = (j + 1)*h;
      const CphisScalar gridY = (i + 1)*h;
      // Right-hand side
      if (f) fData[row] = h2*2.0*PI*PI*sin(PI*gridX)*sin(PI*gridY);
      // Exact solution
      if (u) uData[row] = sin(PI*gridX)*sin(PI*gridY);
      // Initial guess
      if (u0) u0Data[row] = 1.0;
    }
  }
  err = CphisMatFinalize(A);CPHISCHECK(err);

  return CPHIS_SUCCESS;
}

#endif // __TEST_POISSON2D_H__
