#include <cphis.h>
#include <linalg.h>
#include <math.h>
#include <string.h>

int main()
{
  // Define a 4x4 system.
  const CphisScalar A[16] = { 174,   80,  166,   92,
                               80, 1050,  464,  262,
                              166,  464, 3630,  444,
                               92,  262,  444,  518};
  const CphisScalar b[4] = {1, 2, 3, 4};
  CphisScalar LU[16];
  CphisScalar x[4];

  // Factorize matrix and solve linear system.
  memcpy(LU, A, 16*sizeof(CphisScalar));
  CphisLUFactorize(4, LU);
  CphisLUSolve(4, LU, b, x);

  // Compute relative residual norm.
  CphisScalar rnorm = 0.0, bnorm = 0.0;
  for (int i = 0; i < 4; i++) {
    CphisScalar r = b[i];
    for (int j = 0; j < 4; j++) {
      r -= A[4*i + j]*x[j];
    }
    rnorm += r*r;
    bnorm += b[i]*b[i];
  }
  rnorm = sqrt(rnorm);
  bnorm = sqrt(bnorm);

  // Check tolerance.
  if (rnorm/bnorm > 1.0e-15) {
    CPHISCHECK(CPHIS_TEST_FAILED);
  }
  return CPHIS_SUCCESS;
}
