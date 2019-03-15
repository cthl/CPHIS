// linalg.h - declaration of linear algebra functions and types

#ifndef __LINALG_H__
#define __LINALG_H__

#include <cphis.h>
#include <stdlib.h>

// Check compatibility of two vectors.
#define CPHISCHECKVECCOMPAT(X, Y) \
do { \
  if ( \
    (X)->type != (Y)->type || \
    (X)->numElements != (Y)->numElements || \
    (X)->numLocalDOF != (Y)->numLocalDOF \
  ) { \
    CPHISCHECK(CPHIS_INCOMPATIBLE); \
  } \
} while (0)

// Check compatibility for matrix-vector product.
#define CPHISCHECKMATVECCOMPAT(A, X, Y) \
do { \
  if ( \
    !(A)->finalized || \
    (A)->type != (X)->type || \
    (A)->type != (Y)->type || \
    (A)->numElements != (Y)->numElements \
  ) { \
    CPHISCHECK(CPHIS_INCOMPATIBLE); \
  } \
} while (0)

struct _CphisVec
{
  // Linear algebra backend for this vector handle.
  CphisBackendType type;
  // Number of elements (on this rank)
  CphisIndex numElements;
  // Array of global element indices owned by this process
  CphisIndex *elements;
  // Number of local degrees of freedom per element
  int numLocalDOF;
  // Pointer to the actual vector.
  // This could be as simple as an array of scalars or as complex as a
  // distributed vector from a package like Trilinos.
  void *vec;
};

// Default vector implementation
CphisError CphisVecCreate_default(
             CphisVec *vec,
             CphisIndex numElements,
             int numLocalDOF
           );
CphisError CphisVecDestroy_default(CphisVec vec);
CphisError CphisVecNorm2_default(const CphisVec x, CphisReal *norm2);
CphisError CphisVecDot_default(
             const CphisVec x,
             const CphisVec y,
             CphisScalar *dot
           );
CphisError CphisVecAXPY_default(CphisScalar a, const CphisVec x, CphisVec y);
CphisError CphisVecAXPBY_default(
             CphisScalar a,
             const CphisVec x,
             CphisScalar b,
             CphisVec y
           );
CphisError CphisVecScale_default(CphisVec x, CphisScalar a);
CphisError CphisVecSetAll_default(CphisVec vec, CphisScalar val);
CphisError CphisVecAssign_default(const CphisVec x, CphisVec y);
CphisError CphisVecGetData_default(const CphisVec vec, CphisScalar **data);

struct _CphisMat
{
  // Linear algebra backend for this matrix handle.
  CphisBackendType type;
  // Matrix dimensions.
  CphisIndex numElements;
  // Array of global element indices owned by this process
  CphisIndex *elements;
  // Number of local degrees of freedom per element
  int numLocalDOF;
  // Pointer to the actual matrix (see vector class).
  void *mat;
  // A flag to determine whether or not the matrix has been finalized.
  int finalized;
};

// Internal matrix structure for the default implementation
struct _CphisMat_default
{
  // Buffers used during matrix assembly
  CphisIndex *bufferCapacities;
  CphisIndex *bufferSizes;
  CphisIndex **colBuffers;
  CphisScalar **valBuffers;
  // Final matrix data in CRS format
  size_t *rows;
  CphisIndex *cols;
  CphisScalar *vals;
};

// Default matrix implementation
CphisError CphisMatCreate_default(
             CphisMat *mat,
             CphisIndex numElements,
             int numLocalDOF
           );
CphisError CphisMatDestroy_default(CphisMat mat);
CphisError CphisMatVec_default(const CphisMat A, const CphisVec x, CphisVec y);
CphisError CphisMatGetData_default(
             const CphisMat mat,
             CphisIndex row,
             CphisIndex **cols,
             CphisScalar **vals,
             CphisIndex *numEntries
           );
CphisError CphisMatSet_default(
             CphisMat mat,
             CphisIndex i,
             CphisIndex j,
             CphisScalar aij
           );
CphisError CphisMatFinalize_default(CphisMat mat);

// Factorize an nxn matrix. Matrices are given as dense, linear arrays with
// row-first ordering.
void CphisLUFactorize(CphisIndex n, CphisScalar *A);

// Solve an nxn system using an existing LU factorization.
void CphisLUSolve(
       CphisIndex n,
       const CphisScalar *LU,
       const CphisScalar *rhs,
       CphisScalar *sol
     );

#endif // __LINALG__H_
