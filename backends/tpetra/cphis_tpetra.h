// cphis_tpetra.h - declaration of the Tpetra (Trilinos) backend

#ifndef __CPHIS_TPETRA_H__
#define __CPHIS_TPETRA_H__

#include <cphis.h>

// While we implement a C interface, the implementation itself uses Trilinos
// and therefore C++.
#ifdef __cplusplus
extern "C"
{
#endif
CphisError CphisVecCreate_Tpetra(
             CphisVec *vec,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOF
           );
CphisError CphisVecDestroy_Tpetra(CphisVec vec);
CphisError CphisVecNorm2_Tpetra(const CphisVec x, CphisReal *norm2);
CphisError CphisVecDot_Tpetra(
             const CphisVec x,
             const CphisVec y,
             CphisScalar *dot
           );
CphisError CphisVecAXPY_Tpetra(CphisScalar a, const CphisVec x, CphisVec y);
CphisError CphisVecAXPBY_Tpetra(
             CphisScalar a,
             const CphisVec x,
             CphisScalar b,
             CphisVec y
           );
CphisError CphisVecScale_Tpetra(CphisVec x, CphisScalar a);
CphisError CphisVecSetAll_Tpetra(CphisVec vec, CphisScalar val);
CphisError CphisVecAssign_Tpetra(const CphisVec x, CphisVec y);
CphisError CphisVecGetData_Tpetra(const CphisVec vec, CphisScalar **data);

CphisError CphisMatCreate_Tpetra(
             CphisMat *mat,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOF
           );
CphisError CphisMatDestroy_Tpetra(CphisMat mat);
CphisError CphisMatVec_Tpetra(const CphisMat A, const CphisVec x, CphisVec y);
CphisError CphisMatGetData_Tpetra(
             const CphisMat mat,
             CphisIndex row,
             const CphisIndex **cols,
             const CphisScalar **vals,
             CphisIndex *numEntries
           );
CphisError CphisMatSet_Tpetra(
             CphisMat mat,
             CphisIndex i,
             CphisIndex j,
             CphisScalar aij
           );
CphisError CphisMatFinalize_Tpetra(CphisMat mat);
#ifdef __cplusplus
} // extern "C"
#endif

#endif // __CPHIS_TPETRA_H__
