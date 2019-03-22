// cphis_tpetra.cc - implementation of the Tpetra (Trilinos) backend

#include <cphis.h>
#include <cphis_tpetra.h>
#include <linalg.h>
#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <mpi.h>

// Define Tpetra matrix and vector types.
typedef Kokkos::Compat::KokkosOpenMPWrapperNode TpetraNode;
typedef Tpetra::Map<CphisIndex, CphisIndex, TpetraNode> TpetraMap;
typedef Tpetra::Vector<
          CphisScalar,
          CphisIndex,
          CphisIndex,
          TpetraNode
        > TpetraVec;
typedef Tpetra::CrsMatrix<
          CphisScalar,
          CphisIndex,
          CphisIndex,
          TpetraNode
        > TpetraMat;

struct _CphisVec_Tpetra
{
  // Pointer to the actual Tpetra vector
  TpetraVec *vec;
  // Was this vector created by CPHIS or passed by the user?
  bool owned;
};

static CphisError createTpetraMap(
                    CphisIndex numElements,
                    const CphisIndex *elements,
                    int numLocalDOF,
                    Teuchos::RCP<const TpetraMap> &map
                  )
{
  int err;
  long numGlobalElements, numElementsLong;

  numElementsLong = (long) numElements;
  err = MPI_Allreduce(
          &numElementsLong,
          &numGlobalElements,
          1,
          MPI_LONG,
          MPI_SUM,
          MPI_COMM_WORLD
        );
  if (err) {
    CPHISCHECK(CPHIS_MPI_ERROR);
  }

  // Prepare list of elements.
  Teuchos::ArrayRCP<CphisIndex> indices(numElements*numLocalDOF);
  for (CphisIndex e = 0; e < numElements; e++) {
    for (int l = 0; l < numLocalDOF; l++) {
      indices[numLocalDOF*e + l] = numLocalDOF*elements[e] + l;
    }
  }

  Teuchos::RCP<Teuchos::MpiComm<int> > comm
    = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
  map = Teuchos::rcp(
          new TpetraMap(
                numGlobalElements,
                indices.view(0, numElements*numLocalDOF),
                0,
                comm
              )
        );

  return CPHIS_SUCCESS;
}

CphisError CphisVecCreate_Tpetra(
             CphisVec *vec,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOF
           )
{
  CphisError err;

  try {
    (*vec)->vec = new struct _CphisVec_Tpetra;
  }
  catch (...) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  struct _CphisVec_Tpetra *vecVec = (struct _CphisVec_Tpetra*) (*vec)->vec;

  try {
    Teuchos::RCP<const TpetraMap> map;
    err = createTpetraMap(
            numElements,
            elements,
            numLocalDOF,
            map
          );CPHISCHECK(err);
    vecVec->vec = new TpetraVec(map);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }
  vecVec->owned = true;

  return CPHIS_SUCCESS;
}

CphisError CphisVecDestroy_Tpetra(CphisVec vec)
{
  struct _CphisVec_Tpetra *vecVec = (struct _CphisVec_Tpetra*) vec->vec;

  if (vecVec->owned) {
    delete vecVec->vec;
  }
  delete vec->vec;
}

CphisError CphisVecNorm2_Tpetra(const CphisVec x, CphisReal *norm2)
{
  const struct _CphisVec_Tpetra *xVec = (const struct _CphisVec_Tpetra*) x->vec;

  try {
    *norm2 = xVec->vec->norm2();
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecDot_Tpetra(
             const CphisVec x,
             const CphisVec y,
             CphisScalar *dot
           )
{
  const struct _CphisVec_Tpetra *xVec = (const struct _CphisVec_Tpetra*) x->vec;
  const struct _CphisVec_Tpetra *yVec = (const struct _CphisVec_Tpetra*) y->vec;

  try {
    *dot = xVec->vec->dot(*yVec->vec);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecAXPY_Tpetra(CphisScalar a, const CphisVec x, CphisVec y)
{
  const struct _CphisVec_Tpetra *xVec = (const struct _CphisVec_Tpetra*) x->vec;
  struct _CphisVec_Tpetra *yVec = (struct _CphisVec_Tpetra*) y->vec;

  try {
    yVec->vec->update(a, *xVec->vec, 1.0);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecAXPBY_Tpetra(
             CphisScalar a,
             const CphisVec x,
             CphisScalar b,
             CphisVec y
           )
{
  const struct _CphisVec_Tpetra *xVec = (const struct _CphisVec_Tpetra*) x->vec;
  struct _CphisVec_Tpetra *yVec = (struct _CphisVec_Tpetra*) y->vec;

  try {
    yVec->vec->update(a, *xVec->vec, b);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecScale_Tpetra(CphisVec x, CphisScalar a)
{
  struct _CphisVec_Tpetra *xVec = (struct _CphisVec_Tpetra*) x->vec;

  try{
    xVec->vec->scale(a);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }
  
  return CPHIS_SUCCESS;
}

CphisError CphisVecSetAll_Tpetra(CphisVec vec, CphisScalar val)
{
  struct _CphisVec_Tpetra *vecVec = (struct _CphisVec_Tpetra*) vec->vec;

  try {
    vecVec->vec->putScalar(val);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }
  
  return CPHIS_SUCCESS;
}

CphisError CphisVecAssign_Tpetra(const CphisVec x, CphisVec y)
{
  const struct _CphisVec_Tpetra *xVec = (const struct _CphisVec_Tpetra*) x->vec;
  struct _CphisVec_Tpetra *yVec = (struct _CphisVec_Tpetra*) y->vec;

  try {
    *yVec->vec = *xVec->vec;
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  } 

  return CPHIS_SUCCESS;
}

CphisError CphisVecGetData_Tpetra(const CphisVec vec, CphisScalar **data)
{
  struct _CphisVec_Tpetra *vecVec = (struct _CphisVec_Tpetra*) vec->vec;

  try {
    *data = vecVec->vec->getDataNonConst().get();
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }
  
  return CPHIS_SUCCESS;
}

struct _CphisMat_Tpetra
{
  // Pointer to the actual Tpetra matrix
  TpetraMat *mat;
  // Was this matrix created by CPHIS or passed by the user?
  bool owned;
};

CphisError CphisMatCreate_Tpetra(
             CphisMat *mat,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOF
           )
{
  CphisError err;

  try {
    (*mat)->mat = new struct _CphisMat_Tpetra;
  }
  catch (...) {
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  struct _CphisMat_Tpetra *matMat = (struct _CphisMat_Tpetra*) (*mat)->mat;

  try {
    Teuchos::RCP<const TpetraMap> map;
    err = createTpetraMap(
            numElements,
            elements,
            numLocalDOF,
            map
          );CPHISCHECK(err);
    matMat->mat = new TpetraMat(map, 0);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }
  matMat->owned = true;

  return CPHIS_SUCCESS;
}

CphisError CphisMatDestroy_Tpetra(CphisMat mat)
{
  struct _CphisMat_Tpetra *matMat = (struct _CphisMat_Tpetra*) mat->mat;

  if (matMat->owned) {
    delete matMat->mat;
  }
  delete mat->mat;
}

CphisError CphisMatVec_Tpetra(const CphisMat A, const CphisVec x, CphisVec y)
{
  const struct _CphisMat_Tpetra *AMat = (const struct _CphisMat_Tpetra*) A->mat;
  const struct _CphisVec_Tpetra *xVec = (const struct _CphisVec_Tpetra*) x->vec;
  struct _CphisVec_Tpetra *yVec = (struct _CphisVec_Tpetra*) y->vec;

  try {
    AMat->mat->apply(*xVec->vec, *yVec->vec);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatGetData_Tpetra(
             const CphisMat mat,
             CphisIndex row,
             const CphisIndex **cols,
             const CphisScalar **vals,
             CphisIndex *numEntries
           )
{
  const struct _CphisMat_Tpetra *matMat;
  matMat = (const struct _CphisMat_Tpetra*) mat->mat;

  try {
    const int err = matMat->mat->getLocalRowViewRaw(
                                   row,
                                   *numEntries,
                                   *cols,
                                   *vals
                                 );
    if (err) {
      CPHISCHECK(CPHIS_TPETRA_ERROR);
    }
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatSet_Tpetra(
             CphisMat mat,
             CphisIndex i,
             CphisIndex j,
             CphisScalar aij
           )
{
  return CPHIS_SUCCESS;
}

CphisError CphisMatFinalize_Tpetra(CphisMat mat)
{
  struct _CphisMat_Tpetra *matMat = (struct _CphisMat_Tpetra*) mat->mat;

  try {
    matMat->mat->fillComplete();
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}
