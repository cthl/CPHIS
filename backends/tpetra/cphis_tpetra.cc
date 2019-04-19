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
             int numLocalDOF,
             void *from
           )
{
  try {
    CphisError err;
    Teuchos::RCP<const TpetraMap> map;
    err = createTpetraMap(
            numElements,
            elements,
            numLocalDOF,
            map
          );CPHISCHECK(err);
    (*vec)->vec = (void*) new TpetraVec(map);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecDestroy_Tpetra(CphisVec vec)
{
  delete (TpetraVec*) vec->vec;
}

CphisError CphisVecNorm2_Tpetra(const CphisVec x, CphisReal *norm2)
{
  const TpetraVec *xVec = (TpetraVec*) x->vec;

  try {
    *norm2 = xVec->norm2();
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
  const TpetraVec *xVec = (TpetraVec*) x->vec;
  const TpetraVec *yVec = (TpetraVec*) y->vec;

  try {
    *dot = xVec->dot(*yVec);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecAXPY_Tpetra(CphisScalar a, const CphisVec x, CphisVec y)
{
  const TpetraVec *xVec = (TpetraVec*) x->vec;
  TpetraVec *yVec = (TpetraVec*) y->vec;

  try {
    yVec->update(a, *xVec, 1.0);
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
  const TpetraVec *xVec = (TpetraVec*) x->vec;
  TpetraVec *yVec = (TpetraVec*) y->vec;

  try {
    yVec->update(a, *xVec, b);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisVecScale_Tpetra(CphisVec x, CphisScalar a)
{
  TpetraVec *xVec = (TpetraVec*) x->vec;

  try{
    xVec->scale(a);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }
  
  return CPHIS_SUCCESS;
}

CphisError CphisVecSetAll_Tpetra(CphisVec vec, CphisScalar val)
{
  TpetraVec *vecVec = (TpetraVec*) vec->vec;

  try {
    vecVec->putScalar(val);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }
  
  return CPHIS_SUCCESS;
}

CphisError CphisVecAssign_Tpetra(const CphisVec x, CphisVec y)
{
  const TpetraVec *xVec = (TpetraVec*) x->vec;
  TpetraVec *yVec = (TpetraVec*) y->vec;

  try {
    *yVec = *xVec;
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  } 

  return CPHIS_SUCCESS;
}

CphisError CphisVecGetData_Tpetra(const CphisVec vec, CphisScalar **data)
{
  TpetraVec *vecVec = (TpetraVec*) vec->vec;

  try {
    *data = vecVec->getDataNonConst().get();
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }
  
  return CPHIS_SUCCESS;
}

CphisError CphisMatCreate_Tpetra(
             CphisMat *mat,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOF,
             void *from
           )
{
  try {
    CphisError err;
    Teuchos::RCP<const TpetraMap> map;
    err = createTpetraMap(
            numElements,
            elements,
            numLocalDOF,
            map
          );CPHISCHECK(err);
    (*mat)->mat = (void*) new TpetraMat(map, 0);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatDestroy_Tpetra(CphisMat mat)
{
  delete (TpetraMat*) mat->mat;
}

CphisError CphisMatVec_Tpetra(const CphisMat A, const CphisVec x, CphisVec y)
{
  const TpetraMat *AMat = (TpetraMat*) A->mat;
  const TpetraVec *xVec = (TpetraVec*) x->vec;
  TpetraVec *yVec = (TpetraVec*) y->vec;

  try {
    AMat->apply(*xVec, *yVec);
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
  const TpetraMat *matMat = (TpetraMat*) mat->mat;

  try {
    const int err = matMat->getLocalRowViewRaw(
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
  TpetraMat *matMat = (TpetraMat*) mat->mat;

  // The Tpetra matrix has no column map at this point, so we have to use
  // global indices to set the entry.
  const int localDOF = i%mat->numLocalDOF;
  const CphisIndex globalElement = mat->elements[i/mat->numLocalDOF];
  const CphisIndex iGlobal = mat->numLocalDOF*globalElement + localDOF;

  try {
    matMat->insertGlobalValues(iGlobal, 1, &aij, &j);
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatFinalize_Tpetra(CphisMat mat)
{
  TpetraMat *matMat = (TpetraMat*) mat->mat;

  try {
    matMat->fillComplete();
  }
  catch (...) {
    CPHISCHECK(CPHIS_TPETRA_ERROR);
  }

  return CPHIS_SUCCESS;
}
