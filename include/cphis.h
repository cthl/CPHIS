// cphis.h - main header containing the public CPHIS API

#ifndef __CPHIS_H__
#define __CPHIS_H__

#ifdef __cplusplus
extern "C"
{
#endif

//! Basic index type
typedef int CphisIndex;
//! Basic scalar type
typedef double CphisScalar;
//! Basic type for real numbers
typedef double CphisReal;

//! @brief CPHIS error codes
//! @warning Use the `CPHISERRORCMP` macro to compare error codes.
//!          Do not use `==`!
enum
{
  // Error codes must not be negative!
  CPHIS_SUCCESS = 0,
  //! I/O error
  CPHIS_FAILED_IO,
  //! Invalid argument
  CPHIS_INVALID_ARG,
  //! Object is in invalid state.
  CPHIS_INVALID_STATE,
  //! Unkown vector, matrix, solver, or cycle type
  CPHIS_UNKNOWN_TYPE,
  //! Vectors and/or matrices are incompatible with each other or with the
  //! function called.
  CPHIS_INCOMPATIBLE,
  //! An index was out of bounds.
  CPHIS_OUT_OF_BOUNDS,
  //! Uninitialized object
  CPHIS_UNINITIALIZED,
  //! A memory allocation failed.
  CPHIS_FAILED_ALLOC,
  //! Feature has not been implemented.
  CPHIS_NOT_IMPLEMENTED,
  //! The desired backend has not been enabled.
  CPHIS_MISSING_BACKEND,
  //! An error occured inside a thread.
  //! The exact error type cannot be determined.
  CPHIS_ERROR_IN_THREAD,
  //! Test failed. Only returned by tests, not by functions.
  CPHIS_TEST_FAILED,
  //! A call to a Tpetra (Trilinos) function failed.
  CPHIS_TPETRA_ERROR,
  //! A call to MPI failed.
  CPHIS_MPI_ERROR
};
//! @brief CPHIS error type
//! @warning Use the `CPHISERRORCMP` macro to compare error codes.
typedef int CphisError;

//! Print debug message based on an error code and location in source code.
//! @note Users should not call this function directly.
CphisError CphisErrorMsg(
             CphisError err,
             const char *func,
             const char *file,
             int line
           );

//! @brief Macro for error handling
//! Negative error codes are used to indicate that an error has been forwarded
//! from a called function.
//! @note This macro can also be used to "throw" errors from a function.
//!       Using `CPHISCHECK(SOME_ERROR)` instead of `return SOME_ERROR` will
//!       provide better debugging information.
#define CPHISCHECK(ERR) \
do { \
  if ((ERR)) { \
    CphisErrorMsg((ERR), __FUNCTION__, __FILE__, __LINE__); \
    return (ERR) < 0 ? (ERR) : -(ERR); \
  } \
} while (0)

//! Like @link CPHISCHECK @endlink, but it only prints an error message and does
//! not exit the function in which the error occurred.
#define CPHISCHECKTHREAD(ERR) \
do { \
  if ((ERR)) { \
    CphisErrorMsg((ERR), __FUNCTION__, __FILE__, __LINE__); \
  } \
} while (0)

//! @brief Compare error codes.
//! Since we use negative error codes when going up the call stack, we need to
//! provide a way to safely compare error codes.
#define CPHISERRORCMP(ERR, CMP) ((ERR) == (CMP) || -(ERR) == (CMP))

//! @brief Set the function that CPHIS should use to print messages.
//! This can be used to print only on rank #0, redirect output to a file, etc.
//! By default, messages will be printed to STDOUT.
CphisError CphisSetPrintfFunc(void (*func)(const char*, ...));

//! Possible linear algebra backends for the matrix and vector classes etc.
typedef enum
{
  //! Default stand-alone backend for testing
  CPHIS_BACKEND_DEFAULT,
  //! Tpetra (Trilinos) backend
  CPHIS_BACKEND_TPETRA
} CphisBackendType;

// Forward declaration
struct _CphisVec;
//! Opaque vector handle
typedef struct _CphisVec* CphisVec;
//! @brief Create a vector.
//! @param numElements The number of elements. If multiple MPI ranks are used,
//!                    this is the number of elements on the calling rank.
//! @param elements Array of size `numElements` containing the global element
//!                 indices of the elements owned by the calling rank.
//!                 For sequential linear algebra backends, this may be set to
//!                 `NULL`.
//! @param numLocalDOF Number of degrees of freedom per element
//! @param type The type of linear algebra backend to be used for this vector
//! @param from Optional pointer to an existing vector that was created using
//!             the backend specified by `type`. If not `NULL`, this function
//!             will create a non-owning handle around the existing vector.
CphisError CphisVecCreate(
             CphisVec *vec,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOF,
             CphisBackendType type,
             void *from
           );
//! Destroy a vector.
CphisError CphisVecDestroy(CphisVec vec);
//! Get the vector's number of elements.
CphisError CphisVecGetNumElements(const CphisVec vec, CphisIndex *numElements);
//! Get the vector's number of local degrees of freedom per element.
CphisError CphisVecGetNumLocalDOF(const CphisVec vec, int *numLocalDOF);
//! Compute the Euclidean norm of a vector.
CphisError CphisVecNorm2(const CphisVec x, CphisReal *norm2);
//! Compute a dot product.
CphisError CphisVecDot(const CphisVec x, const CphisVec y, CphisScalar *dot);
//! Update a vector, i.e., let \f$y \leftarrow ax + y\f$.
CphisError CphisVecAXPY(CphisScalar a, const CphisVec x, CphisVec y);
//! Update a vector, i.e., let \f$y \leftarrow ax + by\f$.
CphisError CphisVecAXPBY(
             CphisScalar a,
             const CphisVec x,
             CphisScalar b,
             CphisVec y
           );
//! Multiply a vector with a scalar.
CphisError CphisVecScale(CphisVec x, CphisScalar a);
//! Set all vector entries to the given value.
CphisError CphisVecSetAll(CphisVec vec, CphisScalar val);
//! Copy the entries of a vector, i.e., let \f$y=x\f$.
CphisError CphisVecAssign(const CphisVec x, CphisVec y);
//! Get a pointer to the array of vector entries.
CphisError CphisVecGetData(const CphisVec vec, CphisScalar **data);
//! @brief Load a vector from a matrix market file.
//! The vector will use the default linear algebra backend.
CphisError CphisVecFromMatrixMarket(
             CphisVec *vec,
             int numLocalDOF,
             const char *file
           );

// Forward declaration
struct _CphisMat;
//! Opaque matrix handle
typedef struct _CphisMat* CphisMat;
//! @brief Create a matrix.
//! See @link CphisVecCreate @endlink for details.
CphisError CphisMatCreate(
             CphisMat *mat,
             CphisIndex numElements,
             const CphisIndex *elements,
             int numLocalDOF,
             CphisBackendType type,
             void *from
           );
//! Destroy a matrix.
CphisError CphisMatDestroy(CphisMat mat);
//! Get the matrix' number of elements.
CphisError CphisMatGetNumElements(const CphisMat mat, CphisIndex *numElements);
//! Get the matrix' number of local degrees of freedom per element.
CphisError CphisMatGetNumLocalDOF(const CphisMat mat, int *numLocalDOF);
//! Compute sparse matrix-vector product, i.e., \f$y=Ax\f$.
CphisError CphisMatVec(const CphisMat A, const CphisVec x, CphisVec y);
//! Get a pointer to the entries in the given (local) matrix row.
CphisError CphisMatGetData(
             const CphisMat mat,
             CphisIndex row,
             const CphisIndex **cols,
             const CphisScalar **vals,
             CphisIndex *numEntries
           );
//! @brief Set (or add) a matrix entry.
//! @param i Row index. If the matrix is distributed,
//!          this is the local row index.
//! @param j Column index
//! @param aij Value
//! @note It must be safe for different threads to set matrix entries
//!       concurrently, so long as the entries are in different rows.
//! @warning Calling this method twice for the same entry is undefined behavior!
CphisError CphisMatSet(
             CphisMat mat,
             CphisIndex i,
             CphisIndex j,
             CphisScalar aij
           );
//! @brief Finalize matrix assembly.
//! This function must be called before the matrix can be used in computations.
CphisError CphisMatFinalize(CphisMat mat);
//! @brief Load a matrix from a matrix market file.
//! The matrix will use the default linear algebra backend.
CphisError CphisMatFromMatrixMarket(
             CphisMat *mat,
             int numLocalDOF,
             const char *file
           );

//! Solver status flags
typedef enum
{
  //! Solver converged to the desired tolerance.
  CPHIS_CONVERGED = 0,
  //! Solver stopped after reaching the maximum number of iterations.
  CPHIS_MAX_ITER
} CphisConvergenceFlag;

//! Scale solver types
typedef enum
{
  //! Dense LU solver
  CPHIS_SCALE_SOLVER_LU,
  //! Jacobi smoother
  CPHIS_SCALE_SOLVER_JACOBI,
  //! Block Jacobi smoother
  CPHIS_SCALE_SOLVER_BLOCK_JACOBI,
  //! Conjugate gradient solver (unpreconditioned)
  CPHIS_SCALE_SOLVER_CG,
  //! BiCGStab solver (unpreconditioned).
  //! When using multiple threads, the convergence behavior can be *highly*
  //! non-deterministic unless deterministic OpenMP reductions are enforced!
  CPHIS_SCALE_SOLVER_BICGSTAB,
  //! This solver type provides an interface to external solvers from other
  //! packages.
  //! See @link CphisScaleSolverSetExternal @endlink for details.
  CPHIS_SCALE_SOLVER_EXTERNAL
} CphisScaleSolverType;

// Forward declaration
struct _CphisScaleSolver;
//! Opaque scale solver handle
typedef struct _CphisScaleSolver* CphisScaleSolver;

//! Create a scale solver of the given type.
CphisError CphisScaleSolverCreate(
             CphisScaleSolver *solver,
             CphisScaleSolverType type
           );
//! Destroy a scale solver.
CphisError CphisScaleSolverDestroy(CphisScaleSolver solver);
//! @brief Set the maximum number of solver iterations.
//! This has no effect for direct solvers.
CphisError CphisScaleSolverSetMaxIter(CphisScaleSolver solver, int maxIter);
//! @brief Set the relative solver tolerance.
//! See @link CphisConfSetTol @endlink for the definition of the tolerances.
//! @note This may be set to `0.0` if no convergence test is desired, e.g.,
//!       for smoothers.
CphisError CphisScaleSolverSetTol(CphisScaleSolver solver, CphisReal tol);
//! @brief Set the relaxation parameter \f$\omega\f$.
//! This can be used for over-relaxation and under-relaxation.
//! The parameter will be ignored by solvers that cannot use it.
CphisError CphisScaleSolverSetOmega(CphisScaleSolver solver, CphisReal omega);
//! @brief Set up an external solver.
//! This function is used to set up a CPHIS interface to an external solver.
//! The user provides the functions `setupFunc` and `solveFunc`, which are
//! basically the implementation of @link CphisScaleSolverSetup @endlink
//! and @link CphisScaleSolverSolve @endlink for the external solver.
//! When these two CPHIS functions are called, they simply call `setupFunc` and
//! `solveFunc`, forwarding all arguments and adding the `context` pointer.
//! The `context` is a user-provided data structure that most likely contains
//! pointers to the external solver, preconditioner, etc.
CphisError CphisScaleSolverSetExternal(
             CphisScaleSolver solver,
             CphisError (*setupFunc)(CphisScaleSolver, const CphisMat, void*),
             CphisError (*solveFunc)(
                          CphisScaleSolver,
                          const CphisVec,
                          CphisVec,
                          CphisConvergenceFlag*,
                          CphisReal*,
                          int*,
                          void*
                        ),
             void *context
           );
//! @brief Set the matrix and prepare the solver, preconditioner, etc.
//! @warning This function must not be called twice for the same solver!
//!          If a new matrix is to be used, please create a new solver.
CphisError CphisScaleSolverSetup(CphisScaleSolver solver, const CphisMat A);
//! @brief Solve a linear system.
//! @param b Right-hand side of the system
//! @param x Solution vector. It will also be used as an initial guess for
//!          iterative solvers.
//! @param flag Convergence flag (converged, ran out of iterations, etc.).
//!             `NULL` can be passed if the convergence flag is to be ignored.
//! @param residual Achieved relative residual. Meaningless for direct solvers.
//!                 `NULL` can be passed if the residual is to be ignored.
//! @param iter Number of iterations. Meaningless for direct solvers.
//!             `NULL` can be passed if the number of iterations is to be
//!             ignored.
CphisError CphisScaleSolverSolve(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           );

//! Cycles, i.e., the order in which scales are visited
typedef enum
{
  //! V-cycle
  CPHIS_CYCLE_V,
  //! W-cycle
  CPHIS_CYCLE_W,
  //! F-cycle
  CPHIS_CYCLE_F
} CphisCycleType;

//! CPHIS verbosity levels
typedef enum
{
  //! Only report errors.
  CPHIS_VERBOSITY_ERRORS = 0,
  //! Report number of cycles, achieved tolerance, etc.
  CPHIS_VERBOSITY_SUMMARY,
  //! Detailed output in each iteration
  CPHIS_VERBOSITY_DETAILED,
  // Make sure that this is always the last entry!
  //! Maximum verbosity
  CPHIS_VERBOSITY_ALL
} CphisVerbosityLevel;

// Forward declaration
struct _CphisConf;
//! Opaque CPHIS configuration handle
typedef struct _CphisConf* CphisConf;

//! @brief Create a CPHIS configuration.
//! @param numLocalDOF Dimension of local approximation space
//! @param numScales Number of scales into which the local approximation space
//!                  is to be split
//! @param scales Array of size `2*numScales` whose entries `2s`th and
//!               `(2s+1)`th entries define the first and last local DOF in the
//!               `s`th scale
CphisError CphisConfCreate(
             CphisConf *conf,
             int numLocalDOF,
             int numScales,
             const int *scales
           );
//! Destroy a configuration.
CphisError CphisConfDestroy(CphisConf conf);
//! @brief Set relative solver tolerance.
//! CPHIS will stop once \f$\|b-Ax_k\|_2/\|b-Ax_0\|_2<\text{tol}\f$.
CphisError CphisConfSetTol(CphisConf conf, CphisReal tol);
//! @brief Set the maximum number of iterations (cycles).
//! This can be used to stop early in case of slow convergence, or to limit the
//! number of cycles when CPHIS is used as a preconditioner.
CphisError CphisConfSetMaxIter(CphisConf conf, int maxIter);
//! Select the cycling strategy.
CphisError CphisConfSetCycleType(CphisConf conf, CphisCycleType cycle);
//! @brief Select if FMG-type cycles should be used.
//! FMG-type cycles start at the lowest scale and work their way up.
CphisError CphisConfSetFMGCycle(CphisConf conf, int fmgCycle);
//! Set the verbosity level.
CphisError CphisConfSetVerbosity(CphisConf conf, CphisVerbosityLevel verbosity);
//! @brief Set the number of smoothing iterations.
//! @param nu1 Number of smoother iterations before visiting lower scale
//! @param nu2 Number of smoother iterations after visiting lower scale
//! @note Will be ignored on scales where direct solvers are used.
CphisError CphisConfSetNu(CphisConf conf, int nu1, int nu2);
//! @brief Set the solver for the given scale.
//! The solver must have been created using @link CphisScaleSolverCreate
//! @endlink, and all desired parameters (tolerance, number of iterations, etc.)
//! must have been set. Some parameters might be overwritten later, e.g., to
//! ensure the correct number of smoothing iterations (see @link CphisConfSetNu
//! @endlink).
//! @note This function should only be used to set the coarse scale solver
//!       (`scale=0`) or if custom smoothers are used. In order to use one of
//!       the standard CPHIS smoothers on all scales, please use
//!       @link CphisConfSetSmoothers @endlink instead of creating and setting
//!       the smoothers manually for each scale.
CphisError CphisConfSetScaleSolver(
             CphisConf conf,
             int scale,
             CphisScaleSolver solver
           );
//! @brief Set and create the smoothers for all scales except the coarsest one.
//! This function creates a smoother of the desired type for each scale and
//! then sets it. This is more convenient than creating and setting smoothers
//! manually for each scale.
CphisError CphisConfSetSmoothers(CphisConf conf, CphisScaleSolverType type);

// Forward declaration
struct _CphisSolver;
//! @brief Opaque CPHIS solver handle
//! Not to be confused with @link CphisScaleSolver @endlink.
typedef struct _CphisSolver* CphisSolver;

//! @brief Create a CPHIS solver.
//! @param A System matrix
//! @param conf CPHIS configuration
//! @note Neither the system matrix nor the configuration can change after
//!       the solver has been created.
CphisError CphisSolverCreate(
             CphisSolver *solver,
             const CphisConf conf,
             const CphisMat A
           );
//! Destroy a CPHIS solver.
CphisError CphisSolverDestroy(CphisSolver solver);
//! Solve a linear system.
//! @param b Right-hand side of the system
//! @param x Solution vector. It will also be used as the initial guess.
//! @param flag Flag indicating whether the solver converged,
//!             reached the maximum number of iterations, etc.
//! @param residual Achieved relative residual.
//! @param iter Number of iterations (cycles).
CphisError CphisSolverSolve(
             const CphisSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           );
#ifdef __cplusplus
} // extern "C"
#endif

#endif // __CPHIS_H__
