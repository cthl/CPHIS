// scale_solver.h - type definitions for scale solvers

#ifndef __SCALE_SOLVER_H__
#define __SCALE_SOLVER_H__

#include <cphis.h>

struct _CphisScaleSolver
{
  // Solver type
  CphisScaleSolverType type;
  // The system matrix
  CphisMat A;
  // Solver tolerance
  CphisReal tol;
  // Maximum number of iterations
  int maxIter;
  // Relaxation parameter
  CphisReal omega;
  // Additional data for specific solvers
  void *data;
};

CphisError CphisScaleSolverDestroy_LU(CphisScaleSolver solver);
CphisError CphisScaleSolverSetup_LU(CphisScaleSolver solver);
CphisError CphisScaleSolverSolve_LU(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           );

struct _CphisScaleSolverData_Jacobi
{
  CphisScalar *invD;
  CphisVec r;
};

CphisError CphisScaleSolverDestroy_Jacobi(CphisScaleSolver solver);
CphisError CphisScaleSolverSetup_Jacobi(CphisScaleSolver solver);
CphisError CphisScaleSolverSolve_Jacobi(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           );

struct _CphisScaleSolverData_BlockJacobi
{
  CphisScalar *blocks;
  CphisVec r;
  CphisVec z;
};

CphisError CphisScaleSolverDestroy_BlockJacobi(CphisScaleSolver solver);
CphisError CphisScaleSolverSetup_BlockJacobi(CphisScaleSolver solver);
CphisError CphisScaleSolverSolve_BlockJacobi(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           );

struct _CphisScaleSolverData_CG
{
  // Additional vectors needed by CG.
  CphisVec r;
  CphisVec p;
  CphisVec z;
};

CphisError CphisScaleSolverDestroy_CG(CphisScaleSolver solver);
CphisError CphisScaleSolverSetup_CG(CphisScaleSolver solver);
CphisError CphisScaleSolverSolve_CG(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           );

struct _CphisScaleSolverData_BiCGStab
{
  // Additional vectors needed by CG.
  CphisVec r0;
  CphisVec r;
  CphisVec p;
  CphisVec v;
  CphisVec h;
  CphisVec s;
  CphisVec t;
};

CphisError CphisScaleSolverDestroy_BiCGStab(CphisScaleSolver solver);
CphisError CphisScaleSolverSetup_BiCGStab(CphisScaleSolver solver);
CphisError CphisScaleSolverSolve_BiCGStab(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           );

struct _CphisScaleSolverData_External
{
  // Pointer to solver setup function
  CphisError (*setupFunc)(CphisScaleSolver, const CphisMat, void*);
  // Pointer to a function that calls the external solver
  CphisError (*solveFunc)(
               CphisScaleSolver,
               const CphisVec,
               CphisVec,
               CphisConvergenceFlag*,
               CphisReal*,
               int*,
               void*
             );
  // Pointer to the solver context
  void *context;
};

CphisError CphisScaleSolverDestroy_External(CphisScaleSolver solver);
CphisError CphisScaleSolverSetup_External(CphisScaleSolver solver);
CphisError CphisScaleSolverSolve_External(
             CphisScaleSolver solver,
             const CphisVec b,
             CphisVec x,
             CphisConvergenceFlag *flag,
             CphisReal *residual,
             int *iter
           );

#endif // __SCALE_SOLVER_H__
