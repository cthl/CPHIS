// solver.h - declaration of solver functions and structures

#ifndef __SOLVER_H__
#define __SOLVER_H__

#include <cphis.h>
#include <linalg.h>
#include <conf.h>

enum CphisTimer
{
  CPHIS_TIMER_SOLVER = 0, // Must be the first in this enumeration.
  CPHIS_TIMER_PRESMOOTH,
  CPHIS_TIMER_POSTSMOOTH,
  CPHIS_TIMER_COARSE_SOLVE,
  CPHIS_TIMER_RESIDUAL,
  CPHIS_TIMER_PROLONGATION,
  CPHIS_TIMER_SETUP,
  CPHIS_TIMER_SYSTEM_RESIDUAL,
  CPHIS_NUM_TIMERS // Must be the last in this enumeration.
};

struct _CphisSolver
{
  // Solver configuration
  CphisConf conf;
  // System matrix
  CphisMat A;
  // Full residual of the original system
  CphisVec rfull;
  // Arrays for the extracted matrix blocks.
  // The matrix Ascale[s] is the one used by the smoother on scale s.
  // Aresidual[s] is the matrix used to compute the *restricted* residual for
  // scales lower than s.
  // Arhs[s] is used to construct the right-hand side for the smoother.
  // It is only needed on scales where not all d.o.f. are smoothed, i.e., on
  // HSS-type scales.
  //
  // If the sth scale is an HSS-type scale that does *not* overlap with lower
  // scales, the blocks have the following structure:
  //
  // --------------------------------
  // | **************************** | < d.o.f. not part of sth scale
  // | * ---------------------------|
  // | * |           |              | \
  // | * | Ascale[s] | Arhs[s]      | | d.o.f. smoothed
  // | * | (square)  | (wide)       | | on sth scale
  // | * |           |              | /
  // | * |-----------|--------------|
  // | * |           | xxxxxxxxxxxx | \
  // | * | Ares.[s]  | xxxxxxxxxxxx | |
  // | * | (tall)    | xxxxxxxxxxxx | | d.o.f. *not* smoothed
  // | * |           | xxxxxxxxxxxx | | on sth scale
  // | * |           | xxxxxxxxxxxx | |
  // | * |           | xxxxxxxxxxxx | /
  // --------------------------------
  //
  // Notice that Ascale[s] is small, since only some d.o.f. are smoothed on this
  // scale. If the sth scale was of standard p-multigrid type, i.e., if all
  // d.o.f. in the scale were smoothed, the matrix Ascale[s] would extend all
  // the way to the bottom-right corner. In that case the matrix Arhs[s] would
  // vanish, and the matrix Aresidual[s] would extend all the way to the right.
  // Please see below for explanations.
  //
  // The matrix Aresidual[s] is used to compute the residual on the sth scale
  // and restrict it to lower scales in a single step. In general, it contains
  // entries from the original system matrix A that satisfies the following
  // condition:
  //   (1) The row corresponds to a d.o.f. that is smoothed in *any* scale lower
  //       than the current scale.
  // If the sth scale performs an error correction, i.e., it starts with a zero
  // initial guess, then only the d.o.f. that are smoothed on this scale can
  // be nonzero and therefore affect the residual. Hence, Aresidual[s] only
  // needs to contain the entries of A that satisfy (1) and the following
  // condition:
  //   (2) The column corresponds to a d.o.f. that is smoothed on the current
  //       scale.
  // As a result, the matrix Aresidual[s] will be skinny, as depicted above.
  //
  // The matrix Arhs[s] is used to construct the right-hand side of the smoother
  // on HSS-type scales. If only a subset of the d.o.f. is smoothed, the
  // remaining d.o.f. do not change. They can therefore be moved to the
  // right-hand side of the smoothing step.
  CphisMat *Ascale;
  CphisMat *Aresidual;
  CphisMat *Arhs;
  // Vectors needed for the standard p-multigrid method
  CphisVec *r; // Residuals
  CphisVec *e; // Coarse grid corrections
  // Additional vectors are needed for HSS-type scales, i.e., scales where not
  // all of the lower scales are smoothed. This is because the smoothers only
  // operate on a smaller system that does not include the d.o.f. that are not
  // smoothed.
  CphisVec *fscale;
  CphisVec *uscale;
  CphisVec *urhs;
  // Total number of coarse scale solver iterations
  int iterCoarseScale;
  // Timers
  double timers[CPHIS_NUM_TIMERS];
};

#endif // __SOLVER_H__
