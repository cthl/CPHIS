// conf.h - declaration of configuration functions and structures

#ifndef __CONF_H__
#define __CONF_H__

#include <cphis.h>

struct _CphisConf
{
  // Number of degrees of freedom per element
  int numLocalDOF;
  // Number of CPHIS scales
  int numScales;
  // Array containing the index of the first and last local degree of freedom on
  // each scale. The entries scales[2*i] and scales[2*i + 1] determine the
  // beginning and the end of the ith scale.
  // Note that scales can be overlapping.
  int *scales;
  // Array of solver handles, one for each scale.
  CphisScaleSolver *solvers;
  // Array of flags indicating whether we need to destroy a solver or if it is
  // the user's responsibility.
  int *solversOwned;
  // Relative tolerance
  CphisReal tol;
  // Number of times the scale solver is called before cycling (nu1) and after
  // cycling (nu2). These parameters are ignored for direct, i.e.,
  // non-iterative scale solvers.
  int nu1;
  int nu2;
  // Maximum number of iterations (Cycles)
  int maxIter;
  // Type of cycle to be used
  CphisCycleType cycle;
  // Whether or not to use FMG-type cycles
  int fmgCycle;
  // Verbosity level
  CphisVerbosityLevel verbosity;
};

#endif // __CONF_H__
