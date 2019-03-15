// aux.h - declaration of auxiliary functions and types for internal use

#ifndef __AUX_H__
#define __AUX_H__

// Pointer to print function
extern void (*CphisPrintf)(const char*, ...);

// Print a horizontal line.
void CphisPrintHline(int thick);

// Print a vector or a matrix (only for debugging with the default backend).
CphisError CphisVecPrint(const CphisVec vec);
CphisError CphisMatPrint(const CphisMat mat);

#endif // __AUX_H__
