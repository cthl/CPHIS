// io.c - implementation of functions related to terminal and file I/O

#include <cphis.h>
#include <linalg.h>
#include <aux.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

static void CphisPrintf_default(const char *format, ...)
{
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}

// Set the default function used to print messages.
void (*CphisPrintf)(const char*, ...) = CphisPrintf_default;

CphisError CphisSetPrintfFunc(void (*func)(const char*, ...))
{
  CphisPrintf = func;
  return CPHIS_SUCCESS;
}

void CphisPrintHline(int thick)
{
  if (thick) {
    CphisPrintf("==========================================================\n");
  }
  else {
    CphisPrintf("----------------------------------------------------------\n");
  }
}

CphisError CphisVecFromMatrixMarket(
             CphisVec *vec,
             int numLocalDOF,
             const char *file
           )
{
  // Open matrix market file.
  FILE *fp = fopen(file, "r");
  if (!fp) {
    CPHISCHECK(CPHIS_FAILED_IO);
  }

  // Allocate buffer (for one line).
  size_t bufferSize = 1024;
  char *buffer = malloc(bufferSize);
  if (!buffer) {
    fclose(fp);
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  // Read matrix dimensions.
  while (1) {
    if (!fgets(buffer, bufferSize, fp)) {
      free(buffer);
      fclose(fp);
      CPHISCHECK(CPHIS_FAILED_IO);
    }
    if (!strstr(buffer, "%")) {
      // Found the line that contains the matrix or vector dimensions.
      break;
    }
  }
  const CphisIndex numRows = (CphisIndex) strtol(buffer, NULL, 10);
  if (numRows == 0) {
    free(buffer);
    fclose(fp);
    CPHISCHECK(CPHIS_FAILED_IO);
  }
  if (numRows%numLocalDOF != 0) {
    free(buffer);
    fclose(fp);
    CPHISCHECK(CPHIS_INCOMPATIBLE);
  }

  // Create vector.
  CphisError err;
  err = CphisVecCreate(
          vec,
          numRows/numLocalDOF,
          NULL,
          numLocalDOF,
          CPHIS_BACKEND_DEFAULT
        );
  if (err) {
    free(buffer);
    fclose(fp);
    CPHISCHECK(err);
  }

  // Get vector data.
  CphisScalar *vecData;
  err = CphisVecGetData(*vec, &vecData);CPHISCHECK(err);

  // Read file line by line.
  CphisIndex i = 0;
  while (1) {
    // Read next line.
    if (!fgets(buffer, bufferSize, fp)) {
      // Reached end of matrix market file.
      break;
    }

    // Get value.
    const CphisScalar val = strtod(buffer, NULL);

    // Set value.
    vecData[i] = val;

    // Increment row index.
    i++;
  }

  free(buffer);
  fclose(fp);

  return CPHIS_SUCCESS;
}

CphisError CphisMatFromMatrixMarket(
             CphisMat *mat,
             int numLocalDOF,
             const char *file
           )
{
  // Open matrix market file.
  FILE *fp = fopen(file, "r");
  if (!fp) {
    CPHISCHECK(CPHIS_FAILED_IO);
  }

  // Allocate buffer (for one line).
  size_t bufferSize = 1024;
  char *buffer = malloc(bufferSize);
  if (!buffer) {
    fclose(fp);
    CPHISCHECK(CPHIS_FAILED_ALLOC);
  }

  // Read matrix dimensions.
  while (1) {
    if (!fgets(buffer, bufferSize, fp)) {
      free(buffer);
      fclose(fp);
      CPHISCHECK(CPHIS_FAILED_IO);
    }
    if (!strstr(buffer, "%")) {
      // Found the line that contains the matrix or vector dimensions.
      break;
    }
  }
  const CphisIndex numRows = (CphisIndex) strtol(buffer, NULL, 10);
  if (numRows == 0) {
    free(buffer);
    fclose(fp);
    CPHISCHECK(CPHIS_FAILED_IO);
  }
  if (numRows%numLocalDOF != 0) {
    free(buffer);
    fclose(fp);
    CPHISCHECK(CPHIS_INCOMPATIBLE);
  }

  // Create matrix.
  CphisError err;
  err = CphisMatCreate(
          mat,
          numRows/numLocalDOF,
          NULL,
          numLocalDOF,
          CPHIS_BACKEND_DEFAULT
        );
  if (err) {
    free(buffer);
    fclose(fp);
    CPHISCHECK(err);
  }

  // Read file line by line.
  while (1) {
    // Read next line.
    if (!fgets(buffer, bufferSize, fp)) {
      // Reached end of matrix market file.
      break;
    }

    // Get row, column, and (optional) value.
    char *ptr, *eptr;
    ptr = buffer;
    const CphisIndex row = strtol(ptr, &eptr, 10);
    if (ptr == eptr) {
      // Could not find row index.
      CphisMatDestroy(*mat);
      free(buffer);
      fclose(fp);
      CPHISCHECK(CPHIS_FAILED_IO);
    }
    ptr = eptr;
    const CphisIndex col = strtol(ptr, &eptr, 10);
    if (ptr == eptr) {
      // Could not find column index.
      CphisMatDestroy(*mat);
      free(buffer);
      fclose(fp);
      CPHISCHECK(CPHIS_FAILED_IO);
    }
    ptr = eptr;
    CphisScalar val = strtod(ptr, &eptr);
    if (ptr == eptr) {
      // Value was not specified, so it must be one as per matrix market format.
      val = 1.0;
    }
    else if (val == 0.0) {
      // We found a true zero entry, which we will not add to the matrix.
      continue;
    }

    // Set matrix entry. Convert from 1-based to 0-based indices.
    err = CphisMatSet(*mat, row - 1, col - 1, val);
    if (err) {
      CphisMatDestroy(*mat);
      free(buffer);
      fclose(fp);
      CPHISCHECK(err);
    }
  }

  free(buffer);
  fclose(fp);

  // Finalize matrix.
  err = CphisMatFinalize(*mat);CPHISCHECK(err);
  return CPHIS_SUCCESS;
}

CphisError CphisVecPrint(const CphisVec vec)
{
  if (vec->type != CPHIS_BACKEND_DEFAULT) {
    CPHISCHECK(CPHIS_INCOMPATIBLE);
  }

  CphisError err;
  CphisScalar *vecData;
  err = CphisVecGetData(vec, &vecData);CPHISCHECK(err);
  for (CphisIndex k = 0; k < vec->numElements; k++) {
    for (int l = 0; l < vec->numLocalDOF; l++) {
      const CphisIndex row = k*vec->numLocalDOF + l;
      printf("%d [%d, %d]: %e\n", row, k, l, vecData[row]);
    }
  }

  return CPHIS_SUCCESS;
}

CphisError CphisMatPrint(const CphisMat mat)
{
  if (mat->type != CPHIS_BACKEND_DEFAULT) {
    CPHISCHECK(CPHIS_INCOMPATIBLE);
  }
  if (!mat->finalized) {
    CPHISCHECK(CPHIS_INVALID_STATE);
  }

  CphisError err;
  const CphisIndex *cols;
  const CphisScalar *vals;
  CphisIndex numEntries;
  for (CphisIndex k = 0; k < mat->numElements; k++) {
    for (int l = 0; l < mat->numLocalDOF; l++) {
      const CphisIndex row = k*mat->numLocalDOF + l;
      err = CphisMatGetData(
              mat,
              row,
              &cols,
              &vals,
              &numEntries
            );CPHISCHECK(err);
      for (CphisIndex j = 0; j < numEntries; j++) {
        printf(
          "(%d, %d) ([%d, %d], [%d, %d]): %e\n",
          row,
          cols[j],
          k,
          l,
          cols[j]/mat->numLocalDOF,
          cols[j]%mat->numLocalDOF,
          vals[j]
        );
      }
    }
  }
  return CPHIS_SUCCESS;
}
