// errors.c - implementation of functions related to error handling

#include <cphis.h>
#include <aux.h>
#include <stdio.h>

CphisError CphisErrorMsg(
             CphisError err,
             const char *func,
             const char *file,
             int line
           )
{
  if (err < 0) {
    // Error has been forwarded and message was already printed elsewhere.
    CphisPrintf("             ^ called from %s(), %s:%d\n", func, file, line);
    return CPHIS_SUCCESS;
  }

  switch (err) {
    case CPHIS_FAILED_IO:
      CphisPrintf(
        "CPHIS error: I/O operation failed (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    case CPHIS_INVALID_ARG:
      CphisPrintf(
        "CPHIS error: Invalid argument (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    case CPHIS_INVALID_STATE:
      CphisPrintf(
        "CPHIS error: Object is in invalid state (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    case CPHIS_UNKNOWN_TYPE:
      CphisPrintf(
        "CPHIS error: Unknown vector or matrix type (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    case CPHIS_INCOMPATIBLE:
      CphisPrintf(
        "CPHIS error: Incompatible arguments (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    case CPHIS_OUT_OF_BOUNDS:
      CphisPrintf(
        "CPHIS error: Index out of bounds (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    case CPHIS_UNINITIALIZED:
      CphisPrintf(
        "CPHIS error: Uninitialized object or structure (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    case CPHIS_FAILED_ALLOC:
      CphisPrintf(
        "CPHIS error: Memory allocation failed (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    case CPHIS_NOT_IMPLEMENTED:
      CphisPrintf(
        "CPHIS error: Feature has not been implemented (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    case CPHIS_ERROR_IN_THREAD:
      CphisPrintf(
        "CPHIS error: An error occured inside an OpenMP region (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    case CPHIS_TEST_FAILED:
      CphisPrintf(
        "CPHIS error: Test failed (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    #ifdef CPHIS_HAVE_TPETRA
    case CPHIS_TPETRA_ERROR:
      CphisPrintf(
        "CPHIS error: Caught an exception from Tpetra (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    #endif
    case CPHIS_MPI_ERROR:
      CphisPrintf(
        "CPHIS error: An error occured calling an MPI function (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
    default:
      CphisPrintf(
        "CPHIS error: Unknown error (%s(), %s:%d)\n",
        func,
        file,
        line
      );
      break;
  }

  return CPHIS_SUCCESS;
}
