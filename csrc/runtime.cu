#include "common.cuh"

SEVENTY_API int seventy_abi_version(void) { return SEVENTY_ABI_VERSION; }

SEVENTY_API const char *seventy_error_string(int code) {
  switch (code) {
    case SEVENTY_OK:
      return "no error";
    case SEVENTY_ERR_INVALID_ARGUMENT:
      return "invalid argument";
    case SEVENTY_ERR_MISALIGNED:
      return "pointer is not aligned for this kernel";
    default:
      break;
  }
  if (code > 0 && code < 10000) return cudaGetErrorString(static_cast<cudaError_t>(code));
  return "unknown error code";
}
