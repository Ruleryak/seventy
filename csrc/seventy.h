#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#if defined(__cplusplus)
#define SEVENTY_EXTERN extern "C"
#else
#define SEVENTY_EXTERN
#endif
#define SEVENTY_API SEVENTY_EXTERN __attribute__((visibility("default")))

#define SEVENTY_ABI_VERSION 1

/*
 * Every function returns 0 on success. A value below 10000 is a cudaError_t from the launch. A value from
 * 10000 up is one of the codes below. seventy_error_string() gives the text for any code.
 */
enum seventy_status {
  SEVENTY_OK = 0,
  SEVENTY_ERR_INVALID_ARGUMENT = 10001,
  SEVENTY_ERR_MISALIGNED = 10002,
};

SEVENTY_API int seventy_abi_version(void);
SEVENTY_API const char *seventy_error_string(int code);

/*
 * RMS norm of each row of a contiguous FP16 matrix [rows, dim].
 * Without a residual: out = fp16(x * rsqrt(mean(x^2) + eps) * weight).
 * With a residual: s = fp16(x + residual) goes to residual_out, and out is the norm of s.
 * weight is FP32 [dim] (the 1 + w offset of zero-centered norms is already in it).
 * out may alias x, and residual_out may alias residual. dim must be even.
 */
SEVENTY_API int seventy_rms_norm(void *out, void *residual_out, const void *x, const void *residual,
                                 const float *weight, int64_t rows, int dim, float eps, cudaStream_t stream);

/*
 * Dequantize an NVFP4 matrix [n, k] to FP16.
 * packed: uint8 [n, k / 2], two E2M1 values per byte, the even column in the low nibble.
 * scales: FP8 E4M3 bytes [n, k / 16], one scale for each group of 16 values along k.
 * value = e2m1 * (scale / global_scale). k must be a multiple of 16.
 */
SEVENTY_API int seventy_nvfp4_dequantize(void *out, const uint8_t *packed, const uint8_t *scales,
                                         float global_scale, int64_t n, int64_t k, cudaStream_t stream);
