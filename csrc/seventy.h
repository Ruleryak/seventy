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

/*
 * Bytes of FP32 workspace that seventy_nvfp4_gemm needs for m rows, n_padded columns and `splits` k splits.
 */
SEVENTY_API int64_t seventy_nvfp4_gemm_workspace(int m, int n_padded, int splits);

/*
 * out[m, n] (FP16, row stride n) = alpha[n] * x[m, k] (FP16, row stride k) times the NVFP4 weight [n, k],
 * repacked by seventy.nvfp4.pack_for_gemm into wpk (16 bytes per lane, column and 32-wide k chunk) and wsc (the
 * E4M3 group scales, 4 per lane and 64-wide k block). alpha holds 2^7 / global_scale for each output column.
 * 1 <= m <= 64, n_padded is n rounded up to 32, k is a multiple of 64. splits >= 1 divides k between blocks;
 * with splits > 1 the workspace must hold seventy_nvfp4_gemm_workspace() bytes. warps (2, 4 or 8) is the number of
 * 32-column tiles per block.
 */
SEVENTY_API int seventy_nvfp4_gemm(void *out, const void *x, const void *wpk, const void *wsc, const float *alpha,
                                   int m, int n, int n_padded, int k, int splits, int warps, void *workspace,
                                   int64_t workspace_bytes, cudaStream_t stream);

/*
 * The dense FP16 weight [n, k] of a weight repacked for seventy_nvfp4_gemm: e2m1 * e4m3 * alpha / 2^7, rounded once.
 */
SEVENTY_API int seventy_nvfp4_gemm_dequantize(void *out, const void *wpk, const void *wsc, const float *alpha, int n,
                                              int n_padded, int k, cudaStream_t stream);
