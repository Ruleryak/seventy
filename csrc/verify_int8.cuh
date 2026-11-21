#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace seventy {

/*
 * True if verify_int8 takes this shape: at most 48 query rows (query heads per KV head x new tokens) and a block size
 * that is a power of two.
 */
bool verify_int8_supported(int hq, int hkv, int max_query_len, int block_size);

/*
 * Partial verify attention over the INT8 KV cache (csrc/verify_int8.cu) into the workspace layout of
 * seventy_verify_attention: partial_o [tokens, hq, splits, 256], partial_ml [tokens, hq, splits, 2]. table_rows
 * (may be null: sequence b uses row b) gives the block-table row of each sequence. Sequences with more than
 * max_context cached keys are left out (no part written: seventy_verify_attention_hybrid).
 */
int launch_verify_int8(float *partial_o, float *partial_ml, const half *q, const int8_t *k_cache,
                       const int8_t *v_cache, const float *k_scale, const float *v_scale, const int *block_tables,
                       int table_width, const int *table_rows, const int *query_start, const int *context_lens,
                       int num_seqs, int block_size, int hq, int hkv, int splits, int min_split_keys, float scale,
                       cudaStream_t stream, int max_context = 0x7fffffff);

}  // namespace seventy
