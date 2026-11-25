# Mixture-of-experts models (branch `moe`)

This branch adds Qwen mixture-of-experts models to seventy: Qwen3.5-MoE (for example Qwen3.6-35B-A3B) and
Qwen4-Exp (Qwen3.8 Flash-Next). The work is not finished. This page gives the state of the branch.

## The checkpoints

| Model | Checkpoint | Format | Why this one |
|---|---|---|---|
| Qwen3.8 Flash-Next | `nvidia/Qwen3.8-Flash-Next-FP4-Local` | ModelOpt mixed: routed experts NVFP4 trained with quantization-aware distillation (QAD), attention, Gated DeltaNet and shared experts MXFP8, n-gram table NVFP4 (also QAD), block scales CSF-coded | QAD: GPQA Diamond 91.5 against 92.2 for BF16 and ToolEvalBench 88.7 against 89.4 (nvidia's model card). The smallest checkpoint with all 512 experts (95 GiB). |
| Qwen3.8 Flash-Next | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | The same QAD text weights (byte for byte), plain block scales, BF16 vision tower | Read by the same code; its vision tower is not quantized. |
| Qwen3.6-35B-A3B | `nvidia/Qwen3.6-35B-A3B-NVFP4` | ModelOpt mixed: experts, shared experts and LM head NVFP4 (weights only), attention and Gated DeltaNet FP8 with one scale per tensor | No NVFP4 checkpoint of this model with quantization-aware training or distillation exists. This post-training quantization scores like BF16 in unsloth's comparison (MMLU-Pro 85.60 against 85.75, GPQA 87.12 against 86.36, AIME 2025 91.88 against 92.50). |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | BF16 (full weights) | The full-weight model, on 2 or more 32 GB GPUs. |

Other Flash-Next checkpoints that were compared:

- `RadixArk/Qwen3.8-Flash-Next-NVFP4`: post-training quantization of the routed experts only (calibration: 128
  news articles); attention, Gated DeltaNet and shared experts stay BF16, so each token reads more bytes; 126 GiB.
- `nvidia/Qwen3.8-Flash-Next-NVFP4`: post-training quantization (ModelOpt), 124 GiB, scores close to FP8.
- `autotrust/Qwen3.8-Flash-Next-E384-FP4-*`: the QAD checkpoint with 384 of the 512 experts kept (80 GiB); it loses
  5 to 9 points on knowledge benchmarks (GPQA, MMLU). The V2 checkpoint is also fine-tuned for security tasks.
- GGUF checkpoints (the most downloaded, for example `ISTA-DASLab/...-GSQ-RCO-GGUF`) and AWQ or AutoRound INT4
  checkpoints: formats that seventy does not read.

## What the branch has

- The MoE MLP (`seventy/model/moe.py`): the router (softmax over all experts, top k, the k weights renormalized),
  the routed experts split over the GPUs (expert parallel: each GPU keeps a part of the experts) and the shared
  expert split like a dense MLP. The partial sums meet in the all-reduce after the MLP.
- Kernels (`csrc/moe.cu`): the router's top k, a plan that sorts the (token, expert) pairs of each GPU's experts on
  the GPU, and a grouped NVFP4 or FP16 GEMM over the plan. Decode and speculative steps (up to 64 tokens) have no
  host synchronization and run in CUDA graphs. Prefill runs one product per picked expert.
- Checkpoint formats (`seventy/checkpoint.py`): ModelOpt NVFP4 (with CSF-coded block scales), FP8 with one scale per
  tensor or row (kept as FP8 for the FP8 GEMM), MXFP8 (loaded as FP16), and the fused experts of BF16 checkpoints.
- 1, 2, 3 and 4 GPUs. With 2 KV heads and 3 or 4 GPUs, the GPUs share each KV head (`seventy/split.py`).
- The MTP drafter for MoE models.
- The Qwen4-Exp text model (`seventy/model/qwen4_exp.py`): hyper-connections, the n-gram embeddings (PLE) from the
  memory-mapped table in host memory, the QSA indexer and sparse attention, the sigmoid gate of its Gated DeltaNet
  layers. This model has a CPU path only so far.

## Tests

- `tests/test_moe_reference.py` and `tests/test_flash_next_reference.py`: the first layers of the real checkpoints
  against transformers, in FP32 on the CPU.
- `tests/test_moe_kernels.py`, `tests/test_checkpoint_formats.py`, `tests/test_fp8_linear.py`: the kernels and the
  formats against PyTorch references.

## Not done

- Flash-Next on the GPU: QSA kernels (indexer scores, block top-k, sparse attention over the paged cache), the
  n-gram table read on the GPU from host memory, CUDA graphs, its MTP layer and its DFlash2 drafter.
- Faster MoE decode: the shared expert in the grouped GEMM, fewer small kernels per layer, MXFP8 GEMMs.
- The panel's checks for MoE checkpoints (sizes of the expert weights per GPU).
