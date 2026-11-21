# Developing seventy

## What you need

- Linux on x86-64 with one or more NVIDIA V100 GPUs and driver 525 or later (driver branch 580 is the last one for
  Volta).
- The CUDA 12.9 toolkit, for `nvcc`. CUDA 13 cannot compile for sm70.
- [uv](https://docs.astral.sh/uv/), or Python 3.12 with pip.

## Build and install for development

1. Make a virtual environment:

   ```
   uv venv --python 3.12 .venv
   ```

2. Install seventy in editable mode with the test tools. This compiles the kernels. Set `CUDACXX` to your
   `nvcc` if it is not on `PATH`:

   ```
   CUDACXX=/usr/local/cuda-12.9/bin/nvcc uv pip install -e ".[test]"
   ```

   With pip instead of uv, add the PyTorch index for CUDA 12.9:

   ```
   CUDACXX=/usr/local/cuda-12.9/bin/nvcc pip install -e ".[test]" --extra-index-url https://download.pytorch.org/whl/cu129
   ```

   After a change in `csrc/`, run the same install command again to rebuild the kernels.

3. Check the result:

   ```
   .venv/bin/seventy info
   ```

## Run the tests

- All tests that do not need a GPU or the model:

  ```
  .venv/bin/pytest -m "not gpu and not model"
  ```

- Kernel tests on one GPU (they use less than 1 GiB of GPU memory):

  ```
  CUDA_VISIBLE_DEVICES=0 .venv/bin/pytest -m gpu
  ```

- Tests that read the model checkpoint. They find `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` in the Hugging Face
  cache, or the folder in `SEVENTY_TEST_MODEL`:

  ```
  SEVENTY_TEST_MODEL=/path/to/checkpoint .venv/bin/pytest -m model
  ```

## Run from the source tree

- The control panel, for this machine only (open http://127.0.0.1:8270/):

  ```
  .venv/bin/seventy panel
  ```

- The control panel, open to the network. It prints a URL with the UI token:

  ```
  .venv/bin/seventy panel --host 0.0.0.0
  ```

- The control panel with a separate home folder, so a test does not change your settings or history:

  ```
  .venv/bin/seventy panel --port 8271 --home /path/to/test-home
  ```

- The engine alone, without the panel (4 GPUs, with the DFlash2 drafter):

  ```
  .venv/bin/seventy engine --model /path/to/model --draft-model /path/to/drafter --gpus 0,1,2,3
  ```

You can run each command again; the panel keeps its settings and history in its home folder.

## Build the wheel

```
CUDACXX=/usr/local/cuda-12.9/bin/nvcc uv build --wheel
```

The wheel is `dist/seventy-<version>-py3-none-linux_x86_64.whl`. It contains `seventy/_lib/libseventy_kernels.so`
compiled for sm70.

## The kernel library

- The CUDA sources are in `csrc/`. `csrc/seventy.h` declares the C functions. Each returns 0 or an error code.
- Python loads the library with `ctypes` (`seventy/ops/_lib.py`). To test another build of the library, set
  `SEVENTY_KERNELS_LIB` to its path.
- Each kernel has a PyTorch reference in `seventy/ops/reference.py`, and a test in `tests/` that compares the two.
- For profiling, build with line information: add `--config-settings=cmake.define.SEVENTY_LINEINFO=ON` to the
  install command.

## Compare kernel paths

Some parts have an earlier implementation that you can select for an A/B comparison. Set the variable for the
engine or for the tests. Without the variable, the default path runs.

| Variable | Value | Effect |
|---|---|---|
| `SEVENTY_PREFILL_ATTN` | `grouped` | Prefill attention on the verify kernel |
| `SEVENTY_PREFILL_ATTN` | `tiles` | Prefill attention on the per-head prefill kernel |
| `SEVENTY_VERIFY_ATTN` | `wmma` | Verify attention on the WMMA kernel |
| `SEVENTY_AR` | `home` | All-reduce and norm where the home rank of a row also takes the norm |
| `SEVENTY_AR` | `flags` | All-reduce on the flag protocol |
| `SEVENTY_CUSTOM_ALLREDUCE` | `0` | NCCL for all all-reduces |
| `SEVENTY_GEMM_WK` | `0` | NVFP4 GEMM without the warp split over k |
| `SEVENTY_GDN_SCAN` | `single` | Gated DeltaNet prefill state scan in one kernel |
| `SEVENTY_GDN_PREPARE` | `torch` | Gated DeltaNet prefill preparation in PyTorch |
| `SEVENTY_DRAFTER_FUSED` | `0` | Drafter without the fused kernels |
| `SEVENTY_DRAFTER_FP8` | `0` | Drafter layers in FP16 |
| `SEVENTY_DRAFTER_FP8_EXTRA` | `fc,kv_all,sel_hidden,conv` | These drafter projections in FP8 too (fewer names select fewer) |
| `SEVENTY_DRAFTER_WALK` | `torch` | Drafter selector walk in PyTorch |
| `SEVENTY_SPEC_ACCEPT` | `torch` | Speculative acceptance in PyTorch |

For example, to run the engine with the earlier prefill attention:

```
SEVENTY_PREFILL_ATTN=grouped .venv/bin/seventy engine --model /path/to/model --draft-model /path/to/drafter --gpus 0,1,2,3
```

## Speed charts

`bench/results/` has one results file for each release. The docstring of `bench/report.py` gives the format.

To measure a running server and write a results file (2K, 32K and 100K tokens of context with 1, 4 and 8 streams,
two runs of each, about 15 minutes on 4 GPUs):

```
.venv/bin/python bench/sweep.py --url http://127.0.0.1:8000 --release 1.1.0 --out bench/results/1.1.0.json
```

To measure longer contexts too, give the context lengths. The KV cache must hold all streams of the largest cell:

```
.venv/bin/python bench/sweep.py --url http://127.0.0.1:8000 --contexts 2000,32000,100000,128000,192000,256000 --release 1.1.0 --out bench/results/1.1.0.json
```

To make the README charts again (light and dark) from the newest results file:

```
python bench/report.py --images docs/images
```

To print the README tables:

```
python bench/report.py --markdown
```

To make a page that shows each release over the release before it:

```
python bench/report.py --html speed-sheet.html
```
