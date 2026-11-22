# Developing seventy

## What you need

- Linux on x86-64 (glibc 2.28 or later) with one or more NVIDIA V100 GPUs and a driver of branch 580 (the last
  branch for Volta; the tests ran with 580.178.04).
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

## Test an installed wheel

Use these steps to test a release wheel on a machine, for example on a new machine before or after a release.
The wheel and the tests must come from the same release.

1. Install the wheel with the test tools and the reference libraries (Hugging Face Transformers and others).
   The tests compare seventy with these libraries. From the release:

   ```
   uv venv --python 3.12 ~/seventy-test-env &&
   uv pip install --python ~/seventy-test-env --torch-backend=cu129 \
     "seventy[test,reference] @ https://github.com/Ruleryak/seventy/releases/download/v1.1.1/seventy-1.1.1-py3-none-manylinux_2_28_x86_64.whl"
   ```

   From a wheel file:

   ```
   uv venv --python 3.12 ~/seventy-test-env &&
   uv pip install --python ~/seventy-test-env --torch-backend=cu129 \
     "seventy[test,reference] @ file:///path/to/seventy-1.1.1-py3-none-manylinux_2_28_x86_64.whl"
   ```

2. Get the tests of the same release:

   ```
   git clone --depth 1 --branch v1.1.1 https://github.com/Ruleryak/seventy.git seventy-src
   ```

3. Install `objdump` for the tests of the kernel library. It is in the binutils package. On Ubuntu or Debian:

   ```
   sudo apt-get install -y binutils
   ```

   On RHEL, Rocky Linux or AlmaLinux:

   ```
   sudo dnf install -y binutils
   ```

4. Download the model for the tests that read the checkpoint:

   ```
   ~/seventy-test-env/bin/hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4
   ```

5. Run all tests with the `pytest` program of the environment, in the source folder:

   ```
   cd seventy-src && ~/seventy-test-env/bin/pytest
   ```

   Do not use `python -m pytest` here. It puts the current folder first on the module path, so Python imports
   the source folder, which has no kernel library, and the GPU tests skip.

On 4 V100 GPUs with peer-to-peer access, all tests pass. Without peer-to-peer access between the GPUs,
`tests/test_allreduce.py` skips: the engine does not use the custom all-reduce on these GPUs, and
`tests/test_tensor_parallel_gpu.py` tests the NCCL all-reduce that it uses. With fewer than 2 GPUs, the tests
for 2 or more GPUs skip.

The tests in `tests/test_model_reference.py` need about 50 GB of free host memory, and skip with less. The other
tests need about 16 GB.

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

The wheel is `dist/seventy-<version>-py3-none-manylinux_2_28_x86_64.whl`. It contains
`seventy/_lib/libseventy_kernels.so` compiled for sm70.

- The tag `manylinux_2_28` comes from `pyproject.toml`. `tests/test_kernel_library.py` makes sure that the
  library needs glibc 2.28 at most and links the shared CUDA runtime. Run it on each new build:

  ```
  SEVENTY_KERNELS_LIB=/path/to/libseventy_kernels.so pytest tests/test_kernel_library.py
  ```

- The dependencies in `pyproject.toml` are exact pins. Before a release, update them to the versions that the
  release tests ran with.

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
| `SEVENTY_VERIFY_ATTN` | `int8` | Verify attention on the verify kernel at every context length (not the split-D kernel from 32K cached keys) |
| `SEVENTY_VERIFY_SPLITD_CONTEXT` | a number of tokens | Verify attention on the split-D kernel from this many cached keys (default 32768) |
| `SEVENTY_VERIFY_ATTN` | `wmma` | Verify attention on the WMMA kernel |
| `SEVENTY_AR` | `home` | All-reduce and norm where the home rank of a row also takes the norm |
| `SEVENTY_AR` | `flags` | All-reduce on the flag protocol |
| `SEVENTY_CUSTOM_ALLREDUCE` | `0` | NCCL for all all-reduces (the engine does this itself when the GPUs have no peer-to-peer access) |
| `SEVENTY_GEMM_WK` | `0` | NVFP4 GEMM without the warp split over k |
| `SEVENTY_GDN_SCAN` | `single` | Gated DeltaNet prefill state scan in one kernel |
| `SEVENTY_GDN_PREPARE` | `torch` | Gated DeltaNet prefill preparation in PyTorch |
| `SEVENTY_GDN_COMMIT` | `loop` | Gated DeltaNet state commit after speculative verify in one launch per layer |
| `SEVENTY_DRAFTER_FUSED` | `0` | Drafter without the fused kernels |
| `SEVENTY_DRAFTER_FP8` | `0` | Drafter layers in FP16 |
| `SEVENTY_DRAFTER_FP8_WK` | `0` | Drafter FP8 GEMMs on the kernel with a k split over blocks and a reduce kernel |
| `SEVENTY_DRAFTER_FP8_EXTRA` | `none` | The drafter's fc, kv_all, selector and convolution projections in FP16 (or list the ones to keep in FP8, from `fc,kv_all,sel_hidden,conv`) |
| `SEVENTY_DRAFTER_FC_SPLIT` | `0` | The drafter's fc weight whole on every GPU (not split over the input columns) |
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
