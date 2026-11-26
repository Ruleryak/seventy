"""Load libseventy_kernels.so and declare the C functions it exports (see csrc/seventy.h)."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

ABI_VERSION = 1
LIB_NAME = "libseventy_kernels.so"

_P = ctypes.c_void_p
_I32 = ctypes.c_int
_I64 = ctypes.c_int64
_F32 = ctypes.c_float

SIGNATURES = {
    "seventy_abi_version": (_I32, []),
    "seventy_error_string": (ctypes.c_char_p, [_I32]),
    "seventy_rms_norm": (_I32, [_P, _P, _P, _P, _P, _I64, _I32, _F32, _P]),
    "seventy_nvfp4_dequantize": (_I32, [_P, _P, _P, _F32, _I64, _I64, _P]),
    "seventy_nvfp4_gemm_workspace": (_I64, [_I32, _I32, _I32]),
    "seventy_nvfp4_gemm_dequantize": (_I32, [_P, _P, _P, _P, _I32, _I32, _I32, _P]),
    "seventy_attn_prep": (_I32, [_P, _P, _P, _P, _P, _I32, _P, _I32, _P, _P, _P, _I64, _P, _I32, _I32, _I32, _P,
                                 _I32, _I32, _I32, _I32, _F32, _P]),
    "seventy_decode_attention_workspace": (_I64, [_I32, _I32, _I32]),
    "seventy_decode_attention": (_I32, [_P, _P, _P, _P, _P, _P, _I32, _P, _I32, _P, _I32, _P, _P, _I32, _I32, _I32,
                                        _I32, _I32, _I32, _I32, _I32, _F32, _P, _I64, _P]),
    "seventy_flash_attention": (_I32, [_P, _P, _P, _P, _P, _P, _I32, _P, _I32, _P, _I32, _P, _I32, _P, _P, _I32,
                                       _I32, _I32, _I32, _F32, _P]),
    "seventy_verify_attention": (_I32, [_P, _P, _P, _P, _P, _P, _I32, _P, _I32, _P, _I32, _P, _P, _I32, _I32, _I32,
                                        _I32, _I32, _I32, _I32, _I32, _I32, _F32, _P, _I64, _P]),
    "seventy_gdn_conv": (_I32, [_P, _P, _P, _I32, _P, _P, _P, _P, _I32, _I32, _I32, _I32, _P]),
    "seventy_gdn_gates": (_I32, [_P, _P, _P, _I32, _I32, _I32, _P, _P, _I32, _I32, _P]),
    "seventy_gated_rms_norm": (_I32, [_P, _P, _P, _I32, _I32, _P, _I32, _I32, _I32, _F32, _I32, _P]),
    "seventy_grouped_attention": (_I32, [_P, _P, _P, _P, _P, _P, _P, _I32, _P, _I32, _P, _P, _P, _I32, _I32, _I32,
                                         _I32, _I32, _I32, _I32, _I32, _I32, _F32, _P, _I64, _P]),
    "seventy_verify_attention_hybrid": (_I32, [_P, _P, _P, _P, _P, _P, _P, _I32, _P, _I32, _P, _I32, _P, _P,
                                               _I32, _I32, _I32, _I32, _I32, _I32, _I32, _I32, _I32, _I32,
                                               _I32, _F32, _P, _I64, _P]),
    "seventy_prefill_attention": (_I32, [_P, _P, _P, _P, _P, _P, _P, _I32, _P, _I32, _P, _I32, _P, _P, _I32, _I32,
                                         _I32, _I32, _I32, _I32, _I32, _I32, _I32, _F32, _P, _I64, _P]),
    "seventy_topk_workspace": (_I64, [_I32, _I32, _I32]),
    "seventy_topk": (_I32, [_P, _P, _P, _I64, _P, _I64, _I32, _I32, _I32, _I32, _P, _I64, _P]),
    "seventy_spec_accept": (_I32, [_P, _P, _P, _P, _P, _P, _I64, _P, _P, _P, _P, _P, _P, _P, _P, _P, _I32, _P, _I32,
                                   _I32, _I32, _I32, _P]),
    "seventy_drafter_walk": (_I32, [_P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _I32, _I32, _I32, _I32, _P]),
    "seventy_ar_buffer_bytes": (_I64, [_I32, _I64]),
    "seventy_ar_alloc": (_I32, [_I32, _I64, ctypes.POINTER(ctypes.c_void_p), _P]),
    "seventy_ar_open": (_I32, [_P, ctypes.POINTER(ctypes.c_void_p)]),
    "seventy_ar_context": (_I32, [_I32, _I32, _P, _I64, ctypes.POINTER(ctypes.c_void_p)]),
    "seventy_ar_guard": (_I32, [_P, _P, _I64]),
    "seventy_ar_sum_f32": (_I32, [_P, _P, _P, _I32, _I32, _P]),
    "seventy_ar_norm_2shot": (_I32, [_P, _P, _P, _P, _P, _I32, _I32, _F32, _I32, _P]),
    "seventy_ar_norm": (_I32, [_P, _P, _P, _P, _P, _I32, _I32, _F32, _P]),
    "seventy_dense_gemm_workspace": (_I64, [_I32, _I32, _I32]),
    "seventy_dense_gemm_fp8_wk_workspace": (_I64, [_I32, _I32, _I32]),
    "seventy_dense_gemm_fp8_wk": (_I32, [_P, _I32, _P, _P, _P, _I32, _I32, _I32, _I32, _I32, _I32, _P, _I64, _P, _P,
                                         _P]),
    "seventy_dense_gemm_fp8": (_I32, [_P, _I32, _P, _P, _P, _I32, _I32, _I32, _I32, _I32, _I32, _P, _I64, _P, _P]),
    "seventy_dense_gemm": (_I32, [_P, _I32, _P, _P, _I32, _I32, _I32, _I32, _I32, _I32, _P, _I64, _P, _P]),
    "seventy_drafter_norm": (_I32, [_P, _P, _P, _I32, _P, _P, _I32, _I32, _F32, _P]),
    "seventy_drafter_conv": (_I32, [_P, _I32, _P, _I32, _P, _P, _I64, _I32, _I32, _I32, _I32, _P]),
    "seventy_drafter_qk": (_I32, [_P, _P, _P, _P, _I64, _P, _I64, _P, _P, _P, _P, _P, _I32, _I32, _I32, _I32, _I32,
                                  _F32, _P]),
    "seventy_drafter_attention_splits": (_I32, [_I32, _I32, _I32, _I32]),
    "seventy_drafter_attention_workspace": (_I64, [_I32, _I32, _I32]),
    "seventy_drafter_attention": (_I32, [_P, _P, _P, _P, _P, _P, _I32, _I32, _I32, _I32, _I32, _I32, _I32, _F32, _P,
                                         _I64, _P]),
    "seventy_silu_mul": (_I32, [_P, _P, _I64, _I32, _P]),
    "seventy_silu_mul_scale": (_I32, [_P, _P, _P, _I32, _I32, _P]),
    "seventy_gdn_prepare": (_I32, [_P, _P, _P, _P, _P, _P, _P, _I32, _P, _P, _P, _P, _I32, _I32, _I32, _I32, _I32, _P]),
    "seventy_gdn_chunk_scan": (_I32, [_P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _I32, _I32, _I64, _P, _P, _P]),
    "seventy_gdn_commit_layers": (_I32, [_P, _P, _P, _I32, _I64, _I64, _I64, _P, _P, _P, _P, _P, _P, _P, _P,
                                         _I32, _I32, _I32, _I32, _I32, _I32, _I32, _F32, _P]),
    "seventy_gdn_recurrent": (_I32, [_P, _P, _P, _P, _I32, _P, _P, _P, _P, _P, _P, _P, _P, _I32, _I32, _I32, _I32,
                                     _I32, _I32, _I32, _F32, _I32, _P]),
    "seventy_nvfp4_gemm": (_I32, [_P, _P, _P, _P, _P, _I32, _I32, _I32, _I32, _I32, _I32, _P, _I64, _P]),
    "seventy_attention_combine": (_I32, [_P, _P, _P, _I32, _I32, _I32, _I32, _P]),
    "seventy_qsa_pool": (_I32, [_P, _P, _P, _I32, _P, _P, _P, _I32, _P, _P, _I32, _I32, _P, _P, _I32, _I32, _I32,
                                _I32, _I32, _F32, _P]),
    "seventy_qsa_index_prep": (_I32, [_P, _P, _P, _I64, _P, _P, _P, _P, _I32, _I32, _I32, _I32, _F32, _P]),
    "seventy_qsa_select_scores": (_I32, [_P, _I32, _P, _I32, _P, _P, _I32, _I32, _I32, _I32, _I32, _P]),
    "seventy_qsa_select": (_I32, [_P, _I32, _P, _I32, _P, _I32, _P, _P, _I32, _P, _P, _I32, _I32, _I32, _I32, _I32,
                                  _I32, _P]),
    "seventy_qsa_attention": (_I32, [_P, _P, _P, _P, _P, _P, _P, _P, _I32, _P, _I32, _P, _I32, _I32, _I32, _I32,
                                     _I32, _I32, _I32, _F32, _P]),
    "seventy_ple_ids": (_I32, [_P, _P, _P, _P, _P, _P, _I32, _I32, _I32, _I32, _P, _P, _P, _I64, _P]),
    "seventy_ple_gather": (_I32, [_P, _P, _I32, _I32, _I32, _I32, _P, _P, _I64, _I32, _F32, _P]),
    "seventy_ple_conv": (_I32, [_P, _P, _P, _P, _P, _P, _P, _P, _I32, _I32, _I32, _I32, _I32, _P]),
    "seventy_ple_state": (_I32, [_P, _P, _P, _P, _P, _P, _P, _I32, _I32, _I32, _I32, _I64, _P]),
    "seventy_hc_norm": (_I32, [_P, _P, _P, _I32, _I32, _I32, _F32, _P]),
    "seventy_hc_act": (_I32, [_P, _P, _P, _I32, _I32, _I32, _I32, _P]),
    "seventy_hc_combine": (_I32, [_P, _P, _P, _I32, _I32, _I32, _P]),
    "seventy_hc_inject": (_I32, [_P, _P, _P, _I32, _I32, _I32, _P]),
    "seventy_moe_route": (_I32, [_P, _I64, _I32, _I32, _I32, _I32, _P, _P, _P]),
    "seventy_moe_mma": (_I32, [_P, _P, _I32, _P, _P, _P, _I64, _I64, _I32, _I32, _I32, _P, _P, _P, _P, _P, _I32,
                               _I32, _I32, _P, _I32, _P]),
    "seventy_moe_finish": (_I32, [_P, _P, _P, _P, _I32, _I32, _I32, _I32, _P, _P, _I64, _I32, _P]),
    "seventy_moe_plan": (_I32, [_P, _I32, _I32, _I32, _I32, _I32, _P, _P, _P, _P, _P, _P]),
    "seventy_moe_gemm": (_I32, [_P, _P, _I32, _P, _P, _P, _I32, _I32, _I32, _P, _P, _P, _P, _P, _I32, _I32, _P, _I32,
                                _P]),
}


class KernelError(RuntimeError):
    pass


_lib: ctypes.CDLL | None = None
_path: Path | None = None
_error: KernelError | None = None


def candidates() -> list[Path]:
    """Places to look for the library, in order: SEVENTY_KERNELS_LIB, the package folder, then sys.path."""
    override = os.environ.get("SEVENTY_KERNELS_LIB")
    if override:
        return [Path(override)]
    found = [Path(__file__).resolve().parent.parent / "_lib" / LIB_NAME]
    found += [Path(entry) / "seventy" / "_lib" / LIB_NAME for entry in sys.path if entry]
    return found


def lib() -> ctypes.CDLL:
    global _lib, _path, _error
    if _lib is not None:
        return _lib
    if _error is not None:
        raise _error
    for path in candidates():
        if path.is_file():
            handle = ctypes.CDLL(str(path))
            missing = [name for name in SIGNATURES if not hasattr(handle, name)]
            if missing:
                _error = KernelError(f"{path} is from another version of seventy (it has no {missing[0]}); "
                                     "rebuild it or reinstall seventy")
                raise _error
            for name, (restype, argtypes) in SIGNATURES.items():
                fn = getattr(handle, name)
                fn.restype = restype
                fn.argtypes = argtypes
            version = handle.seventy_abi_version()
            if version != ABI_VERSION:
                _error = KernelError(f"{path} has kernel ABI {version}, this seventy needs {ABI_VERSION}; rebuild it")
                raise _error
            _lib, _path = handle, path
            return handle
    _error = KernelError(
        f"{LIB_NAME} not found (looked in: {', '.join(str(p.parent) for p in candidates()[:2])}). "
        "Install a seventy wheel, or build the kernels (see docs/DEVELOPING.md)."
    )
    raise _error


def available() -> bool:
    try:
        lib()
        return True
    except (KernelError, OSError):
        return False


def path() -> Path | None:
    return _path if available() else None


def check(name: str, code: int) -> None:
    if code:
        text = lib().seventy_error_string(code).decode()
        raise KernelError(f"{name} failed: {text} (code {code})")
