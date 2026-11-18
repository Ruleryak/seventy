"""Command line: `seventy <command>`."""

from __future__ import annotations

import argparse
import platform
import sys

from . import __version__


def cmd_info(args: argparse.Namespace) -> int:
    import torch

    from .ops import _lib

    print(f"seventy {__version__}")
    print(f"python {platform.python_version()}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("GPUs: none visible")
    else:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, sm{p.major}{p.minor}")
    if _lib.available():
        print(f"kernels: {_lib.path()} (ABI {_lib.ABI_VERSION})")
    else:
        try:
            _lib.lib()
        except Exception as e:
            print(f"kernels: not available: {e}")
    return 0


def cmd_check_model(args: argparse.Namespace) -> int:
    from .config import UnsupportedModel, load_config

    try:
        cfg = load_config(args.path)
    except UnsupportedModel as e:
        print("seventy cannot serve this model:")
        for reason in e.reasons:
            print(f"  - {reason}")
        return 1
    t = cfg.text
    print(f"{cfg.architecture}: {t.num_layers} layers ({len(t.linear_layers)} Gated DeltaNet, "
          f"{len(t.attention_layers)} attention), hidden {t.hidden_size}, weights {cfg.quant.method}, "
          f"vision {'yes' if cfg.vision else 'no'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seventy", description=__doc__)
    parser.add_argument("--version", action="version", version=f"seventy {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show versions, GPUs and the kernel library").set_defaults(fn=cmd_info)
    check = sub.add_parser("check-model", help="check whether seventy can serve a checkpoint folder")
    check.add_argument("path")
    check.set_defaults(fn=cmd_check_model)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
