"""A tensor-parallel worker: holds one shard of the model and its caches on one GPU, and runs the steps that
rank 0 sends. All ranks run every step together, because the model's collectives need every rank."""

from __future__ import annotations

import faulthandler
import os
import pickle
import signal
import traceback
from pathlib import Path

import torch

from ..checkpoint import Checkpoint
from .. import upload
from ..config import load_config
from ..model.qwen3_5 import Qwen35TextModel
from ..parallel import Parallel
from . import distributed
from .cache import ModelCache
from .graphs import DecodeGraphs
from .runner import ModelRunner, build_step


def load_shard(model_path: str, rank: int, size: int, device, dtype, layers=None):
    """This rank's shard of the model, with the peer-to-peer all-reduce when the start check passes on every rank.
    With tensor parallel, it ends with a first NCCL all-reduce: NCCL then makes its communicator (and its buffers
    in GPU memory) before the caller measures the memory that is left for the KV cache."""
    cfg = load_config(Path(model_path))
    with Checkpoint(Path(model_path)) as ckpt:
        model = Qwen35TextModel(cfg, ckpt, device, dtype, par=Parallel(rank, size), layers=layers)
        model.vision = None
        prefix = "model.visual."
        if cfg.vision is not None and torch.device(device).type == "cuda" and ckpt.names(prefix):
            from ..model.vision import VisionTower

            model.vision = VisionTower(cfg.vision, ckpt, device, par=Parallel(rank, size), prefix=prefix)
    if (size > 1 and torch.device(device).type == "cuda" and os.environ.get("SEVENTY_CUSTOM_ALLREDUCE", "1") != "0"
            and distributed.peer_access(size)):
        import dataclasses

        from .allreduce import CustomAllReduce

        custom = CustomAllReduce(rank, size, torch.device(device), dim=cfg.text.hidden_size)
        if custom.usable:
            model.par = dataclasses.replace(model.par, custom=custom)
        elif rank == 0:
            print(f"seventy: the GPUs report peer access, but the peer-to-peer all-reduce check failed "
                  f"({custom.problem}); every all-reduce uses NCCL without peer-to-peer transfers "
                  f"(NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')}), which is slower", flush=True)
    if size > 1 and torch.device(device).type == "cuda":
        torch.distributed.all_reduce(torch.zeros(1, device=device))
    return cfg, model


def _watch_parent() -> None:
    """End this worker when rank 0's process is gone (it may be waiting in a collective that never completes)."""
    import threading
    import time

    parent = os.getppid()

    def watch():
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                os._exit(71)

    threading.Thread(target=watch, daemon=True, name="parent-watch").start()


def worker_main(rank: int, size: int, port: int, backend: str, device_name: str, model_path: str, dtype_name: str,
                utilization: float, reserve_bytes: int, conn, layers=None, draft_path: str | None = None) -> None:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _watch_parent()
    distributed.share_cpus(size)
    try:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        distributed.init(rank, size, port, backend)
        dtype = getattr(torch, dtype_name)
        cfg, model = load_shard(model_path, rank, size, device, dtype, layers)
        drafter = None
        if draft_path:
            from .engine import make_drafter

            drafter = make_drafter(draft_path, model_path, model, device, dtype, model.par)
        mtp = bool(draft_path) and draft_path.startswith("mtp:")
        upload.release()
        budget = (distributed.memory_budget(device, utilization, reserve_bytes) if device.type == "cuda"
                  else (1 << 40, 1 << 40))
        conn.send(("ready", budget))
        kind, num_blocks, block_size, slots, kv_name = conn.recv()
        kv_dtype = getattr(torch, kv_name)
        cache = ModelCache(cfg.text, num_blocks, block_size, slots, dtype, device, par=Parallel(rank, size),
                           kv_dtype=kv_dtype, extra_attention_layers=(cfg.text.num_layers,) if mtp else ())
        if drafter is not None:
            drafter.allocate(slots)
            cache.slot_peers.append(drafter)
        runner = ModelRunner(model, cache, device, (), drafter)
        conn.send(("ok", None))
        graphs = None
        while True:
            msg = pickle.loads(conn.recv_bytes())
            if msg[0] == "exit":
                graphs = None
                runner.spec_graphs = None
                break
            if msg[0] == "step":
                runner.run_payload(msg[1], msg[2])
            elif msg[0] == "verify":
                runner.verify_payload(msg[1])
            elif msg[0] == "accept":
                runner.accept(*msg[1:])
            elif msg[0] == "context":
                runner.context(*msg[1:], send=False)
            elif msg[0] == "draft":
                runner.draft(msg[1], msg[2], msg[3], [0.0] * len(msg[1]), [None] * len(msg[1]))
            elif msg[0] == "mtp_prefill":
                runner.mtp_prefill(msg[1], send=False)
            elif msg[0] == "mtp_first":
                runner.mtp_first(*msg[1:], send=False)
            elif msg[0] == "mtp_verify":
                runner.mtp_verify(*msg[1:], send=False)
            elif msg[0] == "decode":
                graphs.run(msg[1])
            elif msg[0] == "copy":
                cache.apply(msg[1])
            elif msg[0] == "encode":
                runner.encode_local(msg[1], msg[2])
            elif msg[0] == "release":
                runner.release_local(msg[1])
            elif msg[0] == "spec":
                runner.spec_graphs.run_packed(*msg[1])
            elif msg[0] == "capture_spec":
                from .spec_graphs import SpecGraphs

                runner.spec_graphs = SpecGraphs(model, drafter, cache, device, msg[1], tuple(msg[2]))
                runner.spec_graphs.capture()
                conn.send(("captured", None))
            elif msg[0] == "capture":
                graphs = DecodeGraphs(model, cache, device, msg[1], tuple(msg[2]))
                graphs.capture()
                conn.send(("captured", None))
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
        raise
    finally:
        distributed.shutdown()
