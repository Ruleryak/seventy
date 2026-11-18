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
from ..config import load_config
from ..model.qwen3_5 import Qwen35TextModel
from ..parallel import Parallel
from . import distributed
from .cache import ModelCache
from .graphs import DecodeGraphs
from .step import SequenceChunk, StepInput


def build_step(payload, block_size: int, device) -> StepInput:
    chunks = [SequenceChunk(list(ids), ctx, list(table), slot, logits=mode) for ids, ctx, table, slot, mode in payload]
    return StepInput.build(chunks, block_size, device)


def load_shard(model_path: str, rank: int, size: int, device, dtype, layers=None):
    cfg = load_config(Path(model_path))
    with Checkpoint(Path(model_path)) as ckpt:
        model = Qwen35TextModel(cfg, ckpt, device, dtype, par=Parallel(rank, size), layers=layers)
    return cfg, model


def worker_main(rank: int, size: int, port: int, backend: str, device_name: str, model_path: str, dtype_name: str,
                utilization: float, reserve_bytes: int, conn, layers=None) -> None:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    try:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        distributed.init(rank, size, port, backend)
        dtype = getattr(torch, dtype_name)
        cfg, model = load_shard(model_path, rank, size, device, dtype, layers)
        budget = distributed.memory_budget(device, utilization, reserve_bytes) if device.type == "cuda" else 1 << 40
        conn.send(("ready", budget))
        kind, num_blocks, block_size, slots, kv_name = conn.recv()
        kv_dtype = getattr(torch, kv_name)
        cache = ModelCache(cfg.text, num_blocks, block_size, slots, dtype, device, par=Parallel(rank, size),
                           kv_dtype=kv_dtype)
        conn.send(("ok", None))
        graphs = None
        while True:
            msg = pickle.loads(conn.recv_bytes())
            if msg[0] == "exit":
                graphs = None
                break
            if msg[0] == "step":
                model.forward(build_step(msg[1], block_size, device), cache)
            elif msg[0] == "decode":
                graphs.run(msg[1])
            elif msg[0] == "copy":
                cache.apply(msg[1])
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
