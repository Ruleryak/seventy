"""Runs scheduled work on the model: builds the step input, sends it to the tensor-parallel workers, runs the
forward pass, and returns the logits (rank 0).

A step payload has one tuple per sequence: (token ids, context length, block table, state slot, logits mode,
rope, image rows). rope is an int delta (positions are token index + delta on all axes) or explicit [3][n]
positions; image rows are (row in the chunk, image key, first image token, tokens), whose embeddings every rank
holds from an earlier encode() until release()."""

from __future__ import annotations

import pickle

import torch

from ..engine.cache import ModelCache
from .graphs import DecodeGraphs
from .scheduler import Work
from .spec_graphs import SpecGraphs
from .step import SequenceChunk, StepInput


def _positions(rope, ctx: int, n: int):
    if isinstance(rope, int):
        return None if rope == 0 else [list(range(ctx + rope, ctx + n + rope))] * 3
    return rope


def build_step(payload, block_size: int, device, images: dict | None = None) -> StepInput:
    chunks = [SequenceChunk(list(ids), ctx, list(table), slot, positions=_positions(rope, ctx, len(ids)), logits=mode)
              for ids, ctx, table, slot, mode, rope, _ in payload]
    step = StepInput.build(chunks, block_size, device)
    for b, item in enumerate(payload):
        for row, key, first, count in item[6]:
            step.embeds[step.query_start[b] + row] = images[key][first : first + count]
    return step


class ModelRunner:
    def __init__(self, model, cache: ModelCache, device: torch.device, peers=(), drafter=None):
        self.model = model
        self.cache = cache
        self.device = device
        self.peers = list(peers)
        self.graphs: DecodeGraphs | None = None
        self.drafter = drafter
        self.mtp = drafter is not None and hasattr(drafter, "after_verify")
        self.layers = drafter.cfg.target_layers if drafter is not None else ()
        self._run_step = None
        self._feats = None
        self._stash = None
        self._step = None
        self.spec_graphs: SpecGraphs | None = None
        self.images: dict[str, torch.Tensor] = {}

    def capture_spec(self, max_blocks_per_seq: int, buckets: tuple[int, ...]) -> None:
        self._send(("capture_spec", max_blocks_per_seq, list(buckets)))
        self.spec_graphs = SpecGraphs(self.model, self.drafter, self.cache, self.device, max_blocks_per_seq, buckets)
        self.spec_graphs.capture()
        for p in self.peers:
            if not p.poll(900):
                raise RuntimeError("a tensor-parallel worker did not finish capturing CUDA graphs")
            reply = p.recv()
            if reply[0] != "captured":
                raise RuntimeError(f"a worker failed to capture CUDA graphs: {reply}")

    def spec_step(self, rows: list[dict]) -> dict:
        packed = self.spec_graphs.pack(rows)
        self._send(("spec", packed))
        return self.spec_graphs.run_packed(*packed)

    def _send(self, msg) -> None:
        if self.peers:
            data = pickle.dumps(msg)
            for p in self.peers:
                p.send_bytes(data)

    @staticmethod
    def payload(work: list[Work]) -> list[tuple]:
        out = []
        for w in work:
            r = w.request
            rows = [(row, f"{r.id}/{i}", first, n) for row, i, first, n in r.image_rows(w.start, w.count)]
            out.append((w.ids(), w.start, r.blocks, r.slot, w.logits, r.rope(w.start, w.count), rows))
        return out

    def encode(self, key: str, pixels) -> None:
        """Run the vision tower on a resized uint8 image [H, W, 3] on every rank; keep the embeddings as key."""
        self._send(("encode", key, pixels))
        self.encode_local(key, pixels)

    def encode_local(self, key: str, pixels) -> None:
        self.images[key] = self.model.vision.encode(torch.from_numpy(pixels))

    def release(self, keys: list[str]) -> None:
        if keys:
            self._send(("release", keys))
            self.release_local(keys)

    def release_local(self, keys: list[str]) -> None:
        for k in keys:
            self.images.pop(k, None)

    def capture(self, max_blocks_per_seq: int, buckets: tuple[int, ...]) -> None:
        msg = pickle.dumps(("capture", max_blocks_per_seq, list(buckets)))
        for p in self.peers:
            p.send_bytes(msg)
        self.graphs = DecodeGraphs(self.model, self.cache, self.device, max_blocks_per_seq, buckets)
        self.graphs.capture()
        for p in self.peers:
            if not p.poll(900):
                raise RuntimeError("a tensor-parallel worker did not finish capturing CUDA graphs")
            reply = p.recv()
            if reply[0] != "captured":
                raise RuntimeError(f"a worker failed to capture CUDA graphs: {reply}")

    def copy(self, copies: list[tuple[str, int, int]]) -> None:
        """Make the plan's cache copies (prefix cache) on every rank before the step."""
        self._send(("copy", copies))
        self.cache.apply(copies)

    def run_decode(self, work: list[Work]) -> torch.Tensor:
        """Decode work (one new token per sequence) through the CUDA graphs when a bucket fits."""
        if self.graphs is None or self.graphs.bucket(len(work)) is None:
            return self.run(work)
        payload = [(w.request.token_range(w.start, w.start + 1), w.start, w.request.blocks, w.request.slot, "last",
                    w.request.rope_delta, ()) for w in work]
        self._send(("decode", payload))
        return self.graphs.run(payload)

    def run(self, work: list[Work], capture: bool = False) -> torch.Tensor:
        """FP32 logits for the rows each work item asked for ("last": its last token, "all": every token), in order.
        With capture (speculative decoding), the drafter's context features of every row are kept."""
        payload = self.payload(work)
        self._send(("step", payload, capture))
        step = build_step(payload, self.cache.block_size, self.device, self.images)
        logits, cap = self.model.forward(step, self.cache, capture=self.layers if capture else ())
        self._feats = torch.cat([cap[i] for i in self.layers], dim=-1) if capture else None
        self._run_step = step if capture else None
        return logits

    def run_payload(self, payload, capture: bool) -> None:
        step = build_step(payload, self.cache.block_size, self.device, self.images)
        _, cap = self.model.forward(step, self.cache, capture=self.layers if capture else ())
        self._feats = torch.cat([cap[i] for i in self.layers], dim=-1) if capture else None
        self._run_step = step if capture else None

    def verify_payload(self, payload) -> None:
        self._step = build_step(payload, self.cache.block_size, self.device)
        self._stash = {}
        _, cap = self.model.forward(self._step, self.cache, capture=self.layers, stash=self._stash)
        self._feats = torch.cat([cap[i] for i in self.layers], dim=-1)

    def verify(self, work: list[Work]) -> torch.Tensor:
        """Speculative verify step: logits of every row; state commits wait for accept()."""
        payload = self.payload(work)
        self._send(("verify", payload))
        self._step = build_step(payload, self.cache.block_size, self.device)
        self._stash = {}
        logits, cap = self.model.forward(self._step, self.cache, capture=self.layers, stash=self._stash)
        self._feats = torch.cat([cap[i] for i in self.layers], dim=-1)
        return logits

    def accept(self, commit_lens: list[int], rows: list[int], positions: list[int], slots: list[int]) -> None:
        """Commit the accepted rows of the last verify step and add their context to the drafter."""
        self._send(("accept", commit_lens, rows, positions, slots))
        lens = torch.tensor(commit_lens, dtype=torch.int32, device=self.device)
        self.model.commit(self._step, self.cache, self._stash, lens)
        self._stash = None
        if not self.mtp:
            self.context(rows, positions, slots, send=False)

    def context(self, rows: list[int], positions: list[int], slots: list[int], send: bool = True) -> None:
        """Write the drafter context of rows of the last step's features."""
        if send:
            self._send(("context", rows, positions, slots))
        if not rows:
            return
        dev = self.device
        r = torch.tensor(rows, dtype=torch.long, device=dev)
        self.drafter.write_context(self._feats[r], torch.tensor(positions, dtype=torch.long, device=dev),
                                   torch.tensor(slots, dtype=torch.long, device=dev))

    def draft(self, anchors: list[int], starts: list[int], slots: list[int], temperatures: list[float], generators):
        self._send(("draft", anchors, starts, slots))
        dev = self.device
        return self.drafter.draft(torch.tensor(anchors, device=dev), torch.tensor(starts, device=dev),
                                  torch.tensor(slots, device=dev), temperatures, generators)

    def _tensors(self, *lists, dtype=torch.long):
        return [torch.tensor(x, dtype=dtype, device=self.device) for x in lists]

    def _tables(self, tables: list[list[int]]) -> torch.Tensor:
        """Block tables padded with the scratch block, with columns past the longest table for the MTP drafts
        (which run up to two blocks of rows past a sequence's end; the attention reads the table at each position)."""
        width = max((len(t) for t in tables), default=1) + self.drafter.table_margin(self.cache.block_size)
        out = torch.full((len(tables), max(width, 1)), self.cache.scratch_block, dtype=torch.int32)
        for i, t in enumerate(tables):
            out[i, : len(t)] = torch.tensor(t, dtype=torch.int32)
        return out.to(self.device)

    def mtp_prefill(self, row_tokens: list[int], send: bool = True) -> None:
        """MTP keys and values of the last prefill step's rows (row_tokens: the token after each row)."""
        if send:
            self._send(("mtp_prefill", row_tokens))
        (tokens,) = self._tensors(row_tokens)
        self.drafter.run(tokens, self._feats, self._run_step, self.cache)

    def mtp_first(self, rows, tokens, positions, tables, deltas, slots, send: bool = True):
        """First drafts after a prefill: rows index the last prefill step's rows (each sequence's last prompt row)."""
        if send:
            self._send(("mtp_first", rows, tokens, positions, tables, deltas, slots))
        r, tok, pos, dl, sl = self._tensors(rows, tokens, positions, deltas, slots)
        return self.drafter.first(self._feats[r], tok, pos, self._tables(tables), dl, sl, self.cache)

    def mtp_verify(self, row_tokens, last_rows, positions, tables, deltas, slots, send: bool = True):
        """Drafts after a verify step outside the CUDA graphs: row_tokens is the token after each verify row,
        last_rows each sequence's last accepted row, positions where the next MTP pass of each sequence goes."""
        if send:
            self._send(("mtp_verify", row_tokens, last_rows, positions, tables, deltas, slots))
        rt, lr, pos, dl, sl = self._tensors(row_tokens, last_rows, positions, deltas, slots)
        return self.drafter.after_rows(self._step, self._feats, rt, lr, pos, self._tables(tables), dl, sl, self.cache)

    def close(self) -> None:
        for p in self.peers:
            try:
                p.send_bytes(pickle.dumps(("exit",)))
            except (BrokenPipeError, EOFError, OSError):
                pass
