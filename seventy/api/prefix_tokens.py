"""Tokenization that reuses the tokens of recent prompts.

An agent sends the whole conversation with every turn, so a new chat prompt usually starts with the text of an
earlier one; tokenizing a prompt of 250,000 tokens takes most of a second, and concurrent requests do not overlap.
The tokenizer splits its input at added tokens before anything else, so the text between two special tokens is
tokenized on its own: when the new text shares a cached text up to and including a turn marker (<|im_start|>), the
tokens before that marker are exactly the cached tokens before it. Only the rest is tokenized again.
"""

from __future__ import annotations

import bisect
import threading
from array import array
from dataclasses import dataclass

CHUNK = 65536


def common_prefix(a: str, b: str) -> int:
    """Length of the common prefix of a and b."""
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k : k + CHUNK] == b[k : k + CHUNK]:
        k += CHUNK
    hi = min(n, k + CHUNK)
    lo = k
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[k:mid] == b[k:mid]:
            lo = mid
        else:
            hi = mid - 1
    return min(lo, n)


@dataclass
class Entry:
    text: str
    ids: array
    offsets: list[int]
    positions: list[int]


class PrefixTokenizer:
    """tokenizer.encode(text, add_special_tokens=False).ids, with the tokens of the last `entries` prompts of at
    least `min_chars` characters kept for reuse. Safe to call from several threads."""

    def __init__(self, tokenizer, marker: str = "<|im_start|>", entries: int = 16, min_chars: int = 20000):
        self.tok = tokenizer
        self.marker = marker
        self.marker_id = tokenizer.token_to_id(marker)
        self.max_entries = entries
        self.min_chars = min_chars
        self.entries: list[Entry] = []
        self.lock = threading.Lock()
        self.reused = 0

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode(self, text: str) -> list[int]:
        if self.marker_id is None or len(text) < self.min_chars:
            return self._encode(text)
        with self.lock:
            entries = list(self.entries)
        base, cut, keep = None, 0, 0
        for e in entries:
            if e.text[:256] != text[:256]:
                continue
            i = bisect.bisect_right(e.offsets, common_prefix(text, e.text) - len(self.marker)) - 1
            if i >= 0 and e.offsets[i] > cut:
                base, cut, keep = e, e.offsets[i], i
        if base is None:
            ids = self._encode(text)
            offsets, positions = self._markers(text, ids, 0, 0)
        else:
            tail = self._encode(text[cut:])
            ids = base.ids[: base.positions[keep]].tolist() + tail
            offsets, positions = self._markers(text, tail, cut, base.positions[keep])
            offsets = base.offsets[:keep] + offsets
            positions = base.positions[:keep] + positions
            self.reused += 1
        if len(offsets) == len(positions):
            entry = Entry(text, array("i", ids), offsets, positions)
            with self.lock:
                self.entries = [e for e in self.entries if e is not base][-(self.max_entries - 1):] + [entry]
        return ids

    def _markers(self, text: str, ids: list[int], start: int, first: int) -> tuple[list[int], list[int]]:
        """Character offsets of the marker in text[start:] and the token positions of the marker in ids (shifted
        by first)."""
        offsets, k = [], text.find(self.marker, start)
        while k >= 0:
            offsets.append(k)
            k = text.find(self.marker, k + len(self.marker))
        positions = [first + i for i, t in enumerate(ids) if t == self.marker_id]
        return offsets, positions
