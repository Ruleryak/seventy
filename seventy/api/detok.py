"""Incremental detokenization: the new text of each new token, without splitting multi-byte characters."""

from __future__ import annotations

from tokenizers import Tokenizer


class IncrementalDecoder:
    """Decodes a growing token list. Text that ends in an incomplete UTF-8 sequence is held back until the next
    token completes it. A window of recent tokens keeps each step's decode short."""

    def __init__(self, tokenizer: Tokenizer, window: int = 8):
        self.tok = tokenizer
        self.ids: list[int] = []
        self.prefix_start = 0
        self.read_start = 0
        self.window = window

    def push(self, new_ids: list[int]) -> str:
        self.ids.extend(new_ids)
        prefix = self.tok.decode(self.ids[self.prefix_start : self.read_start], skip_special_tokens=True)
        full = self.tok.decode(self.ids[self.prefix_start :], skip_special_tokens=True)
        if full.endswith("�"):
            return ""
        text = full[len(prefix) :] if full.startswith(prefix) else full
        self.read_start = len(self.ids)
        self.prefix_start = max(0, self.read_start - self.window)
        return text

    def flush(self) -> str:
        prefix = self.tok.decode(self.ids[self.prefix_start : self.read_start], skip_special_tokens=True)
        full = self.tok.decode(self.ids[self.prefix_start :], skip_special_tokens=True)
        self.read_start = len(self.ids)
        return full[len(prefix) :] if full.startswith(prefix) else ""
