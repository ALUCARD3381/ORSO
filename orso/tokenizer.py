"""Dependency-free byte-level BPE tokenizer for ORSO."""

from __future__ import annotations

import base64
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class BPETokenizer:
    """Small byte-level BPE implementation with deterministic merges."""

    def __init__(self, vocab_size: int = 768, min_frequency: int = 2):
        if vocab_size < 256:
            raise ValueError("byte-level BPE requires vocab_size >= 256")
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        self.vocab_size = int(vocab_size)
        self.min_frequency = int(min_frequency)
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        self.merges: list[tuple[int, int]] = []

    @property
    def size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, texts: Iterable[str]) -> "BPETokenizer":
        sequences = [list(text.encode("utf-8")) for text in texts if text]
        if not sequences:
            raise ValueError("BPE training corpus is empty")

        target_merges = self.vocab_size - 256
        for _ in range(target_merges):
            counts: Counter[tuple[int, int]] = Counter()
            for ids in sequences:
                counts.update(zip(ids, ids[1:]))
            candidates = [(freq, pair) for pair, freq in counts.items() if freq >= self.min_frequency]
            if not candidates:
                break
            _, pair = max(candidates, key=lambda item: (item[0], -item[1][0], -item[1][1]))
            new_id = len(self.vocab)
            self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
            self.merges.append(pair)
            sequences = [self._apply_merge(ids, pair, new_id) for ids in sequences]
        return self

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for offset, pair in enumerate(self.merges):
            ids = self._apply_merge(ids, pair, 256 + offset)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        raw = bytearray()
        for idx in ids:
            if idx < 0 or idx >= len(self.vocab):
                raise ValueError(f"token id out of range: {idx}")
            raw.extend(self.vocab[idx])
        return bytes(raw).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "vocab": [base64.b64encode(piece).decode("ascii") for piece in self.vocab],
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_dict(cls, state: dict) -> "BPETokenizer":
        tok = cls(int(state["vocab_size"]), int(state.get("min_frequency", 2)))
        tok.vocab = [base64.b64decode(piece) for piece in state["vocab"]]
        tok.merges = [tuple(map(int, pair)) for pair in state["merges"]]
        return tok

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(state)
