"""Simple causal language-model dataset for ORSO Phase 5."""

from __future__ import annotations

import random
from typing import Iterable, Sequence


class CausalDataset:
    """Slices one token stream into fixed-length next-token prediction samples."""

    def __init__(self, token_ids: Sequence[int], context_length: int):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if len(token_ids) <= context_length:
            raise ValueError("token stream must contain more tokens than context_length")
        self.token_ids = list(map(int, token_ids))
        self.context_length = int(context_length)
        self._size = len(self.token_ids) - self.context_length

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index
        end = start + self.context_length
        return self.token_ids[start:end], self.token_ids[start + 1 : end + 1]

    def batches(self, batch_size: int, shuffle: bool = True, seed: int = 1234):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        indices = list(range(len(self)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            inputs = []
            targets = []
            for idx in chosen:
                x, y = self[idx]
                inputs.append(x)
                targets.append(y)
            yield inputs, targets
