"""Autoregressive inference utilities for ORSO Phase 6."""

from __future__ import annotations

import math
import random
from typing import Iterable


def _sample_index(logits: list[float], temperature: float, top_k: int, rng: random.Random) -> int:
    if not logits:
        raise ValueError("cannot sample from empty logits")
    if temperature <= 0.0:
        return max(range(len(logits)), key=logits.__getitem__)

    scaled = [x / temperature for x in logits]
    indices = list(range(len(scaled)))
    if top_k > 0 and top_k < len(indices):
        indices.sort(key=scaled.__getitem__, reverse=True)
        indices = indices[:top_k]
    max_logit = max(scaled[i] for i in indices)
    weights = [math.exp(scaled[i] - max_logit) for i in indices]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        return max(indices, key=scaled.__getitem__)
    threshold = rng.random() * total
    accum = 0.0
    for i, weight in zip(indices, weights):
        accum += weight
        if accum >= threshold:
            return i
    return indices[-1]


def generate_ids(
    model,
    prompt_ids: Iterable[int],
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 0,
    eos_token_id: int | None = None,
) -> list[int]:
    """Generate token IDs autoregressively using the model's causal logits."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite")
    ids = [int(x) for x in prompt_ids]
    if not ids:
        raise ValueError("prompt_ids cannot be empty")
    rng = random.Random(seed)
    vocab_size = int(model.config.vocab_size)
    for token in ids:
        if token < 0 or token >= vocab_size:
            raise ValueError(f"prompt token out of range: {token}")
    if eos_token_id is not None and (eos_token_id < 0 or eos_token_id >= vocab_size):
        raise ValueError("eos_token_id out of range")

    for _ in range(max_new_tokens):
        context = ids[-model.context_length :]
        logits = model.forward([context])
        seq = int(logits.shape[1])
        vocab = int(logits.shape[2])
        base = (seq - 1) * vocab
        next_id = _sample_index(list(logits.data[base : base + vocab]), temperature, top_k, rng)
        ids.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break
    return ids


def generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 0,
    eos_token_id: int | None = None,
) -> str:
    ids = tokenizer.encode(prompt)
    generated = generate_ids(
        model,
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        eos_token_id=eos_token_id,
    )
    return tokenizer.decode(generated)
