"""Autoregressive inference with optional native KV-cache support."""
from __future__ import annotations

import math
import random
from typing import Iterable


def _sample_index(logits: list[float], temperature: float, top_k: int,
                  rng: random.Random, top_p: float = 1.0) -> int:
    if not logits:
        raise ValueError("cannot sample from empty logits")
    if temperature <= 0.0:
        return max(range(len(logits)), key=logits.__getitem__)
    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    scaled = [x / temperature for x in logits]
    indices = list(range(len(scaled)))
    indices.sort(key=scaled.__getitem__, reverse=True)
    if top_k > 0:
        indices = indices[:min(top_k, len(indices))]

    max_logit = max(scaled[i] for i in indices)
    weights = [math.exp(scaled[i] - max_logit) for i in indices]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        return indices[0]

    if top_p < 1.0:
        kept: list[int] = []
        accum = 0.0
        for i, weight in zip(indices, weights):
            kept.append(i)
            accum += weight / total
            if accum >= top_p:
                break
        indices = kept
        weights = [math.exp(scaled[i] - max_logit) for i in indices]
        total = sum(weights)

    threshold = rng.random() * total
    accum = 0.0
    for i, weight in zip(indices, weights):
        accum += weight
        if accum >= threshold:
            return i
    return indices[-1]


def _last_logits(logits) -> list[float]:
    seq = int(logits.shape[1])
    vocab = int(logits.shape[2])
    base = (seq - 1) * vocab
    return list(logits.data[base:base + vocab])


def _cached_forward(model, ids: list[int]):
    model.reset_kv_cache()
    return model.forward_cached(ids)


def generate_ids(
    model,
    prompt_ids: Iterable[int],
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
    eos_token_id: int | None = None,
    use_cache: bool = True,
) -> list[int]:
    """Generate token IDs; native KV-cache avoids recomputing past K/V tensors."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite")
    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

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
    if max_new_tokens == 0:
        return ids

    cached = bool(use_cache and hasattr(model, "forward_cached") and hasattr(model, "reset_kv_cache"))
    if cached:
        logits = _cached_forward(model, ids)
        for _ in range(max_new_tokens):
            next_id = _sample_index(_last_logits(logits), temperature, top_k, rng, top_p)
            ids.append(next_id)
            if eos_token_id is not None and next_id == eos_token_id:
                break
            logits = model.forward_cached([next_id])
        return ids

    for _ in range(max_new_tokens):
        context = ids[-int(model.config.context_length):]
        logits = model.forward([context])
        next_id = _sample_index(_last_logits(logits), temperature, top_k, rng, top_p)
        ids.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break
    return ids


def generate_text(model, tokenizer, prompt: str, **kwargs) -> str:
    ids = tokenizer.encode(prompt)
    generated = generate_ids(model, ids, **kwargs)
    return tokenizer.decode(generated)
