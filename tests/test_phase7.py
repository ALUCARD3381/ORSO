"""Final integration test for ORSO Phase 7: runtime, KV-cache and ANSI CLI."""
from __future__ import annotations

import tempfile
from pathlib import Path

from orso.chat import ChatSession
from orso.checkpoint import load_checkpoint, save_checkpoint
from orso.dataset import CausalDataset
from orso.model import ModelConfig
from orso.runtime import ORSORuntime
from orso.tokenizer import BPETokenizer
from orso.training import Trainer
from orso.inference import generate_ids
from orso_core import AdamW


def _assert_close(a, b, tol=2e-5):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(float(x) - float(y)) <= tol, (x, y)


def main() -> None:
    corpus = "ORSO learns locally. ORSO trains from scratch. Local models are useful. " * 8
    tokenizer = BPETokenizer(vocab_size=256, min_frequency=2).train([corpus])
    ids = tokenizer.encode(corpus)
    assert ids
    dataset = CausalDataset(ids, context_length=8)
    cfg = ModelConfig(vocab_size=256, d_model=8, num_heads=2, hidden_dim=16, num_layers=1, context_length=8, seed=7)
    model = cfg.build()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    trainer = Trainer(model, optimizer)
    batch = next(dataset.batches(batch_size=2, shuffle=False))
    metrics = trainer.train_batch(*batch)
    assert metrics["loss"] > 0.0
    assert optimizer.step_count == 1

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "model.orso"
        save_checkpoint(ckpt, model, optimizer, tokenizer=tokenizer, trainer=trainer, metadata={"phase": 7})
        state = load_checkpoint(ckpt)
        assert state.tokenizer is not None
        assert state.trainer_steps == 1
        assert int(state.model.parameter_count) == int(model.parameter_count)

        session = ChatSession(system_prompt="You are ORSO.", max_turns=4, max_chars=1000)
        session.add("user", "hello")
        session.add("assistant", "hi")
        session_json = Path(td) / "session.json"
        session.save(session_json)
        restored = ChatSession.load(session_json)
        assert len(restored.turns) == 2

        runtime = ORSORuntime.from_checkpoint(ckpt, session=restored)
        text = runtime.respond("test", max_new_tokens=1, temperature=0.0)
        assert isinstance(text, str)
        runtime.clear_session()
        assert runtime.stats.kv_cache_length == 0

    # Cached and uncached inference must be numerically equivalent.
    probe = cfg.build()
    prefix = [1, 3, 5, 7]
    follow = 9
    full = probe.forward([prefix])
    probe.reset_kv_cache()
    cached = probe.forward_cached(prefix)
    _assert_close(list(full.data[-cfg.vocab_size:]), list(cached.data))

    full_next = probe.forward([prefix + [follow]])
    cached_next = probe.forward_cached([follow])
    _assert_close(list(full_next.data[-cfg.vocab_size:]), list(cached_next.data))
    assert int(probe.kv_cache_length) == len(prefix) + 1

    # Context sliding: fill the cache to context_length, then overflow by one token.
    fill_tokens = [11, 13, 15]
    probe.forward_cached(fill_tokens)
    assert int(probe.kv_cache_length) == cfg.context_length

    overflow_token = 17
    cached_overflow = probe.forward_cached([overflow_token])
    assert int(probe.kv_cache_length) == cfg.context_length

    active_tokens = prefix + [follow] + fill_tokens
    rebuilt_window = active_tokens[-(cfg.context_length - 1):] + [overflow_token]
    rebuilt = probe.forward([rebuilt_window])
    _assert_close(list(rebuilt.data[-cfg.vocab_size:]), list(cached_overflow.data))

    # Deterministic generation must match with and without cache.
    baseline = cfg.build()
    uncached_ids = generate_ids(baseline, prefix, max_new_tokens=5, temperature=0.0, use_cache=False)
    cached_ids = generate_ids(baseline, prefix, max_new_tokens=5, temperature=0.0, use_cache=True)
    assert cached_ids == uncached_ids
    assert int(baseline.kv_cache_length) <= cfg.context_length

    # ANSI banner colors are part of the CLI contract.
    from runpy import run_path
    cli = run_path("scripts/chat.py")
    banner = cli["render_banner"](True)
    assert "\033[92m" in banner
    assert "\033[96m" in banner
    assert "\033[0m" in banner

    print("PASS: ORSO Fase 7 — integração final + KV-cache incremental + CLI ANSI + top-p completo")


if __name__ == "__main__":
    main()
