"""Integration test for ORSO Phase 7."""
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
from orso_core import AdamW


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

    print("PASS: ORSO Fase 7 — integração final + CLI runtime + memória + top-p completo")


if __name__ == "__main__":
    main()
