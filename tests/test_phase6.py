import math
from pathlib import Path
from tempfile import TemporaryDirectory

import orso_core
from orso.checkpoint import load_checkpoint, save_checkpoint
from orso.inference import generate_ids
from orso.model import ModelConfig
from orso.tokenizer import BPETokenizer
from orso.training import CosineScheduler, Trainer


cfg = ModelConfig(
    vocab_size=32,
    d_model=8,
    num_heads=2,
    hidden_dim=16,
    num_layers=1,
    context_length=6,
    seed=77,
).build()
params = cfg.parameters()
optimizer = orso_core.AdamW(params, lr=0.01, weight_decay=0.0)
scheduler = CosineScheduler(optimizer, total_steps=20, warmup_steps=2, min_lr=0.001)
trainer = Trainer(cfg, optimizer, scheduler)
inputs = [[1, 2, 3, 4, 5]]
targets = [[2, 3, 4, 5, 6]]

first = trainer.train_batch(inputs, targets)
assert math.isfinite(first["loss"])
assert optimizer.step_count == 1

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    tok = BPETokenizer(vocab_size=300, min_frequency=1).train([
        "hello ORSO checkpoint",
        "hello ORSO inference",
    ])
    checkpoint = root / "model.orso"
    save_checkpoint(
        checkpoint,
        cfg,
        optimizer,
        tokenizer=tok,
        trainer=trainer,
        scheduler=scheduler,
        metadata={"purpose": "phase6-test"},
    )
    assert checkpoint.exists()

    loaded = load_checkpoint(checkpoint)
    assert loaded.trainer_steps == trainer.steps
    assert loaded.optimizer.step_count == optimizer.step_count
    assert loaded.tokenizer is not None
    assert loaded.tokenizer.decode(loaded.tokenizer.encode("hello ORSO")) == "hello ORSO"
    assert loaded.metadata["purpose"] == "phase6-test"
    assert loaded.scheduler is not None

    before = cfg.parameter_data()
    after = loaded.model.parameter_data()
    assert before == after
    assert optimizer.first_moment() == loaded.optimizer.first_moment()
    assert optimizer.second_moment() == loaded.optimizer.second_moment()
    assert optimizer.lr == loaded.optimizer.lr

    logits_a = cfg.forward(inputs)
    logits_b = loaded.model.forward(inputs)
    assert logits_a.shape == logits_b.shape
    assert logits_a.data == logits_b.data

    ids_a = generate_ids(cfg, [1, 2, 3], max_new_tokens=3, temperature=0.0)
    ids_b = generate_ids(loaded.model, [1, 2, 3], max_new_tokens=3, temperature=0.0)
    assert ids_a == ids_b

    resumed = loaded.build_trainer()
    resumed_result = resumed.train_batch(inputs, targets)
    assert math.isfinite(resumed_result["loss"])
    assert resumed.steps == trainer.steps + 1

print("PASS: ORSO Fase 6 — Checkpoint + resume + tokenizer + inference completo")
