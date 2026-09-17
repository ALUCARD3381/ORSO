"""Tiny end-to-end ORSO Phase 5 training example."""

import orso_core
from orso.dataset import CausalDataset
from orso.model import ModelConfig
from orso.training import Trainer

TOKENS = list(range(16)) * 8
config = ModelConfig(vocab_size=16, d_model=8, num_heads=2, hidden_dim=16, num_layers=1, context_length=4, seed=7)
model = config.build()
optimizer = orso_core.AdamW(model.parameters(), lr=0.01, weight_decay=0.0, max_grad_norm=1.0)
trainer = Trainer(model, optimizer)
dataset = CausalDataset(TOKENS, context_length=config.context_length)

for step, batch in enumerate(dataset.batches(batch_size=2, shuffle=False), 1):
    metrics = trainer.train_batch(*batch)
    if step % 5 == 0:
        print(f"step={step} loss={metrics['loss']:.6f} grad_norm={metrics['grad_norm']:.6f}")
    if step >= 20:
        break
