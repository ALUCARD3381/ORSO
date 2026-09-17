import math

import orso_core
from orso.model import ModelConfig
from orso.training import Trainer


# 1) Native Model shape + parameter inventory.
config = ModelConfig(
    vocab_size=16,
    d_model=8,
    num_heads=2,
    hidden_dim=16,
    num_layers=1,
    context_length=4,
    seed=123,
)
model = config.build()
assert model.parameter_count > 0

inputs = [[1, 2, 3, 4], [2, 3, 4, 5]]
targets = [[2, 3, 4, 5], [3, 4, 5, 6]]
logits = model.forward(inputs)
assert logits.shape == [2, 4, 16]
assert all(math.isfinite(v) for v in logits.data)

# 2) Cross-entropy backward reaches model parameters.
loss = orso_core.cross_entropy(logits, targets)
assert loss.shape == [1]
assert math.isfinite(loss.item())
loss.backward()
params = model.parameters()
assert any(any(abs(g) > 0.0 for g in p.grad) for p in params)
assert all(all(math.isfinite(g) for g in p.grad) for p in params)

# 3) AdamW updates parameters.
optimizer = orso_core.AdamW(params, lr=0.01, weight_decay=0.0)
before = list(params[-1].data)
optimizer.step()
after = list(params[-1].data)
assert before != after
assert optimizer.step_count == 1
optimizer.zero_grad()
assert all(g == 0.0 for p in params for g in p.grad)

# 4) End-to-end training must lower the loss on a tiny deterministic corpus.
trainer = Trainer(model, orso_core.AdamW(params, lr=0.01, weight_decay=0.0))
initial = trainer.train_batch(inputs, targets)["loss"]
last = initial
for _ in range(24):
    last = trainer.train_batch(inputs, targets)["loss"]
assert math.isfinite(last)
assert last < initial * 0.85, (initial, last)

print("PASS: ORSO Fase 5 — Model + AdamW + Cross-Entropy + treino completo")
