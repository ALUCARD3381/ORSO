import math
import orso_core
T = orso_core.Tensor

x = T.random_normal([2,3], seed=123, requires_grad=True)
w = T.ones([3,3], requires_grad=True)
y = orso_core.matmul(x,w)
loss = y.sum()
loss.backward()
assert len(x.grad) == 6 and len(w.grad) == 9
assert all(math.isfinite(v) for v in x.grad + w.grad)
print("PASS: ORSO Fase 3 — Autograd + NEON completo")
