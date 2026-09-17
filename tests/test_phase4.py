import math
import orso_core

T = orso_core.Tensor

# Embeddings
emb_w = T.random_normal([32, 8], stddev=0.1, seed=7, requires_grad=True)
emb = orso_core.embedding(emb_w, [[1,2,3],[3,2,1]])
assert emb.shape == [2,3,8]
emb.sum().backward()
assert all(math.isfinite(v) for v in emb_w.grad)

# RoPE preserves the L2 norm of each rotated pair.
x = T.random_normal([1,4,8], seed=11, requires_grad=True)
y = orso_core.rope(x)
for r in range(4):
    for j in range(0,8,2):
        a = math.hypot(x.get([0,r,j]), x.get([0,r,j+1]))
        b = math.hypot(y.get([0,r,j]), y.get([0,r,j+1]))
        assert abs(a-b) < 1e-4

y.sum().backward()
assert all(math.isfinite(v) for v in x.grad)

# RMSNorm
rn_x = T.random_normal([2,4,8], seed=22, requires_grad=True)
rn_w = T.ones([8], requires_grad=True)
rn = orso_core.rmsnorm(rn_x, rn_w)
assert rn.shape == [2,4,8]
for b in range(2):
    for t in range(4):
        sq = sum(rn.get([b,t,j])**2 for j in range(8)) / 8.0
        assert abs(sq - 1.0) < 1e-4
rn.sum().backward()
assert all(math.isfinite(v) for v in rn_x.grad + rn_w.grad)

# SwiGLU
sx = T.random_normal([2,4,8], seed=30, requires_grad=True)
wg = T.random_normal([8,16], stddev=0.05, seed=31, requires_grad=True)
wu = T.random_normal([8,16], stddev=0.05, seed=32, requires_grad=True)
wd = T.random_normal([16,8], stddev=0.05, seed=33, requires_grad=True)
so = orso_core.swiglu(sx,wg,wu,wd)
assert so.shape == [2,4,8]
so.sum().backward()
for g in (sx.grad,wg.grad,wu.grad,wd.grad): assert all(math.isfinite(v) for v in g)

# Multi-Head Attention + RoPE + causal mask.
ax = T.random_normal([2,4,8], stddev=0.1, seed=40, requires_grad=True)
wq = T.random_normal([8,8], stddev=0.05, seed=41, requires_grad=True)
wk = T.random_normal([8,8], stddev=0.05, seed=42, requires_grad=True)
wv = T.random_normal([8,8], stddev=0.05, seed=43, requires_grad=True)
wo = T.random_normal([8,8], stddev=0.05, seed=44, requires_grad=True)
aout = orso_core.multi_head_attention(ax,wq,wk,wv,wo,2,True)
assert aout.shape == [2,4,8]
assert all(math.isfinite(v) for v in aout.data)
aout.sum().backward()
for g in (ax.grad,wq.grad,wk.grad,wv.grad,wo.grad): assert all(math.isfinite(v) for v in g)

assert isinstance(orso_core.neon_enabled(), bool)
print("PASS: ORSO Fase 4 — Transformer Core completo")
