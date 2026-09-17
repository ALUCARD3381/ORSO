import math
import orso_core

T = orso_core.Tensor

a = T([2, 3], 0.0)
b = T([2, 3], 0.0)
for i, v in enumerate([1,2,3,4,5,6]): a.mutable_data if False else None
for idx, v in [([0,0],1),([0,1],2),([0,2],3),([1,0],4),([1,1],5),([1,2],6)]: a.set(idx, v)
for idx, v in [([0,0],6),([0,1],5),([0,2],4),([1,0],3),([1,1],2),([1,2],1)]: b.set(idx, v)

c = orso_core.add(a,b)
assert c.data == [7,7,7,7,7,7]
r = a.reshape([3,2])
t = r.transpose([1,0])
assert t.shape == [2,3]
m = orso_core.matmul(a, r)
assert m.shape == [2,2]
assert all(abs(x-y)<1e-5 for x,y in zip(m.data,[22,28,49,64]))
print("PASS: ORSO Fase 2 — Tensor Engine completo")
