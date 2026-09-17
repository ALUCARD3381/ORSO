"""Validation suite for ORSO Phase 3: autograd + NEON kernels."""

import math
import sys

sys.path.insert(0, ".")
import orso_core

Tensor = orso_core.Tensor


def assert_close(actual, expected, tol=1e-4, msg=""):
    if len(actual) != len(expected):
        raise AssertionError(f"length mismatch: {len(actual)} != {len(expected)} {msg}")
    for i, (a, e) in enumerate(zip(actual, expected)):
        if not math.isclose(a, e, rel_tol=tol, abs_tol=tol):
            raise AssertionError(f"index {i}: {a} != {e} {msg}")


def vals(t):
    return list(t.data())


def test_phase2_regression():
    a = Tensor([2, 2], [1, 2, 3, 4])
    b = Tensor([2, 2], [5, 6, 7, 8])
    assert_close(vals(a + b), [6, 8, 10, 12])
    assert_close(vals(a * b), [5, 12, 21, 32])
    assert_close(vals(a @ b), [19, 22, 43, 50])
    assert a.transpose().shape == [2, 2]
    assert a.reshape([4]).shape == [4]


def test_grad_dot():
    x = Tensor([3], [1.0, 2.0, 3.0])
    y = Tensor([3], [4.0, 5.0, 6.0])
    x.set_requires_grad(True)
    y.set_requires_grad(True)
    z = x @ y
    assert z.requires_grad()
    z.backward()
    assert_close(vals(x.grad()), [4.0, 5.0, 6.0])
    assert_close(vals(y.grad()), [1.0, 2.0, 3.0])


def test_chain_and_gradient_accumulation():
    x = Tensor([3], [2.0, 3.0, 4.0])
    x.set_requires_grad(True)
    # z = x*x + 3*x; dz/dx = 2*x + 3
    z = (x * x) + (x * 3.0)
    z.backward(Tensor([3], [1.0, 1.0, 1.0]))
    assert_close(vals(x.grad()), [7.0, 9.0, 11.0])

    # backward again accumulates another identical contribution.
    z.backward(Tensor([3], [1.0, 1.0, 1.0]))
    assert_close(vals(x.grad()), [14.0, 18.0, 22.0])
    x.zero_grad()
    assert_close(vals(x.grad()), [0.0, 0.0, 0.0])


def test_division():
    x = Tensor([2], [6.0, 10.0])
    y = Tensor([2], [2.0, 5.0])
    x.set_requires_grad(True)
    y.set_requires_grad(True)
    z = x / y
    z.backward(Tensor([2], [1.0, 1.0]))
    assert_close(vals(x.grad()), [0.5, 0.2])
    assert_close(vals(y.grad()), [-1.5, -0.4])


def test_reshape_transpose_backward():
    x = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    x.set_requires_grad(True)
    y = x.reshape([3, 2]).transpose()
    y.backward(Tensor([2, 3], [1, 1, 1, 1, 1, 1]))
    assert_close(vals(x.grad()), [1, 1, 1, 1, 1, 1])


def test_matmul_matrix_vector_backward():
    a = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    b = Tensor([3], [2, 3, 4])
    a.set_requires_grad(True)
    b.set_requires_grad(True)
    out = a @ b
    out.backward(Tensor([2], [1, 1]))
    assert_close(vals(a.grad()), [2, 3, 4, 2, 3, 4])
    assert_close(vals(b.grad()), [5, 7, 9])


def test_batched_matmul_backward():
    a = Tensor([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8])
    b = Tensor([2, 2, 2], [2, 0, 1, 3, 4, 1, 2, 2])
    a.set_requires_grad(True)
    b.set_requires_grad(True)
    out = a @ b
    out.backward(Tensor([2, 2, 2], [1] * 8))
    # dA = G @ B^T, dB = A^T @ G for each batch.
    assert_close(vals(a.grad()), [2, 4, 2, 4, 5, 4, 5, 4])
    assert_close(vals(b.grad()), [4, 4, 6, 6, 12, 12, 14, 14])


def test_neon_flag_and_api():
    assert isinstance(orso_core.neon_available(), bool)
    x = Tensor([8], list(map(float, range(8))))
    y = Tensor([8], [1.0] * 8)
    assert_close(vals(x + y), [1, 2, 3, 4, 5, 6, 7, 8])
    assert_close(vals(x * y), list(map(float, range(8))))


if __name__ == "__main__":
    test_phase2_regression()
    test_grad_dot()
    test_chain_and_gradient_accumulation()
    test_division()
    test_reshape_transpose_backward()
    test_matmul_matrix_vector_backward()
    test_batched_matmul_backward()
    test_neon_flag_and_api()
    print("PASS: ORSO Fase 3 — Autograd + NEON completo")
