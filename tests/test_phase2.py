import orso_core

Tensor = orso_core.Tensor


def assert_close(got, expected, eps=1e-5):
    assert len(got) == len(expected), (got, expected)
    for a, b in zip(got, expected):
        assert abs(a - b) <= eps, (got, expected)


def test_fundamentals():
    t = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    assert t.shape == [2, 3]
    assert t.ndim == 2
    assert t.size() == 6
    assert t.get([1, 2]) == 6
    t.set([0, 1], 20)
    assert t.get([0, 1]) == 20


def test_zeros_ones_full():
    z = Tensor.zeros([2, 2])
    o = Tensor.ones([2, 2])
    f = Tensor.full([2, 2], 3.5)
    assert_close(z.data(), [0, 0, 0, 0])
    assert_close(o.data(), [1, 1, 1, 1])
    assert_close(f.data(), [3.5, 3.5, 3.5, 3.5])


def test_elementwise_and_scalar():
    a = Tensor([2, 2], [1, 2, 3, 4])
    b = Tensor([2, 2], [5, 6, 7, 8])
    assert_close((a + b).data(), [6, 8, 10, 12])
    assert_close((b - a).data(), [4, 4, 4, 4])
    assert_close((a * b).data(), [5, 12, 21, 32])
    assert_close((b / a).data(), [5, 3, 7 / 3, 2])
    assert_close((a * 2.0).data(), [2, 4, 6, 8])
    assert_close((a / 2.0).data(), [0.5, 1, 1.5, 2])
    assert_close((-a).data(), [-1, -2, -3, -4])


def test_reshape():
    a = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    b = a.reshape([3, 2])
    c = b.reshape([6])
    assert b.shape == [3, 2]
    assert c.shape == [6]
    assert_close(c.data(), [1, 2, 3, 4, 5, 6])


def test_transpose():
    a = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    t = a.transpose()
    assert t.shape == [3, 2]
    assert_close(t.data(), [1, 4, 2, 5, 3, 6])

    x = Tensor([2, 3, 4], list(range(24)))
    p = x.transpose([1, 2, 0])
    assert p.shape == [3, 4, 2]
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert p.get([j, k, i]) == x.get([i, j, k])


def test_matmul():
    a = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    b = Tensor([3, 2], [7, 8, 9, 10, 11, 12])
    c = a @ b
    assert c.shape == [2, 2]
    assert_close(c.data(), [58, 64, 139, 154])

    v = Tensor([3], [1, 2, 3])
    mv = a @ v
    assert mv.shape == [2]
    assert_close(mv.data(), [14, 32])

    vm = v @ b
    assert vm.shape == [2]
    assert_close(vm.data(), [58, 64])

    dot = v @ Tensor([3], [4, 5, 6])
    assert dot.shape == []
    assert abs(dot.item() - 32) < 1e-5


def test_batched_matmul():
    # Two independent 2x3 @ 3x2 multiplications.
    a = Tensor([2, 2, 3], [
        1, 2, 3, 4, 5, 6,
        2, 0, 1, 1, 3, 4,
    ])
    b = Tensor([2, 3, 2], [
        7, 8, 9, 10, 11, 12,
        1, 2, 3, 4, 5, 6,
    ])
    c = a @ b
    assert c.shape == [2, 2, 2]
    assert_close(c.data(), [58, 64, 139, 154, 7, 10, 30, 38])


def test_errors():
    a = Tensor([2, 2], [1, 2, 3, 4])
    b = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    try:
        _ = a + b
        raise AssertionError("shape mismatch should fail")
    except (ValueError, RuntimeError):
        pass

    try:
        _ = a.reshape([3, 3])
        raise AssertionError("invalid reshape should fail")
    except (ValueError, RuntimeError):
        pass


if __name__ == "__main__":
    test_fundamentals()
    test_zeros_ones_full()
    test_elementwise_and_scalar()
    test_reshape()
    test_transpose()
    test_matmul()
    test_batched_matmul()
    test_errors()
    print("PASS: ORSO Fase 2 — Tensor Engine completo")
