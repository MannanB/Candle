import numpy as np
import candle


rng = np.random.default_rng(0)

def T(x):
    return candle.Tensor(x.tolist())

def check(name, actual, expected, **info):
    if np.allclose(actual, expected, rtol=1e-3, atol=1e-3):
        print(f"{name} ✓")
        return

    print(f"\n{name} FAILED")
    for k, v in info.items():
        print(f"{k}: {v}")

    print("actual shape:  ", actual.shape)
    print("expected shape:", expected.shape)

    if actual.shape == expected.shape:
        diff = np.abs(actual - expected)
        i = np.unravel_index(np.argmax(diff), diff.shape)

        print("max error:", diff[i])
        print("at index:", i)
        print("actual:", actual[i])
        print("expected:", expected[i])

    if expected.size <= 100:
        print("\nexpected:\n", expected)
        print("\nactual:\n", actual)
    else:
        print("\nexpected first 20:", expected.flatten()[:20])
        print("actual first 20:  ", actual.flatten()[:20])

    raise AssertionError(name)


for trial in range(12):
    print(f"\n--- trial {trial + 1} ---")

    # ADD
    shape = tuple(rng.integers(2, 20, size=rng.integers(1, 6)))

    a = rng.uniform(-10, 10, shape).astype(np.float32)
    b = rng.uniform(-10, 10, shape).astype(np.float32)

    ca, cb = T(a), T(b)

    check(
        "add",
        (ca + cb).numpy(),
        a + b,
        shape=shape
    )

    # TRANSPOSE
    shape = tuple(rng.integers(2, 15, size=rng.integers(2, 7)))

    a = rng.uniform(-10, 10, shape).astype(np.float32)

    d1 = int(rng.integers(0, len(shape) - 1))
    d2 = d1 + 1

    ca = T(a)

    check(
        "transpose",
        ca.transpose(d1, d2).numpy(),
        np.swapaxes(a, d1, d2),
        input_shape=shape,
        dims=(d1, d2)
    )

    # MATMUL (2D or same-shape batches; no broadcasting)
    batch_rank = trial % 3
    batch_shape = tuple(rng.integers(2, 5, size=batch_rank))
    m = int(rng.integers(2, 40))
    k = int(rng.integers(2, 40))
    n = int(rng.integers(2, 40))

    a = rng.uniform(-5, 5, (*batch_shape, m, k)).astype(np.float32)
    b = rng.uniform(-5, 5, (*batch_shape, k, n)).astype(np.float32)

    ca, cb = T(a), T(b)

    check(
        "matmul",
        ca.matmul(cb).numpy(),
        np.matmul(a, b),
        A_shape=a.shape,
        B_shape=b.shape,
        batch_shape=batch_shape,
        M=m,
        K=k,
        N=n
    )

print("\nALL TESTS PASSED")
