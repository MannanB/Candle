"""Numerical checks for BCE, CCE, and softmax backward."""

import numpy as np
import candle


def tensor(values, grad=False):
    return candle.Tensor(np.asarray(values, dtype=np.float32).tolist(), requires_grad=grad)


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


def softmax(values, dim):
    values = values - values.max(axis=dim, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=dim, keepdims=True)


def main():
    rng = np.random.default_rng(42)
    scale = 2.5
    for shape in ((1,), (17, 19), (3, 4, 2)):
        p = rng.uniform(0.1, 0.9, shape).astype(np.float32)
        y = rng.uniform(0, 1, shape).astype(np.float32)
        prediction, target = tensor(p, True), tensor(y, True)
        loss = candle.losses.bce(prediction, target)
        check(loss.tolist(), -(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())
        loss.add(candle.Tensor.zeroes([]), self_factor=scale).backward()
        check(prediction.grad.numpy(), scale * (p - y) / (p * (1 - p) * p.size))
        check(target.grad.numpy(), scale * (np.log1p(-p) - np.log(p)) / p.size)

    # Sigmoid -> BCE should simplify to (sigmoid(logits) - target) / N.
    values = rng.normal(size=(5, 3)).astype(np.float32)
    y = rng.integers(0, 2, size=values.shape).astype(np.float32)
    logits = tensor(values, True)
    probability = candle.activations.sigmoid(logits)
    candle.losses.bce(probability, tensor(y)).backward()
    check(logits.grad.numpy(), (1 / (1 + np.exp(-values)) - y) / values.size)
    print("PASS BCE: forward, prediction/target gradients, scaled upstream, sigmoid chain")

    for shape, dim in (((3, 7), 1), ((3, 5, 2), 0), ((3, 5, 2), 1), ((3, 5, 2), 2), ((7,), 0), ((257, 3), 1)):
        values = rng.normal(size=shape).astype(np.float32)
        p = softmax(values.astype(np.float64), dim)
        y = rng.uniform(0.1, 1, shape)
        y = (y / y.sum(axis=dim, keepdims=True)).astype(np.float32)
        count = values.size // shape[dim]

        prediction, target = tensor(p, True), tensor(y, True)
        loss = candle.losses.cce(prediction, target, dim)
        check(loss.tolist(), -(y * np.log(p)).sum(axis=dim).mean())
        loss.add(candle.Tensor.zeroes([]), self_factor=scale).backward()
        check(prediction.grad.numpy(), -scale * y / (p * count))
        check(target.grad.numpy(), -scale * np.log(p) / count)

        # Check softmax with arbitrary upstream gradients, independently of CE.
        logits = tensor(values, True)
        probability = candle.activations.softmax(logits, dim)
        upstream_target = rng.normal(size=shape).astype(np.float32)
        candle.losses.mse(probability, tensor(upstream_target)).backward()
        upstream = 2 * (p - upstream_target) / values.size
        expected = p * (upstream - (upstream * p).sum(axis=dim, keepdims=True))
        check(logits.grad.numpy(), expected)

        # Check the combined softmax -> CCE gradient for soft and one-hot targets.
        classes = rng.integers(shape[dim], size=np.delete(shape, dim))
        one_hot = np.moveaxis(np.eye(shape[dim], dtype=np.float32)[classes], -1, dim)
        for labels in (y, one_hot):
            logits = tensor(values, True)
            probability = candle.activations.softmax(logits, dim)
            candle.losses.cce(probability, tensor(labels), dim).backward()
            check(logits.grad.numpy(), (p - labels) / count)
    print("PASS CCE and softmax: all axes, soft/one-hot targets, mean reduction, chain rule")

    # Central differences check the composed loss without using its derivative.
    values = rng.normal(size=(2, 3)).astype(np.float32)
    labels = np.eye(3)[[0, 2]]
    logits = tensor(values, True)
    candle.losses.cce(candle.activations.softmax(logits, 1), tensor(labels)).backward()
    numerical = np.empty_like(values)
    for index in np.ndindex(values.shape):
        plus, minus = values.astype(np.float64), values.astype(np.float64)
        plus[index] += 1e-4
        minus[index] -= 1e-4
        numerical[index] = ((-labels * np.log(softmax(plus, 1))).sum() -
                            (-labels * np.log(softmax(minus, 1))).sum()) / (2e-4 * len(values))
    check(logits.grad.numpy(), numerical)

    # Probabilities of exactly zero/one must not produce infinities or NaNs.
    for fn in (candle.losses.bce, candle.losses.cce):
        prediction = tensor([[0, 1], [1, 0]], True)
        loss = fn(prediction, tensor([[1, 0], [0, 1]]))
        loss.backward()
        assert np.isfinite(loss.tolist())
        assert np.isfinite(prediction.grad.numpy()).all()

    for fn in (candle.losses.bce, candle.losses.cce):
        result = fn(tensor([[0.2, 0.8]]), tensor([[0, 1]]))
        assert not result.requires_grad and result.is_leaf
        try:
            fn(tensor([[0.2, 0.8]]), tensor([1]))
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched shapes should fail")
    print("PASS finite differences, boundary values, inference, shape validation")


if __name__ == "__main__":
    main()
