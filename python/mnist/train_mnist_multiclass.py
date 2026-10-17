r"""Train a 10-class MNIST classifier and compare all five optimizers.

    .venv\Scripts\python.exe python/mnist/train_mnist_multiclass.py

Defaults to 10 epochs per optimizer. Small trial run:
    python python/mnist/train_mnist_multiclass.py --epochs 1 --train-limit 256 --test-limit 128
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
import candle
from mnist.compare_optimizers import OPTIMIZERS, load_data, make_optimizer, save_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    for name in OPTIMIZERS:
        parser.add_argument(f"--{name.lower()}-lr", type=float, help=f"Override {name}'s initial LR")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "multiclass_optimizer_curves")
    args = parser.parse_args()
    for field in ("epochs", "batch_size", "train_limit", "test_limit"):
        value = getattr(args, field)
        if value is not None and value <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    rates = [args.lr, args.lr_decay] + [getattr(args, f"{name.lower()}_lr") for name in OPTIMIZERS]
    if any(rate is not None and (not np.isfinite(rate) or rate <= 0) for rate in rates):
        parser.error("Learning rates and decay must be finite and positive")
    if len(set(args.optimizers)) != len(args.optimizers):
        parser.error("List each optimizer only once")
    return args


def prepare_data(images, labels, rng, limit):
    indices = rng.permutation(len(labels))
    if limit is not None:
        indices = indices[:limit]
    x = np.asarray([images[int(i)] for i in indices], dtype=np.float32)
    x = x.reshape(-1, 784, 1) / 255.0
    y = np.asarray(labels, dtype=np.int64)[indices]
    return x, y


class MulticlassClassifier:
    def __init__(self, initial_parameters):
        self.l1 = candle.Linear(784, 64)
        self.l2 = candle.Linear(64, 16)
        self.l3 = candle.Linear(16, 10)
        self.layers = [self.l1, self.l2, self.l3]
        for parameter, initial in zip(self.parameters(), initial_parameters):
            parameter.inplace_add(candle.Tensor(initial.tolist()), self_factor=0.0)

    def parameters(self):
        return [parameter for layer in self.layers for parameter in (layer.weights, layer.bias)]

    def forward(self, x):
        x = candle.activations.sigmoid(self.l1.forward(x))
        x = candle.activations.sigmoid(self.l2.forward(x))
        return candle.activations.softmax(self.l3.forward(x), dim=1)

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        for i, layer in enumerate(self.layers, start=1):
            layer.save(str(path / f"l{i}.bin"))


def run_batches(model, x, y, batch_size, order, optimizer=None, description="Testing"):
    total_loss, correct, total = 0.0, 0, 0
    training = optimizer is not None
    one_hot = np.eye(10, dtype=np.float32)
    for parameter in model.parameters():
        parameter.requires_grad = training
    try:
        for start in tqdm(range(0, len(order), batch_size), desc=description, leave=False):
            indices = order[start:start + batch_size]
            count = len(indices)
            # Work around Candle's bias broadcasting for singleton batches.
            # The repeated sample has the same mean loss and mean gradient.
            batch_indices = np.repeat(indices, 2) if count == 1 else indices
            inputs = candle.Tensor(x[batch_indices].tolist())
            targets = candle.Tensor(one_hot[y[batch_indices], :, None].tolist())
            if training:
                optimizer.zero_grad()
            output = model.forward(inputs)
            loss = candle.losses.cce(output, targets, dim=1)
            probabilities = output.numpy()
            batch_loss = float(loss.tolist())
            if not np.isfinite(batch_loss) or not np.isfinite(probabilities).all():
                raise RuntimeError(f"Non-finite values during {description}; try a smaller learning rate")
            total_loss += batch_loss * count
            predictions = probabilities[:count].argmax(axis=1).reshape(-1)
            correct += int(np.count_nonzero(predictions == y[indices]))
            total += count
            if training:
                loss.backward()
                optimizer.step()
    finally:
        for parameter in model.parameters():
            parameter.requires_grad = True
    return total_loss / total, correct / total


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    (images, labels), (test_images, test_labels) = load_data(args.data_dir)
    x_train, y_train = prepare_data(images, labels, rng, args.train_limit)
    x_test, y_test = prepare_data(test_images, test_labels, rng, args.test_limit)
    del images, labels, test_images, test_labels
    # Same initialization and data for every optimizer, matching Linear's range.
    initial = [rng.uniform(-1, 1, shape).astype(np.float32) for shape in (
        (64, 784), (64, 1), (16, 64), (16, 1), (10, 16), (10, 1),
    )]
    config = vars(args) | {"train_examples": len(x_train), "test_examples": len(x_test),
                           "architecture": [784, 64, 16, 10], "loss": "CCE"}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"Train: {len(x_train)}, test: {len(x_test)}; {args.epochs} epochs per optimizer")
    rows = []
    for name in args.optimizers:
        model = MulticlassClassifier(initial)
        lr = getattr(args, f"{name.lower()}_lr")
        optimizer = make_optimizer(name, args.lr if lr is None else lr)
        for layer in model.layers:
            optimizer.add_layer(layer)
        optimizer.init_grad()
        shuffle = np.random.default_rng(args.seed)
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_batches(
                model, x_train, y_train, args.batch_size, shuffle.permutation(len(x_train)),
                optimizer, f"{name} {epoch}/{args.epochs}",
            )
            test_loss, test_accuracy = run_batches(
                model, x_test, y_test, args.batch_size, np.arange(len(x_test)),
            )
            rows.append(dict(optimizer=name, epoch=epoch, lr=optimizer.lr,
                             train_loss=train_loss, test_loss=test_loss,
                             train_accuracy=train_accuracy, test_accuracy=test_accuracy))
            print(f"{name:12s} epoch {epoch:2d}: train CCE={train_loss:.5f}, "
                  f"accuracy={train_accuracy:.2%}; test CCE={test_loss:.5f}, accuracy={test_accuracy:.2%}")
            save_results(rows, args.output_dir, loss_name="CCE")
            optimizer.lr *= args.lr_decay
        model.save(args.output_dir / name)
        del optimizer, model
    print(f"Saved curves, metrics.csv, config.json and models to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
