r"""Compare optimizers on mnist_test.py's binary classifier.

From the repository root:
    .venv\Scripts\python.exe python/mnist/compare_optimizers.py

Small trial run:
    python python/mnist/compare_optimizers.py --epochs 1 --train-limit 256 --test-limit 128
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
import candle
from mnist.mnist_data import MnistDataloader

OPTIMIZERS = ("SGD", "SGDMomentum", "AdaGrad", "RMSProp", "Adam")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target", type=int, choices=range(10), default=2)
    parser.add_argument("--positive-fraction", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.05, help="Default initial LR for every optimizer")
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    for name in OPTIMIZERS:
        parser.add_argument(f"--{name.lower()}-lr", type=float, help=f"Override {name}'s initial LR")
    parser.add_argument("--train-limit", type=int, help="Use at most this many balanced training examples")
    parser.add_argument("--test-limit", type=int, help="Use at most this many balanced test examples")
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "optimizer_curves")
    args = parser.parse_args()
    for field in ("epochs", "batch_size", "train_limit", "test_limit"):
        value = getattr(args, field)
        if value is not None and value <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if not 0 < args.positive_fraction < 1:
        parser.error("--positive-fraction must be between 0 and 1")
    rates = [args.lr, args.lr_decay] + [getattr(args, f"{name.lower()}_lr") for name in OPTIMIZERS]
    if any(rate is not None and (not np.isfinite(rate) or rate <= 0) for rate in rates):
        parser.error("Learning rates and decay must be finite and positive")
    if len(set(args.optimizers)) != len(args.optimizers):
        parser.error("List each optimizer only once")
    return args


def load_data(data_dir):
    # Support the nested Kaggle layout used by mnist_data.py and flat IDX files.
    def locate(name):
        for path in (data_dir / name / name, data_dir / name, data_dir / name.replace("-idx", ".idx")):
            if path.is_file():
                return str(path)
        raise FileNotFoundError(f"Cannot find {name} under {data_dir}. Use --data-dir to select MNIST.")

    loader = MnistDataloader(*(locate(name) for name in (
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
    )))
    return loader.load_data()


def balanced_data(images, labels, args, rng, limit):
    labels = np.asarray(labels)
    positive = np.flatnonzero(labels == args.target)
    negative = np.flatnonzero(labels != args.target)
    if not len(positive) or not len(negative):
        raise ValueError("The dataset must include both the target digit and other digits")
    count = min(int(len(positive) * (1 - args.positive_fraction) / args.positive_fraction), len(negative))
    indices = rng.permutation(np.concatenate((positive, rng.choice(negative, count, replace=False))))
    if limit is not None:
        indices = indices[:limit]
    x = np.asarray([images[int(i)] for i in indices], dtype=np.float32)
    x = x.reshape(-1, 784, 1) / 255.0
    y = (labels[indices] == args.target).astype(np.float32).reshape(-1, 1, 1)
    return x, y


class BinaryClassifier:
    """The same 784 -> 64 -> 16 -> 1 sigmoid network as mnist_test.py."""

    def __init__(self, initial_parameters):
        self.layers = [candle.Linear(784, 64), candle.Linear(64, 16), candle.Linear(16, 1)]
        for parameter, initial in zip(self.parameters(), initial_parameters):
            parameter.inplace_add(candle.Tensor(initial.tolist()), self_factor=0.0)

    def parameters(self):
        return [parameter for layer in self.layers for parameter in (layer.weights, layer.bias)]

    def forward(self, inputs):
        for layer in self.layers:
            inputs = candle.activations.sigmoid(layer.forward(inputs))
        return inputs


def make_optimizer(name, lr):
    return {
        "SGD": lambda: candle.SGD(lr),
        "SGDMomentum": lambda: candle.SGDMomentum(lr, momentum=0.9),
        "AdaGrad": lambda: candle.AdaGrad(lr, eps=0.0001),
        "RMSProp": lambda: candle.RMSProp(lr, beta=0.99, eps=0.0001),
        "Adam": lambda: candle.Adam(lr, beta1=0.9, beta2=0.999, eps=0.0001),
    }[name]()


def run_batches(model, x, y, batch_size, order, optimizer=None, description="Testing"):
    total_loss, correct, total = 0.0, 0, 0
    training = optimizer is not None
    # Candle has no no_grad context; disable parameter tracking for evaluation.
    for parameter in model.parameters():
        parameter.requires_grad = training
    try:
        batches = range(0, len(order), batch_size)
        for start in tqdm(batches, desc=description, leave=False):
            indices = order[start:start + batch_size]
            count = len(indices)
            # Candle's bias broadcasting rejects a batch of one. Repeating that
            # example preserves mean loss/gradients; count it once in metrics.
            batch_indices = np.repeat(indices, 2) if count == 1 else indices
            inputs = candle.Tensor(x[batch_indices].tolist())
            labels = candle.Tensor(y[batch_indices].tolist())
            if training:
                optimizer.zero_grad()
            output = model.forward(inputs)
            loss = candle.losses.mse(output, labels)
            predictions = output.numpy()
            batch_loss = float(np.asarray(loss.tolist()).item())
            if not np.isfinite(batch_loss) or not np.isfinite(predictions).all():
                raise RuntimeError(f"Non-finite values during {description}; try a smaller learning rate")
            total_loss += batch_loss * count
            correct += int(np.count_nonzero((predictions[:count] > 0.5) == (y[indices] > 0.5)))
            total += count
            if training:
                loss.backward()
                optimizer.step()
    finally:
        for parameter in model.parameters():
            parameter.requires_grad = True
    return total_loss / total, correct / total


def save_results(rows, output_dir, target=None, loss_name="MSE"):
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for name in dict.fromkeys(row["optimizer"] for row in rows):
        history = [row for row in rows if row["optimizer"] == name]
        epochs = [row["epoch"] for row in history]
        for axis, key in zip(axes.flat, ("train_loss", "test_loss", "train_accuracy", "test_accuracy")):
            values = [row[key] * (100 if "accuracy" in key else 1) for row in history]
            axis.plot(epochs, values, marker="o", markersize=3, label=name)
    for axis, title in zip(axes.flat, (f"Training loss ({loss_name})", f"Test loss ({loss_name})", "Training accuracy (%)", "Test accuracy (%)")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        axis.set_xticks(sorted({row["epoch"] for row in rows}))
        if "accuracy" in title.lower():
            axis.set_ylim(0, 100)
    fig.suptitle("MNIST: digits 0-9" if target is None else f"MNIST: digit {target} vs. other digits")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_accuracy.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    (images, labels), (test_images, test_labels) = load_data(args.data_dir)
    x_train, y_train = balanced_data(images, labels, args, rng, args.train_limit)
    x_test, y_test = balanced_data(test_images, test_labels, args, rng, args.test_limit)
    del images, labels, test_images, test_labels
    # Match Linear's current uniform [-1, 1] initialization, using a repeatable seed.
    initial = [rng.uniform(-1, 1, shape).astype(np.float32) for shape in (
        (64, 784), (64, 1), (16, 64), (16, 1), (1, 16), (1, 1),
    )]
    config = vars(args) | {"train_examples": len(x_train), "test_examples": len(x_test)}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"Train: {len(x_train)}, test: {len(x_test)}; {args.epochs} epochs per optimizer")
    rows = []
    for name in args.optimizers:
        model = BinaryClassifier(initial)
        lr = getattr(args, f"{name.lower()}_lr")
        optimizer = make_optimizer(name, args.lr if lr is None else lr)
        for layer in model.layers:
            optimizer.add_layer(layer)
        optimizer.init_grad()
        # Reset the shuffle generator so every optimizer sees identical batches.
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
            print(f"{name:12s} epoch {epoch:2d}: train MSE={train_loss:.5f}, "
                  f"accuracy={train_accuracy:.2%}; test MSE={test_loss:.5f}, accuracy={test_accuracy:.2%}")
            save_results(rows, args.output_dir, args.target)
            optimizer.lr *= args.lr_decay
        del optimizer, model
    print(f"Saved curves, metrics.csv and config.json to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
