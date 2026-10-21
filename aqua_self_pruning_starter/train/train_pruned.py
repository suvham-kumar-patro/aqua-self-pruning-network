
"""Train an MLP with gradual global pruning.

Run:
python -m train.train_pruned --criterion taylor --sparsity 0.9
python -m train.train_pruned --criterion magnitude --sparsity 0.9
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from engine.tensor import Tensor
from engine.losses import softmax_cross_entropy
from nn.layers import MLP
from nn.optim import Adam
from prune.pruner import GlobalPruner
from prune.schedule import cubic_sparsity
from train.train_mlp import make_data, accuracy


def run_experiment(
    criterion="taylor",
    target_sparsity=0.9,
    seed=7,
    epochs=60,
    output_dir="results",
):
    if criterion not in ("taylor", "magnitude"):
        raise ValueError("Invalid criterion")

    if not 0 <= target_sparsity < 1:
        raise ValueError("Invalid sparsity")

    x, y = make_data(seed=42)

    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    rng = np.random.default_rng(seed)

    model = MLP(rng)
    optimizer = Adam(model.parameters(), lr=0.01)

    pruner = GlobalPruner(
        model,
        optimizer,
        criterion=criterion,
        decay=0.9,
    )

    history = []

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(y_train))
        total_loss = 0.0

        for start in range(0, len(y_train), 64):
            idx = permutation[start:start + 64]

            optimizer.zero_grad()

            logits = model(Tensor(x_train[idx]))
            loss = softmax_cross_entropy(
                logits, y_train[idx]
            )

            if not np.isfinite(loss.data).all():
                raise FloatingPointError(
                    "Non-finite training loss"
                )

            loss.backward()

            # Collect sensitivity before optimizer updates.
            pruner.observe()

            optimizer.step()

            total_loss += float(loss.data) * len(idx)

        # Gradually reach the target by epoch 45.
        scheduled_sparsity = cubic_sparsity(
            epoch,
            start_epoch=5,
            end_epoch=45,
            target_sparsity=target_sparsity,
        )

        if scheduled_sparsity > pruner.sparsity():
            pruner.prune_to(scheduled_sparsity)

        train_loss = total_loss / len(y_train)
        train_acc = accuracy(model, x_train, y_train)
        val_acc = accuracy(model, x_val, y_val)

        achieved = pruner.sparsity()

        active = sum(
            np.count_nonzero(layer.mask)
            for layer in model.layers
        )

        history.append((
            epoch,
            train_loss,
            train_acc,
            val_acc,
            achieved,
            active,
        ))

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:02d} "
                f"| loss={train_loss:.4f} "
                f"| train_acc={train_acc:.3f} "
                f"| val_acc={val_acc:.3f} "
                f"| sparsity={achieved:.3f} "
                f"| active={active}"
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    name = (
        f"{criterion}_s{int(target_sparsity * 100)}"
        f"_seed{seed}"
    )

    csv_path = output / f"{name}.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_accuracy",
            "achieved_sparsity",
            "active_weights",
        ])

        writer.writerows(history)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    epoch_values = [row[0] for row in history]

    axes[0].plot(
        epoch_values,
        [row[3] for row in history],
        label="Validation accuracy",
    )

    axes[0].set(
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(0, 1.02),
        title="Validation accuracy",
    )
    axes[0].legend()

    axes[1].plot(
        epoch_values,
        [row[4] for row in history],
        label="Achieved sparsity",
    )

    axes[1].set(
        xlabel="Epoch",
        ylabel="Sparsity",
        ylim=(0, 1.02),
        title="Progressive pruning",
    )
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=160)
    plt.close(fig)

    print(f"\nFinal validation accuracy: {history[-1][3]:.4f}")
    print(f"Final achieved sparsity: {history[-1][4]:.4f}")
    print(f"Remaining weight connections: {history[-1][5]}")
    print(f"Saved results to {csv_path}")

    
    checkpoint_path = output / f"{name}.npz"

    checkpoint = {}

    for i, layer in enumerate(model.layers):
        checkpoint[f"weight_{i}"] = layer.weight.data.copy()
        checkpoint[f"bias_{i}"] = layer.bias.data.copy()
        checkpoint[f"mask_{i}"] = layer.mask.copy()

    np.savez(checkpoint_path, **checkpoint)

    print(f"Saved trained checkpoint to {checkpoint_path}")

    return history


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--criterion",
        choices=["taylor", "magnitude"],
        default="taylor",
    )

    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    run_experiment(
        criterion=args.criterion,
        target_sparsity=args.sparsity,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
