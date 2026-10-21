"""Train an MLP on three synthetic Gaussian clusters.

Run from the repository root:
    python -m train.train_mlp
"""
from pathlib import Path
import csv
import numpy as np
from engine.tensor import Tensor
from engine.losses import softmax_cross_entropy
from nn.layers import MLP
from nn.optim import Adam


def make_data(seed=42, n_per_class=250):
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0, -1.5], [2.0, -1.5], [0.0, 2.0]])
    xs, ys = [], []
    for label, center in enumerate(centers):
        xs.append(rng.normal(loc=center, scale=0.85,
                             size=(n_per_class, 2)))
        ys.append(np.full(n_per_class, label, dtype=int))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    order = rng.permutation(len(y))
    return x[order], y[order]


def accuracy(model, x, y):
    logits = model(Tensor(x)).data
    return float(np.mean(np.argmax(logits, axis=1) == y))


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    rng = np.random.default_rng(7)
    model = MLP(rng)
    optim = Adam(model.parameters(), lr=0.01)
    history = []

    for epoch in range(1, 61):
        permutation = rng.permutation(len(y_train))
        total_loss = 0.0
        for start in range(0, len(y_train), 64):
            idx = permutation[start:start + 64]
            optim.zero_grad()
            logits = model(Tensor(x_train[idx]))
            loss = softmax_cross_entropy(logits, y_train[idx])
            if not np.isfinite(loss.data).all():
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optim.step()
            total_loss += float(loss.data) * len(idx)

        train_loss = total_loss / len(y_train)
        train_acc = accuracy(model, x_train, y_train)
        val_acc = accuracy(model, x_val, y_val)
        history.append((epoch, train_loss, train_acc, val_acc))
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | loss={train_loss:.4f} "
                  f"| train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

    output = Path('results')
    output.mkdir(exist_ok=True)
    with (output / 'baseline_learning_curve.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_accuracy'])
        writer.writerows(history)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [r[0] for r in history]
    fig, ax = plt.subplots()
    ax.plot(epochs, [r[1] for r in history], label='Training loss')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss',
           title='AQUA baseline training loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'baseline_loss.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, [r[2] for r in history], label='Train accuracy')
    ax.plot(epochs, [r[3] for r in history], label='Validation accuracy')
    ax.set(xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1.02),
           title='AQUA baseline accuracy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'baseline_accuracy.png', dpi=160)
    plt.close(fig)
    print('Saved CSV and plots in results/')


if __name__ == '__main__':
    main()
