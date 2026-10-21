
"""Compare graph-free dense NumPy against sparse NumPy inference."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark.benchmark_checkpoint import load_model
from benchmark.sparse_inference import SparseMLP



def prepare_dense_weights(model):
    return [
        (
            layer.weight.data * layer.mask,
            layer.bias.data.copy(),
        )
        for layer in model.layers
    ]


def dense_numpy_forward(prepared_layers, x):
    for weight, bias in prepared_layers[:-1]:
        x = np.maximum(x @ weight + bias, 0.0)

    weight, bias = prepared_layers[-1]
    return x @ weight + bias



def median_latency_ms(fn, repeats=1000, warmups=100):
    for _ in range(warmups):
        fn()

    times = []

    for _ in range(repeats):
        start = perf_counter()
        fn()
        times.append((perf_counter() - start) * 1000)

    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    sparse = SparseMLP(model)

    x = np.random.default_rng(123).normal(
        size=(args.batch_size, 2)
    )

    prepared_dense = prepare_dense_weights(model)
    dense_fn = lambda: dense_numpy_forward(prepared_dense, x)
    sparse_fn = lambda: sparse(x)

    np.testing.assert_allclose(
        dense_fn(),
        sparse_fn(),
        rtol=1e-10,
        atol=1e-10,
    )

    dense_ms = median_latency_ms(dense_fn)
    sparse_ms = median_latency_ms(sparse_fn)

    total = sum(
        layer.weight.data.size for layer in model.layers
    )
    active = sparse.macs_per_sample()

    row = {
        "checkpoint": Path(args.checkpoint).name,
        "batch_size": args.batch_size,
        "active_weights": active,
        "total_weights": total,
        "achieved_sparsity": 1 - active / total,
        "dense_macs_per_sample": total,
        "sparse_macs_per_sample": active,
        "dense_median_ms": dense_ms,
        "sparse_median_ms": sparse_ms,
        "sparse_over_dense_latency": sparse_ms / dense_ms,
    }

    output = Path("results")
    output.mkdir(exist_ok=True)

    path = output / (
        f"fair_benchmark_{Path(args.checkpoint).stem}.csv"
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=row.keys()
        )
        writer.writeheader()
        writer.writerow(row)

    for key, value in row.items():
        print(f"{key}: {value}")

    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
