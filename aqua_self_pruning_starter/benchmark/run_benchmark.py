
"""Benchmark dense and sparse-aware forward passes."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from engine.tensor import Tensor
from nn.layers import MLP
from nn.optim import Adam
from prune.pruner import GlobalPruner

from benchmark.sparse_inference import (
    SparseMLP,
    dense_macs_per_sample,
)


def measure(fn, repeats=300, warmups=30):
    for _ in range(warmups):
        fn()

    samples = []

    for _ in range(repeats):
        start = perf_counter()
        fn()
        samples.append((perf_counter() - start) * 1000)

    return float(np.median(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model = MLP(np.random.default_rng(7))
    optimizer = Adam(model.parameters())

    pruner = GlobalPruner(
        model, optimizer, criterion="magnitude"
    )
    pruner.prune_to(args.sparsity)

    sparse = SparseMLP(model)

    x = np.random.default_rng(99).normal(
        size=(args.batch_size, 2)
    )

    def dense_forward():
        return model(Tensor(x)).data

    def sparse_forward():
        return sparse(x)

    np.testing.assert_allclose(
        dense_forward(),
        sparse_forward(),
        rtol=1e-10,
        atol=1e-10,
    )

    dense_ms = measure(dense_forward)
    sparse_ms = measure(sparse_forward)

    dense_macs = dense_macs_per_sample(model)
    sparse_macs = sparse.macs_per_sample()

    row = {
        "target_sparsity": args.sparsity,
        "achieved_sparsity": pruner.sparsity(),
        "batch_size": args.batch_size,
        "dense_macs_per_sample": dense_macs,
        "sparse_macs_per_sample": sparse_macs,
        "dense_median_ms": dense_ms,
        "sparse_median_ms": sparse_ms,
        "latency_speedup": dense_ms / sparse_ms,
    }

    output = Path("results")
    output.mkdir(exist_ok=True)

    path = output / (
        f"inference_benchmark_s"
        f"{int(args.sparsity * 100)}.csv"
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    for key, value in row.items():
        print(f"{key}: {value}")

    print(f"Saved {path}")


if __name__ == "__main__":
    main()
