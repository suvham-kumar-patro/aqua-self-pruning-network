
"""Benchmark a trained, saved AQUA model."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from engine.tensor import Tensor
from nn.layers import MLP
from benchmark.sparse_inference import SparseMLP


def load_model(path):
    model = MLP(np.random.default_rng(0))

    with np.load(path, allow_pickle=False) as data:
        for i, layer in enumerate(model.layers):
            layer.weight.data[:] = data[f"weight_{i}"]
            layer.bias.data[:] = data[f"bias_{i}"]
            layer.mask[:] = data[f"mask_{i}"]

    return model


def median_ms(fn, repeats=500):
    for _ in range(50):
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

    dense_ms = median_ms(dense_forward)
    sparse_ms = median_ms(sparse_forward)

    total = sum(
        layer.weight.data.size
        for layer in model.layers
    )

    active = sparse.macs_per_sample()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Active weights: {active}/{total}")
    print(f"Achieved sparsity: {1 - active / total:.4%}")
    print(f"Dense MACs/sample: {total}")
    print(f"Sparse MACs/sample: {active}")
    print(f"Dense median latency: {dense_ms:.6f} ms")
    print(f"Sparse median latency: {sparse_ms:.6f} ms")
    print(f"Sparse/dense latency ratio: {sparse_ms / dense_ms:.3f}x")


if __name__ == "__main__":
    main()
