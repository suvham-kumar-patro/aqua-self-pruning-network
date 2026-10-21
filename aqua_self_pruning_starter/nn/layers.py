
"""Composable neural-network layers."""

import numpy as np
from engine.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features, rng):
        weights = rng.normal(
            0,
            np.sqrt(2.0 / in_features),
            size=(in_features, out_features)
        )

        self.weight = Tensor(weights, requires_grad=True)
        self.bias = Tensor(
            np.zeros(out_features),
            requires_grad=True
        )

        # Only weight connections are prunable.
        self.mask = np.ones_like(self.weight.data)

    def __call__(self, x):
        # Mask multiplication is part of the autodiff graph.
        effective_weight = self.weight * Tensor(self.mask.copy())
        return x @ effective_weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class MLP:
    def __init__(self, rng):
        self.layers = [
            Linear(2, 32, rng),
            Linear(32, 32, rng),
            Linear(32, 3, rng)
        ]

    def __call__(self, x):
        x = self.layers[0](x).relu()
        x = self.layers[1](x).relu()
        return self.layers[2](x)

    def parameters(self):
        return [
            p
            for layer in self.layers
            for p in layer.parameters()
        ]
