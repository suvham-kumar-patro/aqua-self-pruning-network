
"""Sparse-aware NumPy inference with CSR-like weight storage."""

import numpy as np


class SparseLinear:
    def __init__(self, layer):
        weight = layer.weight.data * layer.mask
        self.bias = layer.bias.data.copy()

        self.in_features, self.out_features = weight.shape

        self.indptr = [0]
        indices = []
        values = []

        for row in range(self.in_features):
            cols = np.flatnonzero(layer.mask[row])
            indices.extend(cols.tolist())
            values.extend(weight[row, cols].tolist())
            self.indptr.append(len(indices))

        self.indptr = np.asarray(self.indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64)

    @property
    def nnz(self):
        return len(self.values)

    def __call__(self, x):
        output = np.broadcast_to(
            self.bias,
            (len(x), self.out_features)
        ).copy()

        for row in range(self.in_features):
            start = self.indptr[row]
            end = self.indptr[row + 1]

            for k in range(start, end):
                col = self.indices[k]
                output[:, col] += x[:, row] * self.values[k]

        return output


class SparseMLP:
    def __init__(self, model):
        self.layers = [
            SparseLinear(layer)
            for layer in model.layers
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = np.maximum(layer(x), 0.0)

        return self.layers[-1](x)

    def macs_per_sample(self):
        return sum(layer.nnz for layer in self.layers)


def dense_macs_per_sample(model):
    return sum(
        layer.weight.data.size
        for layer in model.layers
    )
