
import numpy as np

from engine.tensor import Tensor


class WeightMask:
    """Binary mask applied to a trainable weight tensor."""

    def __init__(self, shape):
        self.mask = np.ones(shape, dtype=np.float64)

    def apply(self, weight: Tensor) -> Tensor:
        if weight.shape != self.mask.shape:
            raise ValueError("Mask and weight shapes must match")

        # Multiplication stays inside the autodiff graph.
        return weight * Tensor(self.mask.copy())

    def prune(self, indices):
        self.mask[indices] = 0.0

    def active_count(self):
        return int(np.count_nonzero(self.mask))

    def sparsity(self):
        return 1.0 - self.active_count() / self.mask.size
