
"""Importance scores for network pruning."""

import numpy as np


def magnitude_scores(weight):
    """Magnitude-only pruning baseline."""
    return np.abs(weight.data)


def taylor_scores(weight, gradient=None):
    """
    First-order Taylor importance: abs(weight * gradient).

    Use gradients collected from active connections
    before applying a pruning update.
    """
    if gradient is None:
        gradient = weight.grad

    if gradient is None:
        raise ValueError("Gradient is required")

    if gradient.shape != weight.data.shape:
        raise ValueError("Gradient shape mismatch")

    return np.abs(weight.data * gradient)


def update_saliency(previous, current, decay=0.9):
    """Exponential moving average of saliency."""
    if not 0 <= decay < 1:
        raise ValueError("decay must be in [0, 1)")

    if previous is None:
        return current.copy()

    if previous.shape != current.shape:
        raise ValueError("Saliency shape mismatch")

    return decay * previous + (1 - decay) * current
