
import numpy as np
from engine.tensor import Tensor


def softmax_cross_entropy(logits: Tensor, targets):
    """
    Numerically stable softmax + cross-entropy.

    logits: Tensor with shape (batch_size, num_classes)
    targets: integer class labels with shape (batch_size,)
    returns: scalar Tensor containing mean batch loss
    """
    if logits.data.ndim != 2:
        raise ValueError("Logits must have shape (N, C)")

    n, num_classes = logits.data.shape

    if n == 0 or num_classes == 0:
        raise ValueError("Logits must not be empty")

    targets = np.asarray(targets)

    if targets.shape != (n,):
        raise ValueError("Targets must have shape (N,)")

    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Targets must be integer labels")

    if np.any(targets < 0) or np.any(targets >= num_classes):
        raise ValueError("Target class index out of range")

    # Shift logits for numerical stability.
    shifted = logits.data - np.max(
        logits.data, axis=1, keepdims=True
    )

    exp_logits = np.exp(shifted)

    probabilities = exp_logits / np.sum(
        exp_logits, axis=1, keepdims=True
    )

    # Stable log-sum-exp cross-entropy.
    log_sum_exp = np.log(
        np.sum(exp_logits, axis=1)
    )

    correct_logits = shifted[np.arange(n), targets]

    loss_value = np.mean(log_sum_exp - correct_logits)

    out = Tensor(
        loss_value,
        requires_grad=logits.requires_grad,
        _parents=(logits,)
    )

    def backward():
        if not logits.requires_grad:
            return

        grad = probabilities.copy()
        grad[np.arange(n), targets] -= 1.0
        grad /= n

        # Chain rule: include upstream gradient.
        logits.grad += grad * out.grad

    out._backward = backward

    return out
