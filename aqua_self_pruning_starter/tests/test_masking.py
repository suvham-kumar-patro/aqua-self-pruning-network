
import numpy as np

from engine.tensor import Tensor
from prune.masking import WeightMask


def test_masked_weight_forward_and_backward():
    weight = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True
    )

    mask = WeightMask(weight.shape)
    mask.prune((0, 1))

    effective = mask.apply(weight)

    np.testing.assert_allclose(
        effective.data,
        [[1.0, 0.0], [3.0, 4.0]]
    )

    loss = effective.sum()
    loss.backward()

    np.testing.assert_allclose(
        weight.grad,
        [[1.0, 0.0], [1.0, 1.0]]
    )


def test_mask_sparsity():
    mask = WeightMask((2, 2))

    mask.prune((0, 1))

    assert mask.active_count() == 3
    assert mask.sparsity() == 0.25
