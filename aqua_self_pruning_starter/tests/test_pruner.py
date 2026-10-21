
import numpy as np

from nn.layers import MLP
from nn.optim import Adam
from prune.pruner import GlobalPruner
from prune.schedule import cubic_sparsity


def test_cubic_schedule():
    assert cubic_sparsity(0, 0, 10, 0.9) == 0
    assert np.isclose(
        cubic_sparsity(10, 0, 10, 0.9), 0.9
    )

    midpoint = cubic_sparsity(5, 0, 10, 0.9)
    assert 0 < midpoint < 0.9


def test_global_magnitude_pruning():
    model = MLP(np.random.default_rng(42))
    optimizer = Adam(model.parameters())

    pruner = GlobalPruner(
        model, optimizer, criterion="magnitude"
    )

    achieved = pruner.prune_to(0.5)

    assert abs(achieved - 0.5) < 0.001

    for layer in model.layers:
        assert np.all(
            layer.weight.data[layer.mask == 0] == 0
        )


def test_taylor_requires_observation():
    model = MLP(np.random.default_rng(42))
    optimizer = Adam(model.parameters())

    pruner = GlobalPruner(
        model, optimizer, criterion="taylor"
    )

    try:
        pruner.prune_to(0.5)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Taylor pruning must require gradients"
        )
