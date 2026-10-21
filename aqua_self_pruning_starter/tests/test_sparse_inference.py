
import numpy as np

from engine.tensor import Tensor
from nn.layers import MLP
from nn.optim import Adam
from prune.pruner import GlobalPruner

from benchmark.sparse_inference import (
    SparseMLP,
    dense_macs_per_sample,
)


def test_sparse_inference_matches_dense_masked():
    model = MLP(np.random.default_rng(42))
    optimizer = Adam(model.parameters())

    pruner = GlobalPruner(
        model,
        optimizer,
        criterion="magnitude",
    )

    pruner.prune_to(0.90)

    x = np.random.default_rng(123).normal(size=(32, 2))

    dense_output = model(Tensor(x)).data
    sparse_model = SparseMLP(model)
    sparse_output = sparse_model(x)

    np.testing.assert_allclose(
        sparse_output,
        dense_output,
        rtol=1e-10,
        atol=1e-10,
    )

    assert sparse_model.macs_per_sample() == 118
    assert dense_macs_per_sample(model) == 1184
