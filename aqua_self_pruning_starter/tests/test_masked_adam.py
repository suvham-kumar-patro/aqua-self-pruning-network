
import numpy as np

from engine.tensor import Tensor
from nn.optim import Adam


def test_pruned_weight_cannot_drift():
    w = Tensor([1.0, 2.0], requires_grad=True)
    opt = Adam([w], lr=0.1)

    # Build nonzero Adam momentum.
    w.grad[:] = [1.0, 1.0]
    opt.step()

    assert opt.m[0][0] != 0.0

    # Prune the first connection.
    opt.set_mask(w, np.array([0, 1]))

    assert w.data[0] == 0.0
    assert opt.m[0][0] == 0.0
    assert opt.v[0][0] == 0.0
    assert opt.age[0][0] == 0

    # Even a stale nonzero gradient cannot move it.
    for _ in range(5):
        w.grad[:] = [10.0, 1.0]
        opt.step()

        assert w.data[0] == 0.0
        assert opt.m[0][0] == 0.0
        assert opt.v[0][0] == 0.0


def test_revived_weight_has_fresh_adam_state():
    w = Tensor([1.0, 2.0], requires_grad=True)
    opt = Adam([w], lr=0.1)

    w.grad[:] = [1.0, 1.0]
    opt.step()

    opt.set_mask(w, np.array([0, 1]))

    for _ in range(10):
        w.grad[:] = [0.0, 1.0]
        opt.step()

    # Revive the first connection.
    opt.set_mask(w, np.array([1, 1]))

    assert opt.m[0][0] == 0.0
    assert opt.v[0][0] == 0.0
    assert opt.age[0][0] == 0

    w.grad[:] = [1.0, 1.0]
    opt.step()

    # First Adam update from zero with gradient 1.
    np.testing.assert_allclose(
        w.data[0], -0.1, atol=1e-7
    )

    assert opt.age[0][0] == 1


def test_masked_linear_gradient():
    from nn.layers import Linear

    rng = np.random.default_rng(42)
    layer = Linear(2, 2, rng)

    opt = Adam(layer.parameters())

    layer.mask[0, 1] = 0.0
    opt.set_mask(layer.weight, layer.mask)

    x = Tensor([[1.0, 2.0]])

    output = layer(x)
    loss = output.sum()

    opt.zero_grad()
    loss.backward()

    assert layer.weight.grad[0, 1] == 0.0

    opt.step()

    assert layer.weight.data[0, 1] == 0.0
