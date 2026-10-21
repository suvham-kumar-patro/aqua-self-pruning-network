import numpy as np
from engine.tensor import Tensor
from nn.optim import Adam
from nn.layers import MLP


def test_adam_first_step_matches_formula():
    p = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    p.grad[:] = [0.5, -0.25]
    opt = Adam([p], lr=0.01)
    opt.step()
    expected = np.array([1.0, -2.0]) - 0.01 * np.array([0.5, -0.25]) / (np.array([0.5, 0.25]) + 1e-8)
    np.testing.assert_allclose(p.data, expected, atol=1e-12)


def test_mlp_output_shape():
    model = MLP(np.random.default_rng(7))
    result = model(Tensor(np.zeros((5, 2))))
    assert result.shape == (5, 3)
    assert len(model.parameters()) == 6
