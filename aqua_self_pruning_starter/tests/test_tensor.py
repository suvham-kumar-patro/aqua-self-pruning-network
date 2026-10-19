import numpy as np
import pytest
from engine.tensor import Tensor


def test_broadcast_add_gradient():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    b = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    loss = (x + b).sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, np.ones((2, 3)))
    np.testing.assert_allclose(b.grad, np.array([2.0, 2.0, 2.0]))


def test_mul_and_div_gradients():
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    y = Tensor(np.array([4.0, 5.0]), requires_grad=True)
    ((x * y) / y).sum().backward()
    np.testing.assert_allclose(x.grad, np.ones(2), atol=1e-10)
    np.testing.assert_allclose(y.grad, np.zeros(2), atol=1e-10)


def test_matmul_gradients():
    x = Tensor(np.array([[1., 2.], [3., 4.]]), requires_grad=True)
    w = Tensor(np.array([[2., 0.], [1., 3.]]), requires_grad=True)
    (x @ w).sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((2, 2)) @ w.data.T)
    np.testing.assert_allclose(w.grad, x.data.T @ np.ones((2, 2)))


def test_sum_mean_gradient():
    x = Tensor(np.array([[1., 2.], [3., 4.]]), requires_grad=True)
    x.mean().backward()
    np.testing.assert_allclose(x.grad, np.full((2, 2), 0.25))


def test_activations():
    x = Tensor(np.array([-2., 0., 2.]), requires_grad=True)
    (x.relu().sum() + x.tanh().sum() + x.sigmoid().sum()).backward()
    expected = (np.array([0., 0., 1.]) +
                (1.0 - np.tanh(x.data) ** 2) +
                (1.0 / (1.0 + np.exp(-x.data))) *
                (1.0 - 1.0 / (1.0 + np.exp(-x.data))))
    np.testing.assert_allclose(x.grad, expected, atol=1e-8)


def test_non_scalar_backward_requires_seed():
    x = Tensor(np.array([1., 2.]), requires_grad=True)
    with pytest.raises(ValueError):
        x.backward()
    x.backward(np.array([2., 3.]))
    np.testing.assert_allclose(x.grad, np.array([2., 3.]))
