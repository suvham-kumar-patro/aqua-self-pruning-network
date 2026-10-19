
import numpy as np
import pytest

from engine.tensor import Tensor


EPS = 1e-5


def numerical_gradient(fn, inputs, arg_index):
    """Compute gradients using central finite differences."""
    values = [np.array(x, dtype=float, copy=True) for x in inputs]
    target = values[arg_index]
    gradient = np.zeros_like(target)

    for index in np.ndindex(target.shape):
        original = target[index]

        target[index] = original + EPS
        plus = float(fn(*values))

        target[index] = original - EPS
        minus = float(fn(*values))

        target[index] = original

        gradient[index] = (plus - minus) / (2 * EPS)

    return gradient


def check_gradient(tensor_fn, numpy_fn, *inputs):
    tensors = [Tensor(x, requires_grad=True) for x in inputs]

    output = tensor_fn(*tensors)
    output.backward()

    for i, tensor in enumerate(tensors):
        expected = numerical_gradient(numpy_fn, inputs, i)

        np.testing.assert_allclose(
            tensor.grad,
            expected,
            rtol=1e-4,
            atol=1e-6,
        )


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_arithmetic(op):
    a = np.array([[1.2, 2.3], [3.4, 4.5]])
    b = np.array([1.5, 2.5])

    tensor_ops = {
        "add": lambda x, y: (x + y).sum(),
        "sub": lambda x, y: (x - y).sum(),
        "mul": lambda x, y: (x * y).sum(),
        "div": lambda x, y: (x / y).sum(),
    }

    numpy_ops = {
        "add": lambda x, y: np.sum(x + y),
        "sub": lambda x, y: np.sum(x - y),
        "mul": lambda x, y: np.sum(x * y),
        "div": lambda x, y: np.sum(x / y),
    }

    check_gradient(tensor_ops[op], numpy_ops[op], a, b)


def test_matmul():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.5, 1.5], [2.5, 3.5]])

    check_gradient(
        lambda x, y: (x @ y).sum(),
        lambda x, y: np.sum(x @ y),
        a, b
    )


@pytest.mark.parametrize("op", ["relu", "tanh", "sigmoid"])
def test_activation(op):
    a = np.array([-2.0, -0.5, 0.5, 2.0])

    numpy_ops = {
        "relu": lambda x: np.maximum(x, 0),
        "tanh": np.tanh,
        "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    }

    check_gradient(
        lambda x: getattr(x, op)().sum(),
        lambda x: np.sum(numpy_ops[op](x)),
        a
    )


@pytest.mark.parametrize("op", ["sum", "mean"])
def test_reduction(op):
    a = np.array([[1.2, 2.3], [3.4, 4.5]])

    check_gradient(
        lambda x: getattr(x, op)(),
        lambda x: getattr(np, op)(x),
        a
    )


def test_shared_parent():
    a = np.array([1.2, -0.7, 2.4])

    check_gradient(
        lambda x: (x * x + x).sum(),
        lambda x: np.sum(x * x + x),
        a
    )
