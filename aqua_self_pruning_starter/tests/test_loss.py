
import numpy as np

from engine.tensor import Tensor
from engine.losses import softmax_cross_entropy


def test_cross_entropy_known_value():
    logits = Tensor(
        [[2.0, 1.0, 0.1]],
        requires_grad=True
    )
    targets = np.array([0])

    loss = softmax_cross_entropy(logits, targets)

    np.testing.assert_allclose(
        loss.data,
        0.4170300163,
        atol=1e-8
    )


def test_cross_entropy_gradient():
    logits = Tensor(
        [[2.0, 1.0, 0.1],
         [0.5, 1.5, 2.0]],
        requires_grad=True
    )

    targets = np.array([0, 2])

    loss = softmax_cross_entropy(logits, targets)
    loss.backward()

    probabilities = np.exp(
        logits.data - logits.data.max(
            axis=1, keepdims=True
        )
    )

    probabilities /= probabilities.sum(
        axis=1, keepdims=True
    )

    expected = probabilities.copy()
    expected[np.arange(2), targets] -= 1
    expected /= 2

    np.testing.assert_allclose(
        logits.grad, expected, atol=1e-8
    )


def test_extreme_logits_no_nan():
    logits = Tensor(
        [[1000.0, 1001.0, 999.0],
         [-1000.0, -999.0, -1001.0]],
        requires_grad=True
    )

    loss = softmax_cross_entropy(
        logits, np.array([1, 1])
    )

    loss.backward()

    assert np.isfinite(loss.data).all()
    assert np.isfinite(logits.grad).all()


def test_cross_entropy_finite_difference():
    values = np.array([
        [0.2, 1.3, -0.7],
        [1.1, -0.4, 0.6]
    ])

    targets = np.array([1, 0])
    logits = Tensor(values.copy(), requires_grad=True)

    loss = softmax_cross_entropy(logits, targets)
    loss.backward()

    eps = 1e-5
    numerical = np.zeros_like(values)

    for idx in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()

        plus[idx] += eps
        minus[idx] -= eps

        loss_plus = softmax_cross_entropy(
            Tensor(plus), targets
        ).data

        loss_minus = softmax_cross_entropy(
            Tensor(minus), targets
        ).data

        numerical[idx] = (
            loss_plus - loss_minus
        ) / (2 * eps)

    np.testing.assert_allclose(
        logits.grad,
        numerical,
        rtol=1e-4,
        atol=1e-6
    )
