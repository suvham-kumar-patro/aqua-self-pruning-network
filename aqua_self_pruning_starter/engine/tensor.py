"""A minimal reverse-mode autodiff Tensor built with NumPy.

Design note: gradients are accumulated into parent tensors. Each operation stores
a local backward closure. `backward()` rebuilds a topological order for the
current graph on every call, so gradients are propagated through the graph that
was actually used by the latest forward computation.

This starter intentionally supports 2-D matrix multiplication, which is enough
for the first MLP implementation. Extend it only after the current tests pass.
"""
from __future__ import annotations

import numpy as np


def _as_array(value):
    return value.data if isinstance(value, Tensor) else np.asarray(value, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Sum broadcast axes so `grad` matches the original operand shape."""
    grad = np.asarray(grad, dtype=np.float64)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _backward=None, name=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._parents = tuple(p for p in _parents if p.requires_grad)
        self._backward = _backward or (lambda: None)
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def zero_grad(self):
        if self.requires_grad:
            self.grad[...] = 0.0

    def _result(self, data, parents, backward):
        return Tensor(data, requires_grad=any(p.requires_grad for p in parents),
                      _parents=parents, _backward=backward)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._result(self.data + other.data, (self, other), lambda: None)
        def backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = self._result(-self.data, (self,), lambda: None)
        out._backward = lambda: self.grad.__isub__(out.grad) if self.requires_grad else None
        return out

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -np.asarray(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._result(self.data * other.data, (self, other), lambda: None)
        def backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.shape)
        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._result(self.data / other.data, (self, other), lambda: None)
        def backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad / other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
        out._backward = backward
        return out

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.ndim != 2 or other.ndim != 2:
            raise NotImplementedError("This starter supports 2-D matrix multiplication only.")
        out = self._result(self.data @ other.data, (self, other), lambda: None)
        def backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = self._result(self.data.sum(axis=axis, keepdims=keepdims), (self,), lambda: None)
        def backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                for a in sorted(axes):
                    g = np.expand_dims(g, axis=a)
            self.grad += np.broadcast_to(g, self.shape)
        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def relu(self):
        out = self._result(np.maximum(self.data, 0.0), (self,), lambda: None)
        out._backward = lambda: self.grad.__iadd__(out.grad * (self.data > 0)) if self.requires_grad else None
        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = self._result(value, (self,), lambda: None)
        out._backward = lambda: self.grad.__iadd__(out.grad * (1.0 - value ** 2)) if self.requires_grad else None
        return out

    def sigmoid(self):
        # Stable sigmoid: separate formulas avoid overflow for large magnitudes.
        value = np.empty_like(self.data)
        pos = self.data >= 0
        value[pos] = 1.0 / (1.0 + np.exp(-self.data[pos]))
        exp_x = np.exp(self.data[~pos])
        value[~pos] = exp_x / (1.0 + exp_x)
        out = self._result(value, (self,), lambda: None)
        out._backward = lambda: self.grad.__iadd__(out.grad * value * (1.0 - value)) if self.requires_grad else None
        return out

    def backward(self, grad=None):
        """Backpropagate from this tensor.

        For a scalar output, omitted `grad` means d(output)/d(output) = 1.
        For non-scalar outputs, callers must supply an upstream gradient.
        """
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Pass an explicit gradient for non-scalar outputs.")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != self.shape:
            raise ValueError(f"upstream gradient shape {grad.shape} != output shape {self.shape}")

        topo, seen = [], set()
        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                visit(parent)
            topo.append(node)
        visit(self)

        # Clear intermediate gradients so repeated backward() calls on a graph
        # do not accidentally reuse old intermediate derivatives. Leaf gradients
        # accumulate, matching common autodiff conventions; call zero_grad().
        for node in topo:
            if node is not self and node._parents:
                node.grad = np.zeros_like(node.data) if node.requires_grad else None
        if self.requires_grad:
            if self._parents:
                # The root's upstream gradient is for this backward invocation,
                # not an accumulated leaf gradient.
                self.grad = np.asarray(grad, dtype=np.float64).copy()
            else:
                # A leaf can accumulate gradients across separate backward calls.
                self.grad += grad
        else:
            # Allow backward on a non-leaf result if its parents require gradients.
            self.grad = np.asarray(grad, dtype=np.float64).copy()
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"
