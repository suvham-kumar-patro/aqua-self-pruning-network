
"""Adam optimizer with explicit mask-state management."""

import numpy as np


class Adam:
    def __init__(
        self,
        parameters,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [
            np.zeros_like(p.data)
            for p in self.parameters
        ]

        self.v = [
            np.zeros_like(p.data)
            for p in self.parameters
        ]

        # Each coordinate has its own Adam age.
        self.age = [
            np.zeros_like(p.data, dtype=np.int64)
            for p in self.parameters
        ]

        # All parameters are active initially.
        self.masks = [
            np.ones_like(p.data, dtype=bool)
            for p in self.parameters
        ]

        self.t = 0

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def set_mask(self, parameter, new_mask):
        """Register a mask and reset changed connections."""
        index = next(
            (
                i for i, p in enumerate(self.parameters)
                if p is parameter
            ),
            None
        )

        if index is None:
            raise ValueError("Unknown optimizer parameter")

        new_mask = np.asarray(new_mask)

        if new_mask.shape != parameter.data.shape:
            raise ValueError("Mask shape mismatch")

        if not np.all((new_mask == 0) | (new_mask == 1)):
            raise ValueError("Mask must contain only 0 or 1")

        new_mask = new_mask.astype(bool, copy=True)

        old_mask = self.masks[index]
        changed = old_mask != new_mask

        # Both pruning and revival invalidate old moments.
        self.m[index][changed] = 0.0
        self.v[index][changed] = 0.0
        self.age[index][changed] = 0

        # Pruned connections are physically stored as zero.
        # Revived connections also start from zero.
        parameter.data[changed] = 0.0
        parameter.data[~new_mask] = 0.0

        if parameter.grad is not None:
            parameter.grad[changed] = 0.0
            parameter.grad[~new_mask] = 0.0

        self.m[index][~new_mask] = 0.0
        self.v[index][~new_mask] = 0.0
        self.age[index][~new_mask] = 0

        self.masks[index] = new_mask

    def step(self):
        self.t += 1

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            active = self.masks[i]
            g = np.where(active, p.grad, 0.0)

            if not np.all(np.isfinite(g)):
                raise FloatingPointError(
                    "Non-finite gradient encountered"
                )

            # Clear inactive optimizer state.
            self.m[i][~active] = 0.0
            self.v[i][~active] = 0.0
            self.age[i][~active] = 0

            # Update only active coordinates.
            self.m[i][active] = (
                self.beta1 * self.m[i][active]
                + (1 - self.beta1) * g[active]
            )

            self.v[i][active] = (
                self.beta2 * self.v[i][active]
                + (1 - self.beta2) * g[active] ** 2
            )

            self.age[i][active] += 1

            age = self.age[i][active]

            m_hat = self.m[i][active] / (
                1 - self.beta1 ** age
            )

            v_hat = self.v[i][active] / (
                1 - self.beta2 ** age
            )

            p.data[active] -= (
                self.lr * m_hat
                / (np.sqrt(v_hat) + self.eps)
            )

            # Enforce hard zeros.
            p.data[~active] = 0.0
