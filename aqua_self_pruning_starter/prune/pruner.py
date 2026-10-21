
"""Global pruning with an exact active-weight budget."""

import numpy as np

from prune.criteria import (
    magnitude_scores,
    taylor_scores,
    update_saliency,
)


class GlobalPruner:
    def __init__(self, model, optimizer,
                 criterion="taylor", decay=0.9):
        if criterion not in ("taylor", "magnitude"):
            raise ValueError("Unknown pruning criterion")

        self.layers = model.layers
        self.optimizer = optimizer
        self.criterion = criterion
        self.decay = decay

        self.saliency = [None] * len(self.layers)

        self.total = sum(
            layer.weight.data.size
            for layer in self.layers
        )

    def observe(self):
        """Call after backward(), before optimizer.step()."""
        if self.criterion != "taylor":
            return

        for i, layer in enumerate(self.layers):
            score = taylor_scores(layer.weight)

            self.saliency[i] = update_saliency(
                self.saliency[i],
                score,
                self.decay
            )

    def sparsity(self):
        active = sum(
            np.count_nonzero(layer.mask)
            for layer in self.layers
        )
        return 1 - active / self.total

    def prune_to(self, target_sparsity):
        if not 0 <= target_sparsity < 1:
            raise ValueError("Invalid target sparsity")

        target_pruned = round(
            target_sparsity * self.total
        )

        currently_pruned = sum(
            np.count_nonzero(layer.mask == 0)
            for layer in self.layers
        )

        to_prune = target_pruned - currently_pruned

        if to_prune <= 0:
            return self.sparsity()

        scores = []
        active_locations = []

        for layer_idx, layer in enumerate(self.layers):
            if self.criterion == "magnitude":
                importance = magnitude_scores(
                    layer.weight
                )
            else:
                importance = self.saliency[layer_idx]

                if importance is None:
                    raise RuntimeError(
                        "Collect Taylor saliency before pruning"
                    )

            flat_mask = layer.mask.ravel()
            flat_scores = importance.ravel()

            for weight_idx in np.flatnonzero(flat_mask):
                scores.append(flat_scores[weight_idx])
                active_locations.append(
                    (layer_idx, weight_idx)
                )

        if to_prune > len(active_locations):
            raise ValueError("Insufficient active weights")

        scores = np.asarray(scores)

        if not np.all(np.isfinite(scores)):
            raise FloatingPointError(
                "Non-finite pruning importance"
            )

        # Stable tie-breaking for reproducibility.
        selected = np.argsort(
            scores, kind="stable"
        )[:to_prune]

        for index in selected:
            layer_idx, weight_idx = active_locations[index]
            self.layers[layer_idx].mask.ravel()[
                weight_idx
            ] = 0.0

        # Notify Adam and reset pruned optimizer state.
        for layer in self.layers:
            self.optimizer.set_mask(
                layer.weight,
                layer.mask
            )

        return self.sparsity()
