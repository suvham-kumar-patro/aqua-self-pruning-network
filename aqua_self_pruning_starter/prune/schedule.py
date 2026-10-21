
"""Gradual cubic sparsity scheduling."""


def cubic_sparsity(epoch, start_epoch, end_epoch,
                   target_sparsity):
    if not 0 <= target_sparsity < 1:
        raise ValueError("target_sparsity must be in [0, 1)")

    if end_epoch <= start_epoch:
        raise ValueError("end_epoch must exceed start_epoch")

    if epoch <= start_epoch:
        return 0.0

    if epoch >= end_epoch:
        return target_sparsity

    progress = (
        (epoch - start_epoch)
        / (end_epoch - start_epoch)
    )

    return target_sparsity * (
        1 - (1 - progress) ** 3
    )
