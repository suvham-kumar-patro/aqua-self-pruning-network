
"""Reproducible sparsity-accuracy sweep."""

import csv
from pathlib import Path

import numpy as np

from train.train_pruned import run_experiment


def main():
    output = Path("results")
    output.mkdir(exist_ok=True)

    targets = [0.0, 0.50, 0.75, 0.90, 0.95]
    seeds = [7, 17, 27]
    criteria = ["taylor", "magnitude"]

    rows = []

    for seed in seeds:
        for criterion in criteria:
            for target in targets:
                print(
                    f"\nRunning criterion={criterion}, "
                    f"target={target:.0%}, seed={seed}"
                )

                history = run_experiment(
                    criterion=criterion,
                    target_sparsity=target,
                    seed=seed,
                )

                final = history[-1]

                rows.append({
                    "seed": seed,
                    "criterion": criterion,
                    "target_sparsity": target,
                    "achieved_sparsity": final[4],
                    "active_weights": final[5],
                    "train_accuracy": final[2],
                    "val_accuracy": final[3],
                })

    csv_path = output / "pareto_sweep.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for criterion in criteria:
        mean_sparsity = []
        mean_accuracy = []
        std_accuracy = []

        for target in targets:
            subset = [
                r for r in rows
                if r["criterion"] == criterion
                and r["target_sparsity"] == target
            ]

            mean_sparsity.append(
                100 * np.mean([
                    r["achieved_sparsity"] for r in subset
                ])
            )

            accuracies = [
                100 * r["val_accuracy"] for r in subset
            ]

            mean_accuracy.append(np.mean(accuracies))
            std_accuracy.append(np.std(accuracies, ddof=1))

        ax.errorbar(
            mean_sparsity,
            mean_accuracy,
            yerr=std_accuracy,
            marker="o",
            capsize=4,
            label=criterion.capitalize(),
        )

    ax.set(
        xlabel="Achieved weight sparsity (%)",
        ylabel="Validation accuracy (%)",
        title="AQUA: Sparsity vs. accuracy (3 seeds)",
    )
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output / "pareto_curve.png", dpi=180)
    plt.close(fig)

    print("\nSweep complete.")
    print(f"Raw results: {csv_path}")
    print("Plot: results/pareto_curve.png")


if __name__ == "__main__":
    main()
