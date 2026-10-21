
# AQUA Self-Pruning Network

A NumPy-based implementation of a self-pruning neural network, developed for the Tredence AQUA Platform technical case study.

The project implements a neural network training stack from scratch, including reverse-mode automatic differentiation, a multilayer perceptron (MLP), Adam optimization, gradual global weight pruning, and sparse-aware inference.

It evaluates the trade-offs between model sparsity, classification accuracy, theoretical computation, and measured inference latency.

## Key Features

- **Custom automatic differentiation:** Reverse-mode autodiff supporting arithmetic operations, matrix multiplication, broadcasting, nonlinear activations, and backpropagation.
- **Neural network:** A three-layer MLP with architecture 2 → 32 → 32 → 3.
- **Optimization:** Custom Adam optimizer with mask-aware parameter updates and optimizer-state management.
- **Pruning:** Global unstructured pruning using magnitude and first-order Taylor saliency.
- **Pruning schedule:** Gradual cubic sparsity schedule with post-pruning fine-tuning.
- **Sparse inference:** CSR-like representation that skips pruned weight connections.
- **Evaluation:** Three-seed experiments, accuracy–sparsity Pareto analysis, and latency benchmarking.
- **Testing:** 32 passing automated tests, including numerical gradient checks and sparse-forward equivalence.

## Project Structure

The implementation is located in `aqua_self_pruning_starter/`.

```text
aqua-self-pruning-network/
├── README.md
└── aqua_self_pruning_starter/
    ├── engine/
    │   ├── tensor.py
    │   └── losses.py
    ├── nn/
    │   ├── layers.py
    │   └── optim.py
    ├── prune/
    │   ├── criteria.py
    │   ├── masking.py
    │   ├── schedule.py
    │   └── pruner.py
    ├── train/
    │   ├── train_mlp.py
    │   ├── train_pruned.py
    │   └── run_sweep.py
    ├── benchmark/
    │   ├── sparse_inference.py
    │   ├── benchmark_checkpoint.py
    │   └── benchmark_fair.py
    ├── tests/
    ├── results/
    ├── requirements.txt
    ├── README.md
    └── DESIGN.md
```

## Installation

Requirements: Python 3.10 or later, NumPy, Matplotlib, and pytest.

From the repository root:

```powershell
cd aqua_self_pruning_starter

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Tests

```powershell
python -m pytest -q
```

Verified local result:

```text
32 passed
```

## Train the Dense Baseline

```powershell
python -m train.train_mlp
```

The dense baseline trains for 60 epochs on a synthetic three-class Gaussian dataset.

## Train a Pruned Network

Taylor saliency at 90% target sparsity:

```powershell
python -m train.train_pruned --criterion taylor --sparsity 0.90
```

Magnitude pruning at 90% target sparsity:

```powershell
python -m train.train_pruned --criterion magnitude --sparsity 0.90
```

Training saves learning curves, metrics, and trained model checkpoints in `results/`.

## Reproduce the Experiments

```powershell
python -m train.run_sweep
```

This runs 30 configurations across:

- Sparsity targets: 0%, 50%, 75%, 90%, and 95%.
- Pruning criteria: Taylor saliency and magnitude.
- Random seeds: 7, 17, and 27.

Results are saved to:

- `results/pareto_sweep.csv`
- `results/pareto_curve.png`

## Experimental Results

Final validation accuracy averaged across three initialization seeds:

| Target sparsity | Active weights | Taylor | Magnitude |
| --------------- | -------------: | -----: | --------: |
| 0%              |          1,184 | 97.78% |    97.78% |
| 50%             |            592 | 97.78% |    98.00% |
| 75%             |            296 | 98.00% |    98.44% |
| 90%             |            118 | 98.44% |    98.44% |
| 95%             |             59 | 98.44% |    98.22% |

At 90.03% achieved sparsity, the seed-7 Taylor-pruned network retained 118 of 1,184 weight connections while matching the dense baseline's 98.67% validation accuracy.

These results are specific to the synthetic benchmark and should not be interpreted as evidence of equivalent performance on more complex datasets.

## Inference Benchmark

Run the trained-checkpoint benchmark:

```powershell
python -m benchmark.benchmark_fair --checkpoint results/taylor_s90_seed7.npz
```

Latest measured results for the trained Taylor model at 90.03% sparsity, batch size 64:

| Metric                     |     Value |
| -------------------------- | --------: |
| Dense MACs per sample      |     1,184 |
| Sparse MACs per sample     |       118 |
| Dense NumPy median latency | 0.0163 ms |
| Sparse median latency      | 0.2352 ms |
| Sparse slowdown            |   14.43× |

The sparse implementation substantially reduces counted weight multiplications but is slower in wall-clock execution because Python loops and indexing overhead outweigh arithmetic savings.

The benchmark illustrates why unstructured sparsity does not automatically deliver faster inference.

## Technical Design

See [DESIGN.md](aqua_self_pruning_starter/DESIGN.md) for the autodiff architecture, pruning equations, optimizer behavior, experimental methodology, and limitations.

## Limitations

- Experiments use synthetic Gaussian data.
- Only three initialization seeds were evaluated.
- Pruning is irreversible; weight regrowth is not implemented.
- Sparse inference is implemented in Python/NumPy rather than optimized sparse kernels.
- Reported latency depends on the local hardware and runtime environment.

## Conclusion

The project demonstrates an end-to-end, testable approach to gradually pruning a neural network while monitoring predictive accuracy and computational cost.

On the evaluated synthetic dataset, up to approximately 95% weight sparsity was achieved without a substantial reduction in final validation accuracy. However, sparse execution was slower than dense NumPy, emphasizing the importance of measuring actual deployment performance rather than relying solely on parameter or operation counts.
