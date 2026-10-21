# AQUA Self-Pruning Network — Implementation

This directory contains the complete NumPy implementation of the AQUA Self-Pruning Network case study.

For the full project overview, experiment results, and installation instructions, see the [main README](../README.md).

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run automated tests:

```powershell
python -m pytest -q
```

Train the dense baseline:

```powershell
python -m train.train_mlp
```

Train a Taylor-pruned network:

```powershell
python -m train.train_pruned --criterion taylor --sparsity 0.90
```

Run the complete experiment sweep:

```powershell
python -m train.run_sweep
```

Benchmark trained sparse inference:

```powershell
python -m benchmark.benchmark_fair --checkpoint results/taylor_s90_seed7.npz
```

## Documentation

- [Main README](../README.md) — Overview, setup, commands, and results.
- [DESIGN.md](DESIGN.md) — Architecture, mathematical methodology, implementation decisions, and limitations.
- `results/` — Experiment metrics, plots, and saved model checkpoints.

All commands should be executed from this director

# AQUA Self-Pruning Network — Starter

This is a learning scaffold, not a finished submission. Implement, test, and understand every component before submitting it.

## Setup (Windows + VS Code)

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## Current scope

- `engine/tensor.py`: small reverse-mode autodiff core for NumPy arrays.
- `tests/test_tensor.py`: initial tests for scalar arithmetic, broadcasting, matrix multiplication, and nonlinearities.
- Remaining work: cross-entropy, comprehensive finite-difference gradient checks, Adam, MLP, pruning, sparse inference, experiments, and `DESIGN.md`.

Scikit-learn is not used by this starter. NumPy is used for tensor operations; pytest is used for tests.
