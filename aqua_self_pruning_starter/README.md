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
