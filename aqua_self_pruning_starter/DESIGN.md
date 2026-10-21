
## AQUA Self-Pruning Network — Experimental Findings

### Architecture and training

A custom reverse-mode automatic differentiation engine was implemented using Python and NumPy. A three-layer MLP with dimensions 2 → 32 → 32 → 3 was trained on a synthetic three-class Gaussian dataset using a custom Adam optimizer and softmax cross-entropy loss.

Training used 60 epochs, mini-batches of 64, and a learning rate of 0.01.

### Pruning methodology

Two global pruning criteria were compared:

- **Magnitude:** \( |w| \)
- **First-order Taylor:** \( |w \cdot \partial L / \partial w| \), using an exponential moving average of minibatch importance scores.

A cubic schedule gradually increased sparsity between epochs 5 and 45. Pruned connections were represented by binary masks and enforced as zeros. Adam moments and optimizer ages were reset when mask states changed.

### Experimental results

Across three initialization seeds (7, 17, 27), both methods retained high final validation accuracy at 90–95% sparsity.

At 90% target sparsity, both criteria achieved approximately 98.44% mean validation accuracy, with 118 of 1,184 weight connections remaining.

At 95% target sparsity, Taylor pruning achieved approximately 98.44% mean validation accuracy, compared with 98.22% for magnitude pruning, with 59 weight connections remaining.

These results demonstrate substantial parameter redundancy on this synthetic classification task. They do not establish that Taylor pruning is universally superior.

### Computational performance

The sparse-aware forward implementation reduces counted weight multiply-accumulate operations in proportion to the number of retained connections.

Initial benchmarks showed that the Python-based sparse implementation was slower than dense NumPy despite performing fewer weight multiplications. This is consistent with overhead from Python loops, indexing, and less efficient memory access.

Final graph-free dense-versus-sparse latency measurements should be reported separately.

### Limitations

- Evaluation uses a small synthetic dataset rather than a large real-world benchmark.
- Only three initialization seeds were tested.
- Validation accuracy is quantized because the held-out set contains 150 examples.
- The pruning policy is irreversible; dynamic regrowth was not experimentally evaluated.
- The sparse inference backend is designed for correctness and operation-count analysis, not optimized production latency.
- MAC counts exclude activation functions, bias additions, memory traffic, and indexing overhead.

### Conclusion

The experiments support the claim that substantial unstructured weight sparsity can be achieved without meaningful loss of classification accuracy on this benchmark. However, reducing active parameter count is not sufficient to guarantee lower inference latency. Efficient sparse execution is a separate engineering requirement.
