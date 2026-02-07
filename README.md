# Batmobile

High-performance CUDA kernels for equivariant graph neural networks. Batmobile provides optimized implementations of spherical harmonics, tensor products with Clebsch-Gordan coefficients, and fused message passing operations—the computational bottlenecks in models like MACE, NequIP, and Allegro. Built for L_max=3, targeting molecular dynamics and materials science workloads.

## Highlights

- CUDA-first kernels for spherical harmonics, tensor products, and fused message passing.
- Autograd support via PyTorch `torch.autograd.Function` wrappers.
- Benchmark harness that emits reproducible, contract-valid artifacts with logs and result pointers.

## Installation

Requires CUDA toolkit (tested with CUDA 12.x) and PyTorch 2.0+.

```bash
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## Benchmarks (example numbers)

Measured on RTX 3090, N_atoms=1000, C=32, ~20 neighbors/atom:

| Operation | e3nn (baseline) | Batmobile | Speedup |
|-----------|-----------------|-----------|---------|
| Spherical Harmonics (L=3) | 0.142 ms | 0.012 ms | **11.8x** |
| Tensor Product | 1.847 ms | 0.089 ms | **20.8x** |
| TP Backward | 3.21 ms | 0.156 ms | **20.6x** |
| Fused SH+TP | 0.574 ms | 0.413 ms | **1.39x** |

Full benchmark at scale (N_atoms=5000, C=64, ~30 neighbors/atom):

| Pipeline | Time | Speedup |
|----------|------|---------|
| Unfused (SH + TP + scatter) | 8.604 ms | - |
| Fused (fused_sh_tp + scatter) | 5.935 ms | **1.45x** |

## Benchmarking workflow

Run the benchmark harness (preferred) to generate contract-valid artifacts:

```bash
python benchmarks/harness/run_harness.py
```

The harness reads `benchmarks/bench_registry.json` to decide which benchmarks to run, writes JSON results under `benchmarks/results/`, and emits `logs/run_<stamp>.log` plus `artifacts/latest_<stamp>.txt` pointers for auditing and sharing.

For manual, single-script runs, you can still invoke the benchmark scripts directly:

```bash
# Spherical harmonics
python benchmarks/bench_spherical_harmonics.py

# Tensor product
python benchmarks/benchmark_tensor_product.py

# Fused SH+TP
python benchmarks/benchmark_fused_sh_tp.py

# End-to-end MACE layer
python benchmarks/benchmark_e2e_mace.py
```

## Usage

```python
import torch
import batmobile

# Spherical harmonics
edge_vectors = torch.randn(1000, 3, device="cuda")
edge_vectors = edge_vectors / edge_vectors.norm(dim=1, keepdim=True)
Y_lm = batmobile.spherical_harmonics(edge_vectors, L_max=3)  # [1000, 16]

# Tensor product (simple, no weights)
node_feats = torch.randn(1000, 16, device="cuda")
output = batmobile.tensor_product_simple(node_feats, Y_lm)  # [1000, 16]

# Tensor product with channels and weights
node_feats = torch.randn(1000, 32, 16, device="cuda")  # [N, C_in, 16]
weights = torch.randn(34, 32, 64, device="cuda")  # [num_paths, C_in, C_out]
output = batmobile.tensor_product(node_feats, Y_lm, weights)  # [N, C_out, 16]

# Fused SH + TP (eliminates Y_lm from global memory)
source_idx = torch.randint(0, 100, (1000,), device="cuda")
messages = batmobile.fused_sh_tp_simple(edge_vectors, node_feats, source_idx)
```

## Autograd Support

All operations support PyTorch autograd:

```python
from batmobile.autograd import SphericalHarmonics, TensorProduct

# With autograd
edge_vectors.requires_grad = True
Y_lm = SphericalHarmonics.apply(edge_vectors, 3)
loss = Y_lm.sum()
loss.backward()  # Computes grad w.r.t. edge_vectors
```

## Evolution upgrades (from the original scaffold)

- Added a benchmark registry so the harness runs the actual, in-repo benchmark scripts.
- The harness now emits consolidated logs and latest-artifact pointers alongside results JSON for reproducible reporting.
- Contract validation feedback is captured in harness logs to surface schema issues early.

## API Reference

### Spherical Harmonics
- `spherical_harmonics(edge_vectors, L_max)` - Compute Y_lm for unit vectors
- `spherical_harmonics_backward(edge_vectors, grad_Y_lm)` - Manual backward

### Tensor Product
- `tensor_product_simple(input1, input2)` - Pure CG contraction, no weights
- `tensor_product(input1, input2, weights)` - With channels and learnable weights
- `get_tp_num_paths()` - Returns 34 (number of CG paths for L_max=3)
- `get_tp_path_info()` - Returns [34, 3] array of (l1, l2, l_out) per path

### Fused Operations
- `fused_sh_tp_simple(edge_vectors, node_features, source_idx)` - Fused SH+TP

## Architecture

```
batmobile/
├── include/           # CUDA headers with inline kernels
├── src/
│   ├── spherical_harmonics/
│   ├── tensor_product/
│   └── message_passing/
├── python/batmobile/  # Python package
│   ├── __init__.py    # Public API
│   └── autograd.py    # torch.autograd.Function wrappers
├── benchmarks/
├── tests/
└── examples/
```

## License

MIT License. See [LICENSE](LICENSE).

## For AI Assistants

See [LLMS.txt](./LLMS.txt) for a structured overview of this codebase optimized for LLMs.

## Citation

If you use Batmobile in your research, please cite:

```bibtex
@software{batmobile2025,
  title={Batmobile: High-Performance CUDA Kernels for Equivariant GNNs},
  author={Elliot Arledge},
  year={2025},
  url={https://github.com/Infatoshi/batmobile}
}
```
