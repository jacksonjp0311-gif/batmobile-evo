# Batteries

High-performance CUDA kernels for equivariant graph neural networks. Batteries provides optimized implementations of spherical harmonics, tensor products with Clebsch-Gordan coefficients, and fused message passing operations - the computational bottlenecks in models like MACE, NequIP, and Allegro. Built for L_max=3, targeting molecular dynamics and materials science workloads.

## Installation

Requires CUDA toolkit (tested with CUDA 12.x) and PyTorch 2.0+.

```bash
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## Benchmarks

Measured on RTX 3090, N_atoms=1000, C=32, ~20 neighbors/atom:

| Operation | e3nn (baseline) | Batteries | Speedup |
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

## Usage

```python
import torch
import batteries

# Spherical harmonics
edge_vectors = torch.randn(1000, 3, device="cuda")
edge_vectors = edge_vectors / edge_vectors.norm(dim=1, keepdim=True)
Y_lm = batteries.spherical_harmonics(edge_vectors, L_max=3)  # [1000, 16]

# Tensor product (simple, no weights)
node_feats = torch.randn(1000, 16, device="cuda")
output = batteries.tensor_product_simple(node_feats, Y_lm)  # [1000, 16]

# Tensor product with channels and weights
node_feats = torch.randn(1000, 32, 16, device="cuda")  # [N, C_in, 16]
weights = torch.randn(34, 32, 64, device="cuda")  # [num_paths, C_in, C_out]
output = batteries.tensor_product(node_feats, Y_lm, weights)  # [N, C_out, 16]

# Fused SH + TP (eliminates Y_lm from global memory)
source_idx = torch.randint(0, 100, (1000,), device="cuda")
messages = batteries.fused_sh_tp_simple(edge_vectors, node_feats, source_idx)
```

## Autograd Support

All operations support PyTorch autograd:

```python
from batteries.autograd import SphericalHarmonics, TensorProduct

# With autograd
edge_vectors.requires_grad = True
Y_lm = SphericalHarmonics.apply(edge_vectors, 3)
loss = Y_lm.sum()
loss.backward()  # Computes grad w.r.t. edge_vectors
```

## Reproducing Benchmarks

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
batteries/
├── include/           # CUDA headers with inline kernels
├── src/
│   ├── spherical_harmonics/
│   ├── tensor_product/
│   └── message_passing/
├── python/batteries/  # Python package
│   ├── __init__.py    # Public API
│   └── autograd.py    # torch.autograd.Function wrappers
├── benchmarks/
├── tests/
└── examples/
```

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use Batteries in your research, please cite:

```bibtex
@software{batteries2025,
  title={Batteries: High-Performance CUDA Kernels for Equivariant GNNs},
  author={...},
  year={2025},
  url={https://github.com/.../batteries}
}
```
