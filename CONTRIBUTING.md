# Contributing to batteries

## Building from source

```bash
git clone https://github.com/Infatoshi/batteries
cd batteries
pip install -e .
```

## Running tests

```bash
python -m pytest tests/
```

## Running benchmarks

```bash
python benchmarks/benchmark_e2e_mace.py
python benchmarks/benchmark_fused_sh_tp.py
python benchmarks/bench_spherical_harmonics.py
```

## Code generation

After modifying CG coefficient generation:

```bash
python scripts/gen_fused_tp.py > include/tensor_product_fused.cuh
```

## Pull requests

- Ensure correctness tests pass against e3nn reference
- Include benchmark results for performance changes
- Update LLMS.txt if adding new modules

## Code style

- CUDA kernels: use `__device__ __forceinline__` for hot paths
- Python: follow existing patterns in `python/batteries/`
- Keep CG coefficients compile-time constant for performance
