#!/usr/bin/env python3
"""
Benchmark tensor product CUDA kernel against e3nn.

Tests realistic configurations for MACE-style networks.
"""

import torch
import time
from e3nn import o3
import _batteries


def benchmark_simple_tensor_product():
    """Benchmark simple tensor product (no channels, no weights)."""
    print("\n=== Benchmarking Simple Tensor Product ===")

    # e3nn setup
    irreps_in = o3.Irreps("0e + 1o + 2e + 3o")
    tp_e3nn = o3.FullTensorProduct(irreps_in, irreps_in)
    tp_e3nn = tp_e3nn.cuda()

    device = torch.device("cuda")
    results = []

    for N in [1000, 5000, 10000, 50000, 100000]:
        torch.manual_seed(42)
        input1 = torch.randn(N, 16, device=device, dtype=torch.float32)
        input2 = torch.randn(N, 16, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(5):
            _ = _batteries.tensor_product_simple(input1, input2)
            _ = tp_e3nn(input1, input2)
        torch.cuda.synchronize()

        # Benchmark CUDA kernel
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            _ = _batteries.tensor_product_simple(input1, input2)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / 50 * 1000  # ms

        # Benchmark e3nn (fewer iterations since it's slower at large N)
        num_iters = 20 if N <= 10000 else 10
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = tp_e3nn(input1, input2)
        torch.cuda.synchronize()
        e3nn_time = (time.perf_counter() - start) / num_iters * 1000  # ms

        speedup = e3nn_time / cuda_time
        results.append((N, cuda_time, e3nn_time, speedup))
        print(f"N={N:>7}: CUDA={cuda_time:.3f}ms, e3nn={e3nn_time:.3f}ms, speedup={speedup:.1f}x")

    return results


def benchmark_full_tensor_product():
    """Benchmark full tensor product with channels and weights - CUDA only."""
    print("\n=== Benchmarking Full Tensor Product (CUDA only) ===")
    print("(e3nn FullyConnectedTensorProduct is too slow for direct comparison)")

    device = torch.device("cuda")
    num_paths = _batteries.get_tp_num_paths()

    results = []

    for N in [1000, 10000, 50000, 100000, 200000]:
        C_in = 8
        C_out = 16

        torch.manual_seed(42)
        input1 = torch.randn(N, C_in, 16, device=device, dtype=torch.float32)
        input2 = torch.randn(N, 16, device=device, dtype=torch.float32)
        weights = torch.randn(num_paths, C_in, C_out, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = _batteries.tensor_product(input1, input2, weights)
        torch.cuda.synchronize()

        # Benchmark CUDA kernel
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = _batteries.tensor_product(input1, input2, weights)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / 100 * 1000  # ms

        # Calculate throughput
        # Operations: For each sample, for each path, for each CG entry:
        #   multiply weights, multiply CG, multiply inputs, accumulate
        # Approx: N * num_paths * C_in * C_out * ~18 (avg CG entries per path) * ~4 ops
        total_ops = N * num_paths * C_in * C_out * 18 * 4
        gflops = total_ops / (cuda_time / 1000) / 1e9

        results.append((N, C_in, C_out, cuda_time, gflops))
        print(f"N={N:>7}, C_in={C_in}, C_out={C_out}: {cuda_time:.3f}ms, ~{gflops:.1f} GFLOPS")

    return results


def benchmark_varying_channels():
    """Benchmark with different channel configurations."""
    print("\n=== Benchmarking Varying Channel Configurations ===")

    device = torch.device("cuda")
    num_paths = _batteries.get_tp_num_paths()
    N = 100000  # Fixed number of samples

    configs = [
        (4, 4),    # Small
        (8, 16),   # Typical MACE L0
        (16, 32),  # Typical MACE L1
        (32, 64),  # Larger
        (64, 64),  # Square
    ]

    results = []

    for C_in, C_out in configs:
        torch.manual_seed(42)
        input1 = torch.randn(N, C_in, 16, device=device, dtype=torch.float32)
        input2 = torch.randn(N, 16, device=device, dtype=torch.float32)
        weights = torch.randn(num_paths, C_in, C_out, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = _batteries.tensor_product(input1, input2, weights)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = _batteries.tensor_product(input1, input2, weights)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / 100 * 1000  # ms

        # Calculate throughput
        total_ops = N * num_paths * C_in * C_out * 18 * 4
        gflops = total_ops / (cuda_time / 1000) / 1e9

        results.append((C_in, C_out, cuda_time, gflops))
        print(f"C_in={C_in:>2}, C_out={C_out:>2}: {cuda_time:.3f}ms, ~{gflops:.1f} GFLOPS")

    return results


def main():
    print("=" * 60)
    print("Tensor Product Kernel Benchmarks")
    print("=" * 60)

    # Get GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Number of CG paths: {_batteries.get_tp_num_paths()}")

    benchmark_simple_tensor_product()
    benchmark_full_tensor_product()
    benchmark_varying_channels()

    print("\n" + "=" * 60)
    print("Benchmarks complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
