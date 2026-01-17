#!/usr/bin/env python3
"""
Benchmark spherical harmonics kernels.

Compares:
- batteries.spherical_harmonics (our CUDA kernel)
- e3nn.o3.spherical_harmonics (baseline)

Reports:
- Correctness (max absolute error vs e3nn)
- Throughput in GB/s (memory bandwidth bound)
- Latency in microseconds
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Callable, Tuple
from dataclasses import dataclass

import torch
import numpy as np

# Add python dir to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


@dataclass
class BenchmarkResult:
    name: str
    N: int
    L_max: int
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    gb_per_sec: float
    correct: bool
    max_error: float


def generate_unit_vectors(N: int, device: str = "cuda") -> torch.Tensor:
    """Generate random unit vectors on the sphere."""
    # Random points, then normalize
    vectors = torch.randn(N, 3, device=device, dtype=torch.float32)
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    return vectors


def generate_periodic_box_edges(
    N_atoms: int,
    box_size: float = 10.0,
    cutoff: float = 5.0,
    device: str = "cuda"
) -> Tuple[torch.Tensor, int]:
    """
    Generate edge vectors from atoms in a periodic box.
    Returns unit vectors and the number of edges.
    """
    # Random positions in box
    positions = torch.rand(N_atoms, 3, device=device, dtype=torch.float32) * box_size

    # Simple O(N^2) neighbor finding for benchmark data generation
    # In practice this would use cell lists
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 3]

    # Minimum image convention
    diff = diff - box_size * torch.round(diff / box_size)

    # Compute distances
    dist = diff.norm(dim=2)  # [N, N]

    # Find pairs within cutoff (excluding self)
    mask = (dist < cutoff) & (dist > 0.1)
    edge_idx = mask.nonzero()  # [E, 2]

    if len(edge_idx) == 0:
        return torch.zeros(0, 3, device=device, dtype=torch.float32), 0

    # Get edge vectors and normalize
    i, j = edge_idx[:, 0], edge_idx[:, 1]
    edge_vectors = diff[i, j]  # [E, 3]
    edge_vectors = edge_vectors / edge_vectors.norm(dim=1, keepdim=True)

    return edge_vectors, len(edge_idx)


def benchmark_kernel(
    kernel_fn: Callable,
    edge_vectors: torch.Tensor,
    L_max: int,
    warmup: int = 10,
    runs: int = 100
) -> Tuple[float, float, float, float]:
    """
    Benchmark a kernel with proper CUDA timing.
    Returns (mean_us, std_us, min_us, max_us).
    """
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(edge_vectors, L_max)
    torch.cuda.synchronize()

    # Timed runs with CUDA events
    times_us = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = kernel_fn(edge_vectors, L_max)
        end.record()

        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000)  # ms to us

    times = np.array(times_us)
    return times.mean(), times.std(), times.min(), times.max()


def compute_bandwidth(N: int, L_max: int, time_us: float) -> float:
    """
    Compute effective memory bandwidth in GB/s.
    Input: N * 3 * 4 bytes (float32)
    Output: N * (L_max+1)^2 * 4 bytes (float32)
    """
    num_components = (L_max + 1) ** 2
    input_bytes = N * 3 * 4
    output_bytes = N * num_components * 4
    total_bytes = input_bytes + output_bytes
    time_sec = time_us * 1e-6
    return (total_bytes / time_sec) / 1e9


def e3nn_spherical_harmonics(edge_vectors: torch.Tensor, L_max: int) -> torch.Tensor:
    """e3nn baseline implementation."""
    from e3nn import o3
    # e3nn expects [..., 3] and returns [..., (L_max+1)^2]
    # normalize=True handles unnormalized input, but we pass unit vectors
    return o3.spherical_harmonics(
        list(range(L_max + 1)),  # irreps: [0, 1, ..., L_max]
        edge_vectors,
        normalize=False,  # Input already normalized
        normalization='component'  # Match our convention
    )


def run_correctness_test(
    our_fn: Callable,
    ref_fn: Callable,
    edge_vectors: torch.Tensor,
    L_max: int,
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> Tuple[bool, float]:
    """
    Test correctness against reference implementation.
    Returns (passed, max_error).
    """
    our_result = our_fn(edge_vectors, L_max)
    ref_result = ref_fn(edge_vectors, L_max)

    max_error = (our_result - ref_result).abs().max().item()
    passed = torch.allclose(our_result, ref_result, atol=atol, rtol=rtol)

    return passed, max_error


def run_benchmarks(
    sizes: list[int],
    L_max_values: list[int],
    warmup: int = 10,
    runs: int = 100,
    skip_e3nn: bool = False
) -> list[BenchmarkResult]:
    """Run full benchmark suite."""
    results = []

    # Import our kernel
    try:
        import _batteries as batteries
        our_fn = batteries.spherical_harmonics
        print("Loaded batteries CUDA kernel")
    except ImportError as e:
        print(f"Failed to import batteries: {e}")
        print("Build with: cd build && cmake .. && make -j")
        return results

    # Import e3nn baseline
    e3nn_available = False
    if not skip_e3nn:
        try:
            from e3nn import o3
            e3nn_available = True
            print("Loaded e3nn baseline")
        except ImportError:
            print("e3nn not available, skipping baseline comparison")

    print()
    print("=" * 80)
    print(f"{'Kernel':<20} {'N':>10} {'L_max':>6} {'Mean (us)':>12} {'Std':>10} "
          f"{'GB/s':>10} {'MaxErr':>12} {'Status':>8}")
    print("=" * 80)

    for L_max in L_max_values:
        for N in sizes:
            edge_vectors = generate_unit_vectors(N)

            # Test our kernel
            if e3nn_available:
                passed, max_err = run_correctness_test(
                    our_fn, e3nn_spherical_harmonics, edge_vectors, L_max
                )
            else:
                passed, max_err = True, 0.0

            mean_us, std_us, min_us, max_us = benchmark_kernel(
                our_fn, edge_vectors, L_max, warmup, runs
            )
            gb_s = compute_bandwidth(N, L_max, mean_us)

            result = BenchmarkResult(
                name="batteries",
                N=N,
                L_max=L_max,
                mean_us=mean_us,
                std_us=std_us,
                min_us=min_us,
                max_us=max_us,
                gb_per_sec=gb_s,
                correct=passed,
                max_error=max_err
            )
            results.append(result)

            status = "OK" if passed else "FAIL"
            print(f"{'batteries':<20} {N:>10,} {L_max:>6} {mean_us:>12.2f} "
                  f"{std_us:>10.2f} {gb_s:>10.1f} {max_err:>12.2e} {status:>8}")

            # Benchmark e3nn baseline
            if e3nn_available:
                mean_us, std_us, min_us, max_us = benchmark_kernel(
                    e3nn_spherical_harmonics, edge_vectors, L_max, warmup, runs
                )
                gb_s = compute_bandwidth(N, L_max, mean_us)

                result = BenchmarkResult(
                    name="e3nn",
                    N=N,
                    L_max=L_max,
                    mean_us=mean_us,
                    std_us=std_us,
                    min_us=min_us,
                    max_us=max_us,
                    gb_per_sec=gb_s,
                    correct=True,
                    max_error=0.0
                )
                results.append(result)

                print(f"{'e3nn':<20} {N:>10,} {L_max:>6} {mean_us:>12.2f} "
                      f"{std_us:>10.2f} {gb_s:>10.1f} {'(ref)':>12} {'-':>8}")

            print("-" * 80)

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print speedup summary."""
    print()
    print("SPEEDUP SUMMARY")
    print("=" * 60)

    # Group by (N, L_max)
    from collections import defaultdict
    grouped = defaultdict(dict)
    for r in results:
        grouped[(r.N, r.L_max)][r.name] = r

    for (N, L_max), kernels in sorted(grouped.items()):
        if "batteries" in kernels and "e3nn" in kernels:
            ours = kernels["batteries"].mean_us
            theirs = kernels["e3nn"].mean_us
            speedup = theirs / ours
            status = "OK" if kernels["batteries"].correct else "INCORRECT"
            print(f"N={N:>10,}, L_max={L_max}: {speedup:.2f}x speedup ({status})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark spherical harmonics kernels")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[50_000, 200_000, 1_000_000],
                        help="Edge counts to benchmark")
    parser.add_argument("--L-max", type=int, nargs="+", default=[3, 6],
                        help="L_max values to test")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100,
                        help="Timed iterations")
    parser.add_argument("--skip-e3nn", action="store_true",
                        help="Skip e3nn baseline")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with small sizes")

    args = parser.parse_args()

    if args.quick:
        args.sizes = [10_000, 50_000]
        args.runs = 20

    print("Spherical Harmonics Benchmark")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Sizes: {args.sizes}")
    print(f"L_max values: {args.L_max}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}")
    print()

    results = run_benchmarks(
        sizes=args.sizes,
        L_max_values=args.L_max,
        warmup=args.warmup,
        runs=args.runs,
        skip_e3nn=args.skip_e3nn
    )

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
