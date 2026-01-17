#!/usr/bin/env python3
"""Benchmark fused SH+TP vs unfused (sh + tp separate kernels).

Compares:
  Unfused: sh = SH(edge_vectors) -> tp = TP(src_features, sh) -> scatter
  Fused:   tp = fused_sh_tp(edge_vectors, node_features, sources) -> scatter

The fused kernel eliminates the intermediate Y_lm tensor from global memory.
"""

import torch
import time
import _batteries


def create_edge_index(N_atoms, avg_neighbors=20):
    """Create random edge index with approximately avg_neighbors per atom."""
    edges = []
    for target in range(N_atoms):
        num_neighbors = torch.randint(1, avg_neighbors * 2, (1,)).item()
        sources = torch.randint(0, N_atoms, (num_neighbors,))
        for src in sources:
            if src != target:
                edges.append((src.item(), target))

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)

    edges = list(set(edges))  # Remove duplicates
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    return edge_index


def benchmark_unfused(node_features, edge_vectors, edge_index, num_warmup=10, num_bench=100):
    """Benchmark unfused pipeline: SH -> TP_simple -> scatter_add."""
    device = node_features.device
    N_atoms = node_features.size(0)
    N_edges = edge_vectors.size(0)
    C = node_features.size(1)

    sources = edge_index[0]
    targets = edge_index[1]

    # Warmup
    for _ in range(num_warmup):
        # 1. Compute spherical harmonics
        sh = _batteries.spherical_harmonics(edge_vectors, 3)

        # 2. Gather source features
        src_features = node_features[sources]  # [N_edges, C, 16]

        # 3. For each channel, compute tensor product
        src_flat = src_features.reshape(N_edges * C, 16)
        sh_expanded = sh.unsqueeze(1).expand(-1, C, -1).reshape(N_edges * C, 16)
        messages_flat = _batteries.tensor_product_simple(src_flat, sh_expanded)
        messages = messages_flat.reshape(N_edges, C, 16)

        # 4. Scatter add to target nodes
        output = torch.zeros(N_atoms, C, 16, device=device)
        output.index_add_(0, targets, messages)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_bench):
        sh = _batteries.spherical_harmonics(edge_vectors, 3)
        src_features = node_features[sources]
        src_flat = src_features.reshape(N_edges * C, 16)
        sh_expanded = sh.unsqueeze(1).expand(-1, C, -1).reshape(N_edges * C, 16)
        messages_flat = _batteries.tensor_product_simple(src_flat, sh_expanded)
        messages = messages_flat.reshape(N_edges, C, 16)
        output = torch.zeros(N_atoms, C, 16, device=device)
        output.index_add_(0, targets, messages)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_bench * 1000  # ms

    return elapsed


def benchmark_fused_sh_tp(node_features, edge_vectors, edge_index, num_warmup=10, num_bench=100):
    """Benchmark fused pipeline: fused_sh_tp -> scatter_add."""
    device = node_features.device
    N_atoms = node_features.size(0)
    N_edges = edge_vectors.size(0)
    C = node_features.size(1)

    sources = edge_index[0]
    targets = edge_index[1]

    # Warmup
    for _ in range(num_warmup):
        # 1. Fused SH + TP
        messages = _batteries.fused_sh_tp_simple(edge_vectors, node_features, sources)

        # 2. Scatter add to target nodes
        output = torch.zeros(N_atoms, C, 16, device=device)
        output.index_add_(0, targets, messages)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_bench):
        messages = _batteries.fused_sh_tp_simple(edge_vectors, node_features, sources)
        output = torch.zeros(N_atoms, C, 16, device=device)
        output.index_add_(0, targets, messages)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_bench * 1000  # ms

    return elapsed


def verify_correctness(node_features, edge_vectors, edge_index):
    """Verify that fused and unfused produce the same results."""
    device = node_features.device
    N_atoms = node_features.size(0)
    N_edges = edge_vectors.size(0)
    C = node_features.size(1)

    sources = edge_index[0]
    targets = edge_index[1]

    # Unfused
    sh = _batteries.spherical_harmonics(edge_vectors, 3)
    src_features = node_features[sources]
    src_flat = src_features.reshape(N_edges * C, 16)
    sh_expanded = sh.unsqueeze(1).expand(-1, C, -1).reshape(N_edges * C, 16)
    messages_unfused = _batteries.tensor_product_simple(src_flat, sh_expanded)
    messages_unfused = messages_unfused.reshape(N_edges, C, 16)

    output_unfused = torch.zeros(N_atoms, C, 16, device=device)
    output_unfused.index_add_(0, targets, messages_unfused)

    # Fused
    messages_fused = _batteries.fused_sh_tp_simple(edge_vectors, node_features, sources)

    output_fused = torch.zeros(N_atoms, C, 16, device=device)
    output_fused.index_add_(0, targets, messages_fused)

    # Compare messages (before scatter)
    msg_diff = (messages_fused - messages_unfused).abs()
    msg_max = msg_diff.max().item()
    msg_mean = msg_diff.mean().item()

    # Compare outputs (after scatter)
    out_diff = (output_fused - output_unfused).abs()
    out_max = out_diff.max().item()
    out_mean = out_diff.mean().item()

    return {
        "msg_max_diff": msg_max,
        "msg_mean_diff": msg_mean,
        "out_max_diff": out_max,
        "out_mean_diff": out_mean,
    }


def main():
    device = torch.device("cuda")

    print("Fused SH+TP Benchmark")
    print("=" * 60)
    print()
    print("Comparing:")
    print("  Unfused: sh = SH(vecs) -> tp = TP(features, sh) -> scatter")
    print("  Fused:   tp = fused_sh_tp(vecs, features, src) -> scatter")
    print()
    print("The fused kernel eliminates Y_lm from global memory.")
    print()

    # Test configurations
    configs = [
        {"N_atoms": 100, "C": 16, "avg_neighbors": 10},
        {"N_atoms": 1000, "C": 32, "avg_neighbors": 20},
        {"N_atoms": 5000, "C": 64, "avg_neighbors": 30},
    ]

    for cfg in configs:
        N_atoms = cfg["N_atoms"]
        C = cfg["C"]
        avg_neighbors = cfg["avg_neighbors"]

        print("-" * 60)
        print(f"Config: N_atoms={N_atoms}, C={C}, avg_neighbors={avg_neighbors}")

        # Create test data
        edge_index = create_edge_index(N_atoms, avg_neighbors).to(device)
        N_edges = edge_index.size(1)
        print(f"N_edges={N_edges} (actual avg={N_edges/N_atoms:.1f})")

        # Memory savings from fusing
        y_lm_bytes = N_edges * 16 * 4
        print(f"Y_lm memory eliminated: {y_lm_bytes / 1024 / 1024:.2f} MB")

        edge_vectors = torch.randn(N_edges, 3, device=device)
        edge_vectors = edge_vectors / edge_vectors.norm(dim=1, keepdim=True)
        node_features = torch.randn(N_atoms, C, 16, device=device)

        # Verify correctness first
        result = verify_correctness(node_features, edge_vectors, edge_index)
        print(f"Correctness check:")
        print(f"  Messages: max_diff={result['msg_max_diff']:.2e}, mean_diff={result['msg_mean_diff']:.2e}")
        print(f"  Output:   max_diff={result['out_max_diff']:.2e}, mean_diff={result['out_mean_diff']:.2e}")

        if result["msg_max_diff"] > 1e-4:
            print("  WARNING: Results differ significantly!")
        else:
            print("  PASSED")

        # Benchmark
        unfused_time = benchmark_unfused(node_features, edge_vectors, edge_index)
        fused_time = benchmark_fused_sh_tp(node_features, edge_vectors, edge_index)

        speedup = unfused_time / fused_time
        print(f"\nTiming (100 iterations):")
        print(f"  Unfused: {unfused_time:.3f} ms")
        print(f"  Fused:   {fused_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print()

    print("=" * 60)
    print("Done")


if __name__ == "__main__":
    main()
