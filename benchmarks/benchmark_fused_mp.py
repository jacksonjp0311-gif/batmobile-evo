#!/usr/bin/env python3
"""Benchmark fused message passing vs unfused (separate kernels)."""

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


def benchmark_unfused_simple(node_features, edge_vectors, edge_index, num_warmup=10, num_bench=100):
    """Benchmark unfused pipeline: SH -> TP_simple -> scatter_add (no weights)."""
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
        # Reshape for batched processing
        src_flat = src_features.reshape(N_edges * C, 16)  # [N_edges*C, 16]
        sh_expanded = sh.unsqueeze(1).expand(-1, C, -1).reshape(N_edges * C, 16)  # [N_edges*C, 16]
        edge_output_flat = _batteries.tensor_product_simple(src_flat, sh_expanded)  # [N_edges*C, 16]
        edge_output = edge_output_flat.reshape(N_edges, C, 16)

        # 4. Scatter add to target nodes
        node_output = torch.zeros(N_atoms, C, 16, device=device)
        node_output.index_add_(0, targets, edge_output)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_bench):
        sh = _batteries.spherical_harmonics(edge_vectors, 3)
        src_features = node_features[sources]
        src_flat = src_features.reshape(N_edges * C, 16)
        sh_expanded = sh.unsqueeze(1).expand(-1, C, -1).reshape(N_edges * C, 16)
        edge_output_flat = _batteries.tensor_product_simple(src_flat, sh_expanded)
        edge_output = edge_output_flat.reshape(N_edges, C, 16)
        node_output = torch.zeros(N_atoms, C, 16, device=device)
        node_output.index_add_(0, targets, edge_output)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_bench * 1000  # ms

    return elapsed


def benchmark_fused(node_features, edge_vectors, row_ptr, col_idx, weights, num_warmup=10, num_bench=100):
    """Benchmark fused message passing kernel."""
    # Warmup
    for _ in range(num_warmup):
        output = _batteries.fused_message_passing(
            node_features, edge_vectors, weights, row_ptr, col_idx
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_bench):
        output = _batteries.fused_message_passing(
            node_features, edge_vectors, weights, row_ptr, col_idx
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_bench * 1000  # ms

    return elapsed


def benchmark_fused_simple(node_features, edge_vectors, edge_index, row_ptr, col_idx, num_warmup=10, num_bench=100):
    """Benchmark fused message passing kernel (simple version, no weights)."""
    # Warmup
    for _ in range(num_warmup):
        output = _batteries.fused_message_passing_simple(
            node_features, edge_vectors, edge_index, row_ptr, col_idx
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_bench):
        output = _batteries.fused_message_passing_simple(
            node_features, edge_vectors, edge_index, row_ptr, col_idx
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_bench * 1000  # ms

    return elapsed


def main():
    device = torch.device("cuda")

    # MACE-like dimensions
    N_atoms = 1000
    C_in = 32
    C_out = 32
    avg_neighbors = 20  # ~20 neighbors per atom

    print(f"Fused Message Passing Benchmark")
    print(f"================================")
    print(f"N_atoms={N_atoms}, C_in={C_in}, C_out={C_out}, avg_neighbors={avg_neighbors}")
    print()

    # Create test data
    edge_index = create_edge_index(N_atoms, avg_neighbors).to(device)
    N_edges = edge_index.size(1)
    print(f"N_edges={N_edges} (avg neighbors={N_edges/N_atoms:.1f})")

    # Create random edge vectors (unit vectors)
    edge_vectors = torch.randn(N_edges, 3, device=device)
    edge_vectors = edge_vectors / edge_vectors.norm(dim=1, keepdim=True)

    # Create node features
    node_features = torch.randn(N_atoms, C_in, 16, device=device)

    # Create weights
    weights = torch.randn(C_in, C_out, device=device) / (C_in ** 0.5)

    # Convert to CSR format for fused kernel
    print("\nConverting edge_index to CSR format...")
    row_ptr, col_idx = _batteries.edge_index_to_csr(edge_index, N_atoms)
    print(f"row_ptr shape: {row_ptr.shape}, col_idx shape: {col_idx.shape}")

    # Verify CSR conversion
    edge_counts = row_ptr[1:] - row_ptr[:-1]
    print(f"Edges per atom: min={edge_counts.min().item()}, max={edge_counts.max().item()}, mean={edge_counts.float().mean().item():.1f}")

    print("\n" + "=" * 60)
    print("Benchmarking Simple Message Passing (no weights)")
    print("=" * 60)

    # Benchmark simple fused
    print("\nBenchmarking fused_simple message passing...")
    try:
        fused_simple_time = benchmark_fused_simple(node_features, edge_vectors, edge_index, row_ptr, col_idx)
        print(f"Fused simple time: {fused_simple_time:.3f} ms")
    except Exception as e:
        print(f"Fused simple failed: {e}")
        import traceback
        traceback.print_exc()
        fused_simple_time = None

    # Benchmark unfused for comparison
    print("\nBenchmarking unfused (SH -> TP -> scatter_add)...")
    try:
        unfused_time = benchmark_unfused_simple(node_features, edge_vectors, edge_index)
        print(f"Unfused time: {unfused_time:.3f} ms")
    except Exception as e:
        print(f"Unfused failed: {e}")
        unfused_time = None

    if unfused_time and fused_simple_time:
        speedup = unfused_time / fused_simple_time
        print(f"\nSpeedup (fused vs unfused): {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("Correctness Verification")
    print("=" * 60)

    # Quick correctness check - compare outputs
    print("\nVerifying correctness with small inputs...")
    small_N = 10
    small_edges = create_edge_index(small_N, 5).to(device)
    small_N_edges = small_edges.size(1)
    small_vectors = torch.randn(small_N_edges, 3, device=device)
    small_vectors = small_vectors / small_vectors.norm(dim=1, keepdim=True)
    small_features = torch.randn(small_N, 4, 16, device=device)  # C=4 for simplicity

    small_row_ptr, small_col_idx = _batteries.edge_index_to_csr(small_edges, small_N)

    # Compute with fused simple
    try:
        fused_out = _batteries.fused_message_passing_simple(
            small_features, small_vectors, small_edges, small_row_ptr, small_col_idx
        )

        # Compute reference (unfused)
        sources = small_edges[0]
        targets = small_edges[1]
        sh = _batteries.spherical_harmonics(small_vectors, 3)
        src_feat = small_features[sources]  # [N_edges, C, 16]

        # Manual TP and scatter
        ref_out = torch.zeros_like(small_features)
        for e in range(small_N_edges):
            src_f = src_feat[e]  # [C, 16]
            sh_e = sh[e]  # [16]
            target = targets[e].item()

            # For each channel, compute simple TP
            for c in range(src_f.size(0)):
                tp_out = _batteries.tensor_product_simple(
                    src_f[c:c+1],  # [1, 16]
                    sh_e.unsqueeze(0)  # [1, 16]
                )
                ref_out[target, c] += tp_out[0]

        # Compare
        diff = (fused_out - ref_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"Max abs diff: {max_diff:.6e}")
        print(f"Mean abs diff: {mean_diff:.6e}")

        if max_diff < 1e-4:
            print("PASSED: Outputs match within tolerance")
        else:
            print("WARNING: Outputs differ significantly")
    except Exception as e:
        print(f"Correctness check failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
