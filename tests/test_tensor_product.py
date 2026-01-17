#!/usr/bin/env python3
"""
Test tensor product CUDA kernel against e3nn reference.

Tests both:
1. Simple tensor product (pure CG contraction, no weights)
2. Full tensor product with channels and learnable weights
"""

import torch
import numpy as np
from e3nn import o3

import _batteries


def test_simple_tensor_product():
    """Test simple tensor product (no channels, no weights) against e3nn."""
    print("\n=== Testing Simple Tensor Product ===")

    # e3nn tensor product setup for L_max=3 x L_max=3 -> L_max=3
    irreps_in = o3.Irreps("0e + 1o + 2e + 3o")  # L=0,1,2,3 with correct parities

    # Create a simple tensor product that outputs all valid L values
    # For testing CG correctness, we use FullTensorProduct
    tp = o3.FullTensorProduct(irreps_in, irreps_in)
    print(f"e3nn FullTensorProduct output: {tp.irreps_out}")

    # Test parameters
    N = 1000
    device = torch.device("cuda")

    # Random inputs
    torch.manual_seed(42)
    input1 = torch.randn(N, 16, device=device, dtype=torch.float32)
    input2 = torch.randn(N, 16, device=device, dtype=torch.float32)

    # e3nn computation
    # Note: e3nn's FullTensorProduct output is larger because it outputs all (l1,l2)->l paths
    # We need to compare against our implementation which sums all paths to the same output l

    # Our CUDA kernel
    output_cuda = _batteries.tensor_product_simple(input1, input2)

    print(f"Input1 shape: {input1.shape}")
    print(f"Input2 shape: {input2.shape}")
    print(f"Output CUDA shape: {output_cuda.shape}")

    # Build reference using manual CG application
    # For verification, compute the tensor product manually using e3nn's wigner_3j
    output_ref = torch.zeros_like(output_cuda)

    # L offsets for indexing
    def l_offset(l):
        return l * l

    for l_out in range(4):  # L_out = 0, 1, 2, 3
        for l1 in range(4):
            for l2 in range(4):
                # Triangle inequality
                if not (abs(l1 - l2) <= l_out <= l1 + l2):
                    continue

                # Get wigner 3j / CG coefficients
                w3j = o3.wigner_3j(l1, l2, l_out)
                cg = np.sqrt(2 * l_out + 1) * w3j.numpy()

                # Extract input slices
                in1_slice = input1[:, l_offset(l1):l_offset(l1) + 2*l1 + 1]
                in2_slice = input2[:, l_offset(l2):l_offset(l2) + 2*l2 + 1]

                # Apply CG contraction
                # out[n, m_out] += sum_{m1, m2} cg[m1, m2, m_out] * in1[n, m1] * in2[n, m2]
                for m1 in range(2*l1 + 1):
                    for m2 in range(2*l2 + 1):
                        for m_out in range(2*l_out + 1):
                            c = cg[m1, m2, m_out]
                            if abs(c) > 1e-10:
                                output_ref[:, l_offset(l_out) + m_out] += (
                                    c * in1_slice[:, m1] * in2_slice[:, m2]
                                )

    # Compare
    diff = torch.abs(output_cuda - output_ref)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nMax difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    # Per-L breakdown
    for l in range(4):
        start = l_offset(l)
        end = start + 2*l + 1
        l_diff = diff[:, start:end].max().item()
        print(f"  L={l}: max diff = {l_diff:.2e}")

    if max_diff < 1e-5:
        print("\n[PASS] Simple tensor product matches e3nn reference!")
        return True
    else:
        print("\n[FAIL] Simple tensor product does not match e3nn reference")
        # Debug: show first few values
        print("\nFirst sample comparison:")
        print(f"  CUDA:  {output_cuda[0, :8].cpu().numpy()}")
        print(f"  Ref:   {output_ref[0, :8].cpu().numpy()}")
        return False


def test_full_tensor_product():
    """Test full tensor product with channels and weights against e3nn."""
    print("\n=== Testing Full Tensor Product with Channels ===")

    # Test parameters
    N = 1000
    C_in = 8
    C_out = 16
    device = torch.device("cuda")

    num_paths = _batteries.get_tp_num_paths()
    print(f"Number of CG paths: {num_paths}")

    # Get path info
    path_info = _batteries.get_tp_path_info()
    print(f"Path info shape: {path_info.shape}")
    print("First 10 paths (l1, l2, l_out):")
    for i in range(min(10, num_paths)):
        print(f"  {i}: ({path_info[i, 0]}, {path_info[i, 1]}) -> {path_info[i, 2]}")

    # Random inputs
    torch.manual_seed(42)
    input1 = torch.randn(N, C_in, 16, device=device, dtype=torch.float32)
    input2 = torch.randn(N, 16, device=device, dtype=torch.float32)
    weights = torch.randn(num_paths, C_in, C_out, device=device, dtype=torch.float32)

    # Our CUDA kernel
    output_cuda = _batteries.tensor_product(input1, input2, weights)

    print(f"\nInput1 shape: {input1.shape}")
    print(f"Input2 shape: {input2.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Output CUDA shape: {output_cuda.shape}")

    # Build reference manually
    # output[n, c_out, :] = sum_{path} sum_{c_in} weight[path, c_in, c_out] *
    #                       sum_{cg} cg_coeff * input1[n, c_in, m1] * input2[n, m2]

    def l_offset(l):
        return l * l

    output_ref = torch.zeros(N, C_out, 16, device=device, dtype=torch.float32)

    path_idx = 0
    for l1 in range(4):
        for l2 in range(4):
            for l_out in range(4):
                # Triangle inequality
                if not (abs(l1 - l2) <= l_out <= l1 + l2):
                    continue

                # This is a valid path
                w3j = o3.wigner_3j(l1, l2, l_out)
                cg = np.sqrt(2 * l_out + 1) * w3j.numpy()

                offset1 = l_offset(l1)
                offset2 = l_offset(l2)
                offset_out = l_offset(l_out)

                # For each sample
                for n in range(N):
                    for c_out in range(C_out):
                        for c_in in range(C_in):
                            w = weights[path_idx, c_in, c_out].item()

                            for m1 in range(2*l1 + 1):
                                for m2 in range(2*l2 + 1):
                                    for m_out in range(2*l_out + 1):
                                        c = cg[m1, m2, m_out]
                                        if abs(c) > 1e-10:
                                            output_ref[n, c_out, offset_out + m_out] += (
                                                w * c *
                                                input1[n, c_in, offset1 + m1].item() *
                                                input2[n, offset2 + m2].item()
                                            )

                path_idx += 1

    # Compare
    diff = torch.abs(output_cuda - output_ref)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nMax difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print("\n[PASS] Full tensor product matches reference!")
        return True
    else:
        print("\n[FAIL] Full tensor product does not match reference")
        return False


def test_full_tensor_product_vectorized():
    """Vectorized test for full tensor product (faster than loop-based)."""
    print("\n=== Testing Full Tensor Product (Vectorized) ===")

    N = 1000
    C_in = 8
    C_out = 16
    device = torch.device("cuda")

    num_paths = _batteries.get_tp_num_paths()
    path_info = _batteries.get_tp_path_info().cpu().numpy()

    torch.manual_seed(42)
    input1 = torch.randn(N, C_in, 16, device=device, dtype=torch.float32)
    input2 = torch.randn(N, 16, device=device, dtype=torch.float32)
    weights = torch.randn(num_paths, C_in, C_out, device=device, dtype=torch.float32)

    # CUDA kernel
    output_cuda = _batteries.tensor_product(input1, input2, weights)

    # Vectorized reference
    def l_offset(l):
        return l * l

    output_ref = torch.zeros(N, C_out, 16, device=device, dtype=torch.float32)

    for path_idx in range(num_paths):
        l1 = int(path_info[path_idx, 0])
        l2 = int(path_info[path_idx, 1])
        l_out = int(path_info[path_idx, 2])

        # Get CG coefficients
        w3j = o3.wigner_3j(l1, l2, l_out)
        cg = torch.tensor(np.sqrt(2 * l_out + 1) * w3j.numpy(),
                         device=device, dtype=torch.float32)

        offset1 = l_offset(l1)
        offset2 = l_offset(l2)
        offset_out = l_offset(l_out)

        # Extract slices
        # in1: [N, C_in, 2*l1+1]
        # in2: [N, 2*l2+1]
        in1 = input1[:, :, offset1:offset1 + 2*l1 + 1]  # [N, C_in, 2l1+1]
        in2 = input2[:, offset2:offset2 + 2*l2 + 1]     # [N, 2l2+1]

        # Weights for this path: [C_in, C_out]
        w = weights[path_idx]  # [C_in, C_out]

        # Compute tensor product for this path
        # We need: sum_{c_in} w[c_in, c_out] * sum_{m1,m2} cg[m1,m2,m_out] * in1[n,c_in,m1] * in2[n,m2]

        # Einsum: "ijk,nm,icm,nk->njk" where i=N, c=C_in, m=2l1+1, k=2l2+1, j=2l_out+1, o=C_out
        # But we need to contract properly...

        # in1: [N, C_in, 2l1+1] -> need m1 index
        # in2: [N, 2l2+1] -> need m2 index
        # cg: [2l1+1, 2l2+1, 2l_out+1]
        # w: [C_in, C_out]

        # Step 1: tensor contraction of in1 and in2 with cg
        # prod[n, c_in, m_out] = sum_{m1, m2} cg[m1, m2, m_out] * in1[n, c_in, m1] * in2[n, m2]

        # Reshape for broadcasting
        # in1: [N, C_in, 2l1+1, 1, 1]
        # in2: [N, 1, 1, 2l2+1, 1]
        # cg: [1, 1, 2l1+1, 2l2+1, 2l_out+1]
        # Product: [N, C_in, 2l1+1, 2l2+1, 2l_out+1]
        # Sum over m1, m2: [N, C_in, 2l_out+1]

        prod = in1.unsqueeze(3).unsqueeze(4) * in2.unsqueeze(1).unsqueeze(2).unsqueeze(4) * cg.unsqueeze(0).unsqueeze(1)
        prod = prod.sum(dim=(2, 3))  # [N, C_in, 2l_out+1]

        # Step 2: weight contraction
        # out[n, c_out, m_out] = sum_{c_in} w[c_in, c_out] * prod[n, c_in, m_out]
        out_path = torch.einsum('ncm,co->nom', prod, w)  # [N, C_out, 2l_out+1]

        # Accumulate
        output_ref[:, :, offset_out:offset_out + 2*l_out + 1] += out_path

    # Compare
    diff = torch.abs(output_cuda - output_ref)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Input1 shape: {input1.shape}")
    print(f"Input2 shape: {input2.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Output CUDA shape: {output_cuda.shape}")
    print(f"\nMax difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    # Per-L breakdown
    for l in range(4):
        start = l_offset(l)
        end = start + 2*l + 1
        l_diff = diff[:, :, start:end].max().item()
        print(f"  L={l}: max diff = {l_diff:.2e}")

    if max_diff < 1e-4:
        print("\n[PASS] Full tensor product matches vectorized reference!")
        return True
    else:
        print("\n[FAIL] Full tensor product does not match vectorized reference")
        # Debug
        print("\nFirst sample, first channel comparison:")
        print(f"  CUDA: {output_cuda[0, 0, :8].cpu().numpy()}")
        print(f"  Ref:  {output_ref[0, 0, :8].cpu().numpy()}")
        return False


def test_path_info():
    """Test that path info matches expected CG structure."""
    print("\n=== Testing Path Info ===")

    num_paths = _batteries.get_tp_num_paths()
    path_info = _batteries.get_tp_path_info().cpu().numpy()

    print(f"Number of paths: {num_paths}")
    print(f"Path info shape: {path_info.shape}")

    # Count expected paths (all valid l1, l2, l_out combinations)
    expected_paths = 0
    for l1 in range(4):
        for l2 in range(4):
            for l_out in range(4):
                if abs(l1 - l2) <= l_out <= l1 + l2:
                    expected_paths += 1

    print(f"Expected paths: {expected_paths}")

    if num_paths == expected_paths:
        print("[PASS] Number of paths matches expected!")
        return True
    else:
        print(f"[FAIL] Expected {expected_paths} paths but got {num_paths}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Tensor Product CUDA Kernel Tests")
    print("=" * 60)

    results = []

    # Test path info first
    results.append(("Path Info", test_path_info()))

    # Test simple tensor product
    results.append(("Simple TP", test_simple_tensor_product()))

    # Test full tensor product (vectorized is faster)
    results.append(("Full TP (Vectorized)", test_full_tensor_product_vectorized()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[ALL TESTS PASSED]")
    else:
        print("\n[SOME TESTS FAILED]")

    exit(0 if all_passed else 1)
