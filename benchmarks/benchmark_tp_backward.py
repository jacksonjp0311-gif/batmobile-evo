#!/usr/bin/env python3
"""Benchmark tensor product backward pass."""

import torch
import time
import _batteries


def benchmark_backward():
    device = torch.device("cuda")

    # MACE-like dimensions
    N = 1000       # samples
    C_in = 32      # input channels
    C_out = 32     # output channels
    num_irreps = 16  # (3+1)^2 for L_max=3
    num_paths = 34

    # Create inputs
    input1 = torch.randn(N, C_in, num_irreps, device=device)
    input2 = torch.randn(N, num_irreps, device=device)
    weights = torch.randn(num_paths, C_in, C_out, device=device)

    # Forward pass to get output shape
    output = _batteries.tensor_product(input1, input2, weights)
    grad_output = torch.randn_like(output)

    # Warmup
    for _ in range(10):
        grad_input1, grad_input2, grad_weights = _batteries.tensor_product_backward(
            input1, input2, weights, grad_output
        )
    torch.cuda.synchronize()

    # Benchmark forward
    start = time.perf_counter()
    for _ in range(100):
        out = _batteries.tensor_product(input1, input2, weights)
    torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start) / 100 * 1000  # ms

    # Benchmark full backward
    start = time.perf_counter()
    for _ in range(100):
        grad_input1, grad_input2, grad_weights = _batteries.tensor_product_backward(
            input1, input2, weights, grad_output
        )
    torch.cuda.synchronize()
    all_time = (time.perf_counter() - start) / 100 * 1000  # ms

    # Benchmark individual gradients
    start = time.perf_counter()
    for _ in range(100):
        grad_input1, _, _ = _batteries.tensor_product_backward(
            input1, input2, weights, grad_output,
            True, False, False
        )
    torch.cuda.synchronize()
    grad_input1_time = (time.perf_counter() - start) / 100 * 1000

    start = time.perf_counter()
    for _ in range(100):
        _, grad_input2, _ = _batteries.tensor_product_backward(
            input1, input2, weights, grad_output,
            False, True, False
        )
    torch.cuda.synchronize()
    grad_input2_time = (time.perf_counter() - start) / 100 * 1000

    start = time.perf_counter()
    for _ in range(100):
        _, _, grad_weights = _batteries.tensor_product_backward(
            input1, input2, weights, grad_output,
            False, False, True
        )
    torch.cuda.synchronize()
    grad_weights_time = (time.perf_counter() - start) / 100 * 1000

    print(f"N={N}, C_in={C_in}, C_out={C_out}")
    print(f"Forward:          {forward_time:.3f} ms")
    print(f"")
    print(f"Testing individual gradients:")
    print(f"  grad_input1 only: {grad_input1_time:.3f} ms ({grad_input1_time/forward_time:.1f}x forward)")
    print(f"  grad_input2 only: {grad_input2_time:.3f} ms ({grad_input2_time/forward_time:.1f}x forward)")
    print(f"  grad_weights only: {grad_weights_time:.3f} ms ({grad_weights_time/forward_time:.1f}x forward)")
    print(f"  All gradients:    {all_time:.3f} ms ({all_time/forward_time:.1f}x forward)")
    print(f"")
    print(f"Backward/Forward ratio: {all_time/forward_time:.2f}x")

    # Verify correctness by computing reference with PyTorch
    print()
    print("Verifying correctness with numerical gradients (small scale)...")

    # Use smaller tensors for numerical gradient check
    N_small = 10
    input1_small = torch.randn(N_small, C_in, num_irreps, device=device, dtype=torch.float64)
    input2_small = torch.randn(N_small, num_irreps, device=device, dtype=torch.float64)
    weights_small = torch.randn(num_paths, C_in, C_out, device=device, dtype=torch.float64)
    output_small = _batteries.tensor_product(input1_small.float(), input2_small.float(), weights_small.float()).double()
    grad_output_small = torch.randn_like(output_small)

    # Get custom backward gradients
    custom_grad_input1, custom_grad_input2, custom_grad_weights = _batteries.tensor_product_backward(
        input1_small.float(), input2_small.float(), weights_small.float(), grad_output_small.float()
    )

    # Compute numerical gradients for a few elements
    eps = 1e-4

    # Check grad_input1[0, 0, 0]
    input1_plus = input1_small.clone()
    input1_plus[0, 0, 0] += eps
    input1_minus = input1_small.clone()
    input1_minus[0, 0, 0] -= eps
    out_plus = _batteries.tensor_product(input1_plus.float(), input2_small.float(), weights_small.float()).double()
    out_minus = _batteries.tensor_product(input1_minus.float(), input2_small.float(), weights_small.float()).double()
    numerical_grad_input1_000 = ((out_plus - out_minus) * grad_output_small).sum().item() / (2 * eps)
    analytic_grad_input1_000 = custom_grad_input1[0, 0, 0].double().item()

    # Check grad_input2[0, 0]
    input2_plus = input2_small.clone()
    input2_plus[0, 0] += eps
    input2_minus = input2_small.clone()
    input2_minus[0, 0] -= eps
    out_plus = _batteries.tensor_product(input1_small.float(), input2_plus.float(), weights_small.float()).double()
    out_minus = _batteries.tensor_product(input1_small.float(), input2_minus.float(), weights_small.float()).double()
    numerical_grad_input2_00 = ((out_plus - out_minus) * grad_output_small).sum().item() / (2 * eps)
    analytic_grad_input2_00 = custom_grad_input2[0, 0].double().item()

    # Check grad_weights[0, 0, 0]
    weights_plus = weights_small.clone()
    weights_plus[0, 0, 0] += eps
    weights_minus = weights_small.clone()
    weights_minus[0, 0, 0] -= eps
    out_plus = _batteries.tensor_product(input1_small.float(), input2_small.float(), weights_plus.float()).double()
    out_minus = _batteries.tensor_product(input1_small.float(), input2_small.float(), weights_minus.float()).double()
    numerical_grad_weights_000 = ((out_plus - out_minus) * grad_output_small).sum().item() / (2 * eps)
    analytic_grad_weights_000 = custom_grad_weights[0, 0, 0].double().item()

    print("Spot check numerical vs analytic gradients:")
    print(f"  grad_input1[0,0,0]: numerical={numerical_grad_input1_000:.6f}, analytic={analytic_grad_input1_000:.6f}, rel_err={abs(numerical_grad_input1_000 - analytic_grad_input1_000) / (abs(numerical_grad_input1_000) + 1e-10):.2e}")
    print(f"  grad_input2[0,0]:   numerical={numerical_grad_input2_00:.6f}, analytic={analytic_grad_input2_00:.6f}, rel_err={abs(numerical_grad_input2_00 - analytic_grad_input2_00) / (abs(numerical_grad_input2_00) + 1e-10):.2e}")
    print(f"  grad_weights[0,0,0]: numerical={numerical_grad_weights_000:.6f}, analytic={analytic_grad_weights_000:.6f}, rel_err={abs(numerical_grad_weights_000 - analytic_grad_weights_000) / (abs(numerical_grad_weights_000) + 1e-10):.2e}")


if __name__ == "__main__":
    benchmark_backward()
