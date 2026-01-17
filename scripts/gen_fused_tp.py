#!/usr/bin/env python3
"""
Generate fully unrolled fused tensor product CUDA kernel.
All CG coefficients are compile-time constants.
"""

import numpy as np
from e3nn import o3
from typing import List, Tuple

def generate_cg_operations():
    """Generate all CG operations for L_max=3."""
    L_MAX = 3
    irreps = o3.Irreps("0e + 1o + 2e + 3o")

    # Calculate offsets for each L
    offsets = [0]
    for l in range(1, L_MAX + 1):
        offsets.append(offsets[-1] + (2*(l-1) + 1))
    # offsets = [0, 1, 4, 9]

    paths = []
    for l1 in range(L_MAX + 1):
        for l2 in range(L_MAX + 1):
            for l_out in range(abs(l1 - l2), min(l1 + l2, L_MAX) + 1):
                # Get Wigner 3j symbols and convert to CG coefficients
                # CG = sqrt(2*l_out + 1) * Wigner3j
                w3j = o3.wigner_3j(l1, l2, l_out).numpy()
                cg = np.sqrt(2 * l_out + 1) * w3j

                entries = []
                for m1 in range(2*l1 + 1):
                    for m2 in range(2*l2 + 1):
                        for m_out in range(2*l_out + 1):
                            coeff = cg[m1, m2, m_out]
                            if abs(coeff) > 1e-10:
                                entries.append((m1, m2, m_out, float(coeff)))

                if entries:
                    paths.append({
                        'l1': l1, 'l2': l2, 'l_out': l_out,
                        'offset1': offsets[l1],
                        'offset2': offsets[l2],
                        'offset_out': offsets[l_out],
                        'entries': entries
                    })

    return paths

def generate_fused_header():
    """Generate the complete fused tensor product header."""
    paths = generate_cg_operations()

    lines = []
    lines.append("// Auto-generated FUSED tensor product operations")
    lines.append("// All CG paths fully unrolled at compile time for L_max=3")
    lines.append("// Generated from e3nn Wigner 3j symbols")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cuda_runtime.h>")
    lines.append("")
    lines.append("namespace batteries {")
    lines.append("namespace fused {")
    lines.append("")
    lines.append("constexpr int NUM_IRREPS = 16;       // (3+1)^2")
    lines.append(f"constexpr int NUM_PATHS = {len(paths)};")
    lines.append("")

    # Generate simple (unweighted) tensor product
    lines.append("// Simple tensor product: out += in1 x in2 (no weights)")
    lines.append("__device__ __forceinline__ void tensor_product_simple_unrolled(")
    lines.append("    const float* __restrict__ in1,")
    lines.append("    const float* __restrict__ in2,")
    lines.append("    float* __restrict__ out")
    lines.append(") {")

    for path_idx, path in enumerate(paths):
        l1, l2, l_out = path['l1'], path['l2'], path['l_out']
        off1, off2, off_out = path['offset1'], path['offset2'], path['offset_out']
        entries = path['entries']

        lines.append(f"    // Path {path_idx}: ({l1},{l2})->{l_out}, {len(entries)} entries")
        for m1, m2, m_out, coeff in entries:
            idx1 = off1 + m1
            idx2 = off2 + m2
            idx_out = off_out + m_out
            lines.append(f"    out[{idx_out}] += {coeff:.10f}f * in1[{idx1}] * in2[{idx2}];")
        lines.append("")

    lines.append("}")
    lines.append("")

    # Generate weighted tensor product for full TP
    lines.append("// Full tensor product: processes one (sample, c_out) pair")
    lines.append("// out[c_out, 16] = sum over c_in, paths of w[path,c_in,c_out] * CG * in1[c_in,m1] * in2[m2]")
    lines.append("template<int C_IN, int C_OUT>")
    lines.append("__device__ __forceinline__ void tensor_product_full_unrolled(")
    lines.append("    const float* __restrict__ in1,      // [C_IN, 16] for this sample")
    lines.append("    const float* __restrict__ in2,      // [16] for this sample")
    lines.append("    const float* __restrict__ weights,  // [NUM_PATHS, C_IN, C_OUT]")
    lines.append("    float* __restrict__ out,            // [16] output for this (sample, c_out)")
    lines.append("    int c_out")
    lines.append(") {")
    lines.append("    float acc[16] = {0};")
    lines.append("")
    lines.append("    #pragma unroll")
    lines.append("    for (int c_in = 0; c_in < C_IN; c_in++) {")
    lines.append("        const float* inp1 = in1 + c_in * 16;")
    lines.append("")

    # For each path, generate the weighted operation
    for path_idx, path in enumerate(paths):
        l1, l2, l_out = path['l1'], path['l2'], path['l_out']
        off1, off2, off_out = path['offset1'], path['offset2'], path['offset_out']
        entries = path['entries']

        lines.append(f"        // Path {path_idx}: ({l1},{l2})->{l_out}")
        lines.append(f"        {{")
        lines.append(f"            const float w = weights[{path_idx} * C_IN * C_OUT + c_in * C_OUT + c_out];")
        for m1, m2, m_out, coeff in entries:
            idx1 = off1 + m1
            idx2 = off2 + m2
            idx_out = off_out + m_out
            lines.append(f"            acc[{idx_out}] += w * {coeff:.10f}f * inp1[{idx1}] * in2[{idx2}];")
        lines.append(f"        }}")
        lines.append("")

    lines.append("    }")
    lines.append("")
    lines.append("    // Write output")
    lines.append("    #pragma unroll")
    lines.append("    for (int i = 0; i < 16; i++) {")
    lines.append("        out[i] = acc[i];")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # Generate a version with dynamic C_in/C_out
    lines.append("// Dynamic channel version (for runtime flexibility)")
    lines.append("__device__ __forceinline__ void tensor_product_full_dynamic(")
    lines.append("    const float* __restrict__ in1,      // [C_in, 16] for this sample")
    lines.append("    const float* __restrict__ in2,      // [16] for this sample")
    lines.append("    const float* __restrict__ weights,  // [NUM_PATHS, C_in, C_out]")
    lines.append("    float* __restrict__ out,            // [16] output for this (sample, c_out)")
    lines.append("    int c_in_count,")
    lines.append("    int c_out_count,")
    lines.append("    int c_out")
    lines.append(") {")
    lines.append("    float acc[16] = {0};")
    lines.append("")
    lines.append("    for (int c_in = 0; c_in < c_in_count; c_in++) {")
    lines.append("        const float* inp1 = in1 + c_in * 16;")
    lines.append("")

    for path_idx, path in enumerate(paths):
        l1, l2, l_out = path['l1'], path['l2'], path['l_out']
        off1, off2, off_out = path['offset1'], path['offset2'], path['offset_out']
        entries = path['entries']

        lines.append(f"        // Path {path_idx}: ({l1},{l2})->{l_out}")
        lines.append(f"        {{")
        lines.append(f"            const float w = weights[{path_idx} * c_in_count * c_out_count + c_in * c_out_count + c_out];")
        for m1, m2, m_out, coeff in entries:
            idx1 = off1 + m1
            idx2 = off2 + m2
            idx_out = off_out + m_out
            lines.append(f"            acc[{idx_out}] += w * {coeff:.10f}f * inp1[{idx1}] * in2[{idx2}];")
        lines.append(f"        }}")

    lines.append("    }")
    lines.append("")
    lines.append("    #pragma unroll")
    lines.append("    for (int i = 0; i < 16; i++) {")
    lines.append("        out[i] = acc[i];")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # =========================================================================
    # BACKWARD PASS FUNCTIONS
    # =========================================================================

    lines.append("// =========================================================================")
    lines.append("// BACKWARD PASS - Fused implementations")
    lines.append("// =========================================================================")
    lines.append("")

    # Simple backward (no weights)
    lines.append("// Simple tensor product backward: computes grad_in1 and grad_in2")
    lines.append("// For forward: out[m_out] += cg * in1[m1] * in2[m2]")
    lines.append("// Backward: grad_in1[m1] += cg * grad_out[m_out] * in2[m2]")
    lines.append("//           grad_in2[m2] += cg * grad_out[m_out] * in1[m1]")
    lines.append("__device__ __forceinline__ void tensor_product_simple_backward_unrolled(")
    lines.append("    const float* __restrict__ in1,")
    lines.append("    const float* __restrict__ in2,")
    lines.append("    const float* __restrict__ grad_out,")
    lines.append("    float* __restrict__ grad_in1,")
    lines.append("    float* __restrict__ grad_in2")
    lines.append(") {")

    for path_idx, path in enumerate(paths):
        l1, l2, l_out = path['l1'], path['l2'], path['l_out']
        off1, off2, off_out = path['offset1'], path['offset2'], path['offset_out']
        entries = path['entries']

        lines.append(f"    // Path {path_idx}: ({l1},{l2})->{l_out}, {len(entries)} entries")
        for m1, m2, m_out, coeff in entries:
            idx1 = off1 + m1
            idx2 = off2 + m2
            idx_out = off_out + m_out
            lines.append(f"    grad_in1[{idx1}] += {coeff:.10f}f * grad_out[{idx_out}] * in2[{idx2}];")
            lines.append(f"    grad_in2[{idx2}] += {coeff:.10f}f * grad_out[{idx_out}] * in1[{idx1}];")
        lines.append("")

    lines.append("}")
    lines.append("")

    # Full backward for grad_input1
    lines.append("// Full tensor product backward for grad_input1")
    lines.append("// Each call computes grad_in1[c_in, :] for one (sample, c_in) pair")
    lines.append("// grad_in1[c_in, m1] = sum_{c_out, paths} w[path,c_in,c_out] * cg * grad_out[c_out,m_out] * in2[m2]")
    lines.append("__device__ __forceinline__ void tensor_product_backward_input1_unrolled(")
    lines.append("    const float* __restrict__ in2,           // [16] for this sample")
    lines.append("    const float* __restrict__ weights,       // [NUM_PATHS, C_in, C_out]")
    lines.append("    const float* __restrict__ grad_out,      // [C_out, 16] for this sample")
    lines.append("    float* __restrict__ grad_in1,            // [16] output for this (sample, c_in)")
    lines.append("    int c_in,")
    lines.append("    int c_in_count,")
    lines.append("    int c_out_count")
    lines.append(") {")
    lines.append("    float acc[16] = {0};")
    lines.append("")
    lines.append("    for (int c_out = 0; c_out < c_out_count; c_out++) {")
    lines.append("        const float* go = grad_out + c_out * 16;")
    lines.append("")

    for path_idx, path in enumerate(paths):
        l1, l2, l_out = path['l1'], path['l2'], path['l_out']
        off1, off2, off_out = path['offset1'], path['offset2'], path['offset_out']
        entries = path['entries']

        lines.append(f"        // Path {path_idx}: ({l1},{l2})->{l_out}")
        lines.append(f"        {{")
        lines.append(f"            const float w = weights[{path_idx} * c_in_count * c_out_count + c_in * c_out_count + c_out];")
        for m1, m2, m_out, coeff in entries:
            idx1 = off1 + m1
            idx2 = off2 + m2
            idx_out = off_out + m_out
            lines.append(f"            acc[{idx1}] += w * {coeff:.10f}f * go[{idx_out}] * in2[{idx2}];")
        lines.append(f"        }}")

    lines.append("    }")
    lines.append("")
    lines.append("    #pragma unroll")
    lines.append("    for (int i = 0; i < 16; i++) {")
    lines.append("        grad_in1[i] = acc[i];")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # Full backward for grad_input2
    lines.append("// Full tensor product backward for grad_input2")
    lines.append("// Computes grad_in2[:] for one sample (sum over all c_in, c_out)")
    lines.append("// grad_in2[m2] = sum_{c_in, c_out, paths} w[path,c_in,c_out] * cg * grad_out[c_out,m_out] * in1[c_in,m1]")
    lines.append("__device__ __forceinline__ void tensor_product_backward_input2_unrolled(")
    lines.append("    const float* __restrict__ in1,           // [C_in, 16] for this sample")
    lines.append("    const float* __restrict__ weights,       // [NUM_PATHS, C_in, C_out]")
    lines.append("    const float* __restrict__ grad_out,      // [C_out, 16] for this sample")
    lines.append("    float* __restrict__ grad_in2,            // [16] output for this sample")
    lines.append("    int c_in_count,")
    lines.append("    int c_out_count")
    lines.append(") {")
    lines.append("    float acc[16] = {0};")
    lines.append("")
    lines.append("    for (int c_in = 0; c_in < c_in_count; c_in++) {")
    lines.append("        const float* inp1 = in1 + c_in * 16;")
    lines.append("")
    lines.append("        for (int c_out = 0; c_out < c_out_count; c_out++) {")
    lines.append("            const float* go = grad_out + c_out * 16;")
    lines.append("")

    for path_idx, path in enumerate(paths):
        l1, l2, l_out = path['l1'], path['l2'], path['l_out']
        off1, off2, off_out = path['offset1'], path['offset2'], path['offset_out']
        entries = path['entries']

        lines.append(f"            // Path {path_idx}: ({l1},{l2})->{l_out}")
        lines.append(f"            {{")
        lines.append(f"                const float w = weights[{path_idx} * c_in_count * c_out_count + c_in * c_out_count + c_out];")
        for m1, m2, m_out, coeff in entries:
            idx1 = off1 + m1
            idx2 = off2 + m2
            idx_out = off_out + m_out
            lines.append(f"                acc[{idx2}] += w * {coeff:.10f}f * go[{idx_out}] * inp1[{idx1}];")
        lines.append(f"            }}")

    lines.append("        }")
    lines.append("    }")
    lines.append("")
    lines.append("    #pragma unroll")
    lines.append("    for (int i = 0; i < 16; i++) {")
    lines.append("        grad_in2[i] = acc[i];")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # Per-pair backward for grad_input2 (for parallelized kernel)
    lines.append("// Per-pair tensor product backward for grad_input2")
    lines.append("// Processes a single (c_in, c_out) pair, accumulates into acc[16]")
    lines.append("// This enables parallelization over (c_in, c_out) pairs")
    lines.append("__device__ __forceinline__ void tensor_product_backward_input2_per_pair(")
    lines.append("    const float* __restrict__ inp1,          // [16] for this c_in")
    lines.append("    const float* __restrict__ go,            // [16] for this c_out")
    lines.append("    const float* __restrict__ weights,       // [NUM_PATHS, C_in, C_out]")
    lines.append("    float* __restrict__ acc,                 // [16] accumulator (caller's local array)")
    lines.append("    int c_in,")
    lines.append("    int c_out,")
    lines.append("    int c_in_count,")
    lines.append("    int c_out_count")
    lines.append(") {")

    for path_idx, path in enumerate(paths):
        l1, l2, l_out = path['l1'], path['l2'], path['l_out']
        off1, off2, off_out = path['offset1'], path['offset2'], path['offset_out']
        entries = path['entries']

        lines.append(f"    // Path {path_idx}: ({l1},{l2})->{l_out}")
        lines.append(f"    {{")
        lines.append(f"        const float w = weights[{path_idx} * c_in_count * c_out_count + c_in * c_out_count + c_out];")
        for m1, m2, m_out, coeff in entries:
            idx1 = off1 + m1
            idx2 = off2 + m2
            idx_out = off_out + m_out
            lines.append(f"        acc[{idx2}] += w * {coeff:.10f}f * go[{idx_out}] * inp1[{idx1}];")
        lines.append(f"    }}")

    lines.append("}")
    lines.append("")

    # Full backward for grad_weights
    lines.append("// Full tensor product backward for grad_weights")
    lines.append("// For one sample: computes contribution to grad_weights for all paths")
    lines.append("// grad_w[path,c_in,c_out] += sum_{cg} cg * in1[c_in,m1] * in2[m2] * grad_out[c_out,m_out]")
    lines.append("// Returns partial sums that need to be accumulated across samples")
    lines.append("__device__ __forceinline__ void tensor_product_backward_weights_unrolled(")
    lines.append("    const float* __restrict__ in1,           // [C_in, 16] for this sample")
    lines.append("    const float* __restrict__ in2,           // [16] for this sample")
    lines.append("    const float* __restrict__ grad_out,      // [C_out, 16] for this sample")
    lines.append("    float* __restrict__ grad_weights,        // [NUM_PATHS, C_in, C_out] - atomicAdd to this")
    lines.append("    int c_in_count,")
    lines.append("    int c_out_count")
    lines.append(") {")
    lines.append("    for (int c_in = 0; c_in < c_in_count; c_in++) {")
    lines.append("        const float* inp1 = in1 + c_in * 16;")
    lines.append("")
    lines.append("        for (int c_out = 0; c_out < c_out_count; c_out++) {")
    lines.append("            const float* go = grad_out + c_out * 16;")
    lines.append("")

    for path_idx, path in enumerate(paths):
        l1, l2, l_out = path['l1'], path['l2'], path['l_out']
        off1, off2, off_out = path['offset1'], path['offset2'], path['offset_out']
        entries = path['entries']

        lines.append(f"            // Path {path_idx}: ({l1},{l2})->{l_out}")
        lines.append(f"            {{")
        lines.append(f"                float grad_w = 0.0f;")
        for m1, m2, m_out, coeff in entries:
            idx1 = off1 + m1
            idx2 = off2 + m2
            idx_out = off_out + m_out
            lines.append(f"                grad_w += {coeff:.10f}f * inp1[{idx1}] * in2[{idx2}] * go[{idx_out}];")
        lines.append(f"                atomicAdd(&grad_weights[{path_idx} * c_in_count * c_out_count + c_in * c_out_count + c_out], grad_w);")
        lines.append(f"            }}")

    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    lines.append("} // namespace fused")
    lines.append("} // namespace batteries")

    return "\n".join(lines)

if __name__ == "__main__":
    header = generate_fused_header()

    output_path = "/home/infatoshi/cuda/batteries/include/tensor_product_fused.cuh"
    with open(output_path, "w") as f:
        f.write(header)

    print(f"Generated {output_path}")

    # Print some stats
    paths = generate_cg_operations()
    total_entries = sum(len(p['entries']) for p in paths)
    print(f"Total paths: {len(paths)}")
    print(f"Total CG entries: {total_entries}")
