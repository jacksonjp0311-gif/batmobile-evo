#!/usr/bin/env python3
"""
Generate Clebsch-Gordan coefficient structures for tensor products.

Uses e3nn's wigner_3j to compute the real CG coefficients for real spherical harmonics.
Outputs sparse structures optimized for GPU computation.

The CG coefficients relate:
  (u ⊗ v)_l^m = Σ_{m1,m2} C_{l1,m1,l2,m2}^{l,m} × u_{l1}^{m1} × v_{l2}^{m2}

Selection rules (enforced by wigner_3j):
  - |l1 - l2| <= l <= l1 + l2 (triangle inequality)
  - m = m1 + m2 (for complex SH), but real SH mix m values
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, NamedTuple
from dataclasses import dataclass
from e3nn import o3


@dataclass
class CGBlock:
    """A single (l1, l2, l_out) block of CG coefficients."""
    l1: int
    l2: int
    l_out: int
    # Sparse entries: (m1_idx, m2_idx, m_out_idx, coefficient)
    # m_idx is 0-indexed within the l block (0 to 2l)
    entries: List[Tuple[int, int, int, float]]

    @property
    def num_entries(self) -> int:
        return len(self.entries)


@dataclass
class CGStructure:
    """Complete CG structure for a tensor product."""
    l1_max: int
    l2_max: int
    l_out_max: int
    blocks: List[CGBlock]

    # Precomputed offsets for indexing into flattened irrep arrays
    # offset[l] = sum of (2*l' + 1) for l' < l
    offsets1: List[int]  # for input1
    offsets2: List[int]  # for input2
    offsets_out: List[int]  # for output

    @property
    def num_components1(self) -> int:
        return (self.l1_max + 1) ** 2

    @property
    def num_components2(self) -> int:
        return (self.l2_max + 1) ** 2

    @property
    def num_components_out(self) -> int:
        return (self.l_out_max + 1) ** 2

    def total_cg_entries(self) -> int:
        return sum(b.num_entries for b in self.blocks)


def compute_l_offset(l_max: int) -> List[int]:
    """Compute offset for each l value: offset[l] = number of components before l."""
    offsets = [0]
    for l in range(l_max):
        offsets.append(offsets[-1] + 2 * l + 1)
    return offsets


def generate_cg_block(l1: int, l2: int, l_out: int, threshold: float = 1e-10) -> CGBlock:
    """
    Generate CG coefficients for a single (l1, l2) -> l_out block.

    Uses e3nn's wigner_3j which gives the real CG coefficients for real spherical harmonics.
    The relationship is: C^{l,m}_{l1,m1,l2,m2} = sqrt(2*l+1) * wigner_3j(l1, l2, l, m1, m2, -m) * (-1)^m

    But e3nn's o3.wigner_3j already handles the real SH convention.
    """
    # Check triangle inequality
    if not (abs(l1 - l2) <= l_out <= l1 + l2):
        return CGBlock(l1, l2, l_out, [])

    # Get wigner 3j symbols from e3nn
    # Shape: [2*l1+1, 2*l2+1, 2*l_out+1]
    w3j = o3.wigner_3j(l1, l2, l_out)

    # Convert to CG coefficients: C = sqrt(2*l_out + 1) * w3j
    cg = np.sqrt(2 * l_out + 1) * w3j.numpy()

    entries = []
    for m1_idx in range(2 * l1 + 1):
        for m2_idx in range(2 * l2 + 1):
            for m_out_idx in range(2 * l_out + 1):
                coeff = cg[m1_idx, m2_idx, m_out_idx]
                if abs(coeff) > threshold:
                    entries.append((m1_idx, m2_idx, m_out_idx, float(coeff)))

    return CGBlock(l1, l2, l_out, entries)


def generate_cg_structure(l1_max: int, l2_max: int, l_out_max: int) -> CGStructure:
    """Generate complete CG structure for tensor product."""
    blocks = []

    for l1 in range(l1_max + 1):
        for l2 in range(l2_max + 1):
            for l_out in range(l_out_max + 1):
                # Triangle inequality
                if abs(l1 - l2) <= l_out <= l1 + l2:
                    block = generate_cg_block(l1, l2, l_out)
                    if block.num_entries > 0:
                        blocks.append(block)

    return CGStructure(
        l1_max=l1_max,
        l2_max=l2_max,
        l_out_max=l_out_max,
        blocks=blocks,
        offsets1=compute_l_offset(l1_max + 1),
        offsets2=compute_l_offset(l2_max + 1),
        offsets_out=compute_l_offset(l_out_max + 1),
    )


def export_cg_structure_for_cuda(cg: CGStructure) -> Dict:
    """
    Export CG structure in a format suitable for CUDA kernel.

    Returns a dictionary with:
    - block_info: [num_blocks, 6] array with (l1, l2, l_out, offset1, offset2, offset_out) per block
    - entry_info: [total_entries, 3] array with (m1_idx, m2_idx, m_out_idx) per entry
    - cg_coeffs: [total_entries] array of coefficient values
    - block_ranges: [num_blocks, 2] array with (start_idx, end_idx) for entries
    """
    block_info = []
    entry_info = []
    cg_coeffs = []
    block_ranges = []

    current_idx = 0
    for block in cg.blocks:
        start_idx = current_idx

        # Compute offsets for this block
        offset1 = cg.offsets1[block.l1]
        offset2 = cg.offsets2[block.l2]
        offset_out = cg.offsets_out[block.l_out]

        block_info.append([block.l1, block.l2, block.l_out, offset1, offset2, offset_out])

        for m1_idx, m2_idx, m_out_idx, coeff in block.entries:
            entry_info.append([m1_idx, m2_idx, m_out_idx])
            cg_coeffs.append(coeff)
            current_idx += 1

        block_ranges.append([start_idx, current_idx])

    return {
        'l1_max': cg.l1_max,
        'l2_max': cg.l2_max,
        'l_out_max': cg.l_out_max,
        'num_components1': cg.num_components1,
        'num_components2': cg.num_components2,
        'num_components_out': cg.num_components_out,
        'num_blocks': len(cg.blocks),
        'num_entries': len(cg_coeffs),
        'block_info': np.array(block_info, dtype=np.int32),
        'entry_info': np.array(entry_info, dtype=np.int32),
        'cg_coeffs': np.array(cg_coeffs, dtype=np.float32),
        'block_ranges': np.array(block_ranges, dtype=np.int32),
    }


def print_cg_summary(cg: CGStructure):
    """Print summary of CG structure."""
    print(f"CG Structure: L1_max={cg.l1_max}, L2_max={cg.l2_max}, L_out_max={cg.l_out_max}")
    print(f"  Components: in1={cg.num_components1}, in2={cg.num_components2}, out={cg.num_components_out}")
    print(f"  Blocks: {len(cg.blocks)}")
    print(f"  Total CG entries: {cg.total_cg_entries()}")
    print()
    print("  Block breakdown:")
    for block in cg.blocks:
        print(f"    ({block.l1}, {block.l2}) -> {block.l_out}: {block.num_entries} entries")


def generate_cuda_header(cg_data: Dict, filename: str = "cg_coefficients.cuh"):
    """Generate a C++ header with compile-time CG coefficients."""

    header = f"""// Auto-generated Clebsch-Gordan coefficients
// L1_max={cg_data['l1_max']}, L2_max={cg_data['l2_max']}, L_out_max={cg_data['l_out_max']}

#pragma once

#include <cuda_runtime.h>

namespace batteries {{
namespace cg {{

constexpr int L1_MAX = {cg_data['l1_max']};
constexpr int L2_MAX = {cg_data['l2_max']};
constexpr int L_OUT_MAX = {cg_data['l_out_max']};

constexpr int NUM_COMPONENTS1 = {cg_data['num_components1']};
constexpr int NUM_COMPONENTS2 = {cg_data['num_components2']};
constexpr int NUM_COMPONENTS_OUT = {cg_data['num_components_out']};

constexpr int NUM_BLOCKS = {cg_data['num_blocks']};
constexpr int NUM_ENTRIES = {cg_data['num_entries']};

// Block info: [l1, l2, l_out, offset1, offset2, offset_out]
__device__ __constant__ int BLOCK_INFO[NUM_BLOCKS][6] = {{
"""
    for row in cg_data['block_info']:
        header += f"    {{{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]}}},\n"
    header = header.rstrip(',\n') + "\n};\n\n"

    header += """// Block ranges: [start_idx, end_idx] into entry arrays
__device__ __constant__ int BLOCK_RANGES[NUM_BLOCKS][2] = {
"""
    for row in cg_data['block_ranges']:
        header += f"    {{{row[0]}, {row[1]}}},\n"
    header = header.rstrip(',\n') + "\n};\n\n"

    header += """// Entry info: [m1_idx, m2_idx, m_out_idx]
__device__ __constant__ int ENTRY_INFO[NUM_ENTRIES][3] = {
"""
    for row in cg_data['entry_info']:
        header += f"    {{{row[0]}, {row[1]}, {row[2]}}},\n"
    header = header.rstrip(',\n') + "\n};\n\n"

    header += """// CG coefficients (float32)
__device__ __constant__ float CG_COEFFS[NUM_ENTRIES] = {
"""
    # Format coefficients in rows of 8
    coeffs = cg_data['cg_coeffs']
    for i in range(0, len(coeffs), 8):
        chunk = coeffs[i:i+8]
        header += "    " + ", ".join(f"{c:.10f}f" for c in chunk) + ",\n"
    header = header.rstrip(',\n') + "\n};\n\n"

    # Add host-accessible path info
    header += """// Host-accessible path info (l1, l2, l_out) for each block
// Used by get_path_info_l3() since __device__ __constant__ memory cannot be accessed from host
constexpr int HOST_PATH_INFO[NUM_BLOCKS][3] = {
"""
    for row in cg_data['block_info']:
        header += f"    {{{row[0]}, {row[1]}, {row[2]}}},\n"
    header = header.rstrip(',\n') + "\n};\n\n"

    header += """} // namespace cg
} // namespace batteries
"""

    return header


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CG coefficients for tensor products")
    parser.add_argument("--l1-max", type=int, default=3, help="Max L for input 1")
    parser.add_argument("--l2-max", type=int, default=3, help="Max L for input 2")
    parser.add_argument("--l-out-max", type=int, default=3, help="Max L for output")
    parser.add_argument("--output-header", type=str, default=None, help="Output C++ header file")
    parser.add_argument("--output-npz", type=str, default=None, help="Output NPZ file")

    args = parser.parse_args()

    print(f"Generating CG coefficients for L1_max={args.l1_max}, L2_max={args.l2_max}, L_out_max={args.l_out_max}")

    cg = generate_cg_structure(args.l1_max, args.l2_max, args.l_out_max)
    print_cg_summary(cg)

    cg_data = export_cg_structure_for_cuda(cg)

    if args.output_header:
        header = generate_cuda_header(cg_data)
        with open(args.output_header, 'w') as f:
            f.write(header)
        print(f"\nWrote C++ header to {args.output_header}")

    if args.output_npz:
        np.savez(args.output_npz, **cg_data)
        print(f"Wrote NPZ to {args.output_npz}")
