#!/usr/bin/env python3
"""
Correctness tests for spherical harmonics kernel.
Tests against e3nn reference implementation.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


def e3nn_reference(edge_vectors: torch.Tensor, L_max: int) -> torch.Tensor:
    """e3nn reference implementation."""
    from e3nn import o3
    return o3.spherical_harmonics(
        list(range(L_max + 1)),
        edge_vectors,
        normalize=False,
        normalization='component'
    )


@pytest.fixture
def batteries():
    """Load batteries module."""
    import _batteries
    return _batteries


class TestSphericalHarmonics:
    """Test spherical harmonics computation."""

    @pytest.mark.parametrize("L_max", [0, 1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("N", [1, 10, 100, 1000])
    def test_random_vectors(self, batteries, L_max: int, N: int):
        """Test with random unit vectors."""
        torch.manual_seed(42)
        vectors = torch.randn(N, 3, device="cuda", dtype=torch.float32)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        our_result = batteries.spherical_harmonics(vectors, L_max)
        ref_result = e3nn_reference(vectors, L_max)

        assert our_result.shape == ref_result.shape
        assert torch.allclose(our_result, ref_result, atol=1e-5, rtol=1e-5), \
            f"Max error: {(our_result - ref_result).abs().max().item()}"

    @pytest.mark.parametrize("L_max", [3, 6])
    def test_axis_aligned(self, batteries, L_max: int):
        """Test with axis-aligned vectors (poles and equator)."""
        vectors = torch.tensor([
            [1.0, 0.0, 0.0],   # +x
            [-1.0, 0.0, 0.0],  # -x
            [0.0, 1.0, 0.0],   # +y
            [0.0, -1.0, 0.0],  # -y
            [0.0, 0.0, 1.0],   # +z (north pole)
            [0.0, 0.0, -1.0],  # -z (south pole)
        ], device="cuda", dtype=torch.float32)

        our_result = batteries.spherical_harmonics(vectors, L_max)
        ref_result = e3nn_reference(vectors, L_max)

        assert torch.allclose(our_result, ref_result, atol=1e-5, rtol=1e-5), \
            f"Max error: {(our_result - ref_result).abs().max().item()}"

    @pytest.mark.parametrize("L_max", [3, 6])
    def test_diagonal_vectors(self, batteries, L_max: int):
        """Test with diagonal vectors."""
        inv_sqrt2 = 1.0 / np.sqrt(2)
        inv_sqrt3 = 1.0 / np.sqrt(3)

        vectors = torch.tensor([
            [inv_sqrt2, inv_sqrt2, 0.0],
            [inv_sqrt2, 0.0, inv_sqrt2],
            [0.0, inv_sqrt2, inv_sqrt2],
            [inv_sqrt3, inv_sqrt3, inv_sqrt3],
            [-inv_sqrt3, inv_sqrt3, inv_sqrt3],
        ], device="cuda", dtype=torch.float32)

        our_result = batteries.spherical_harmonics(vectors, L_max)
        ref_result = e3nn_reference(vectors, L_max)

        assert torch.allclose(our_result, ref_result, atol=1e-5, rtol=1e-5), \
            f"Max error: {(our_result - ref_result).abs().max().item()}"

    @pytest.mark.parametrize("L_max", [3, 6])
    def test_near_poles(self, batteries, L_max: int):
        """Test vectors near poles (edge case for phi computation)."""
        eps = 1e-7
        vectors = torch.tensor([
            [eps, 0.0, 1.0],
            [0.0, eps, 1.0],
            [eps, eps, 1.0],
            [eps, 0.0, -1.0],
        ], device="cuda", dtype=torch.float32)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        our_result = batteries.spherical_harmonics(vectors, L_max)
        ref_result = e3nn_reference(vectors, L_max)

        # Near poles, some tolerance needed due to phi ambiguity
        assert torch.allclose(our_result, ref_result, atol=1e-4, rtol=1e-4), \
            f"Max error: {(our_result - ref_result).abs().max().item()}"

    def test_output_shape(self, batteries):
        """Test output tensor shapes."""
        N = 100
        vectors = torch.randn(N, 3, device="cuda", dtype=torch.float32)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        for L_max in range(7):
            result = batteries.spherical_harmonics(vectors, L_max)
            expected_shape = (N, (L_max + 1) ** 2)
            assert result.shape == expected_shape, \
                f"L_max={L_max}: expected {expected_shape}, got {result.shape}"

    def test_empty_input(self, batteries):
        """Test with empty input tensor."""
        vectors = torch.empty(0, 3, device="cuda", dtype=torch.float32)
        result = batteries.spherical_harmonics(vectors, 3)
        assert result.shape == (0, 16)

    def test_inplace_version(self, batteries):
        """Test in-place computation."""
        N = 100
        L_max = 3
        vectors = torch.randn(N, 3, device="cuda", dtype=torch.float32)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        # Pre-allocate output
        output = torch.empty(N, 16, device="cuda", dtype=torch.float32)
        batteries.spherical_harmonics_inplace(vectors, output, L_max)

        # Compare with regular version
        expected = batteries.spherical_harmonics(vectors, L_max)
        assert torch.allclose(output, expected)

    @pytest.mark.parametrize("L_max", [3, 6])
    def test_large_batch(self, batteries, L_max: int):
        """Test with large batch size."""
        N = 100_000
        torch.manual_seed(123)
        vectors = torch.randn(N, 3, device="cuda", dtype=torch.float32)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        our_result = batteries.spherical_harmonics(vectors, L_max)
        ref_result = e3nn_reference(vectors, L_max)

        # Sample check for large batches
        assert torch.allclose(our_result, ref_result, atol=1e-5, rtol=1e-5), \
            f"Max error: {(our_result - ref_result).abs().max().item()}"


class TestSphericalHarmonicsProperties:
    """Test mathematical properties of spherical harmonics."""

    def test_l0_is_constant(self, batteries):
        """Y_0^0 should be constant (1/sqrt(4*pi))."""
        N = 100
        vectors = torch.randn(N, 3, device="cuda", dtype=torch.float32)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        result = batteries.spherical_harmonics(vectors, 0)

        # Y_0^0 = 1/sqrt(4*pi) with component normalization = 1
        # e3nn uses component normalization where Y_0^0 = 1
        expected = torch.ones(N, 1, device="cuda", dtype=torch.float32)

        # Check variance is near zero
        assert result.std().item() < 1e-5, "Y_0^0 should be constant"

    def test_orthonormality(self, batteries):
        """Spherical harmonics should be orthonormal on the sphere."""
        # Generate uniform points on sphere using Fibonacci lattice
        N = 5000
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = torch.arange(N, dtype=torch.float32)

        phi = 2 * np.pi * indices / golden_ratio
        z = 1 - 2 * (indices + 0.5) / N

        x = torch.sqrt(1 - z**2) * torch.cos(phi)
        y = torch.sqrt(1 - z**2) * torch.sin(phi)

        vectors = torch.stack([x, y, z], dim=1).cuda()

        L_max = 3
        Y = batteries.spherical_harmonics(vectors, L_max)

        # Approximate integral by sum (uniform sampling)
        # <Y_l^m, Y_l'^m'> should be ~delta_{ll'} delta_{mm'}
        gram = Y.T @ Y / N * 4 * np.pi

        # Should be approximately identity
        identity = torch.eye(16, device="cuda", dtype=torch.float32)
        max_off_diag = (gram - identity).abs().max().item()

        # Allow some tolerance for finite sampling
        assert max_off_diag < 0.1, f"Orthonormality violated: max error = {max_off_diag}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
