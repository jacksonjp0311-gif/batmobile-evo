"""
Autograd-compatible wrappers for batteries CUDA kernels.

These wrap the fast CUDA forward passes with analytical backward implementations
to enable gradient-based computations (e.g., force = -dE/dR).
"""

import torch
from torch import Tensor
from typing import Optional, Tuple
import math

# Import the CUDA kernels
# _batteries is in the project root, not inside the python package
import _batteries


class SphericalHarmonicsFunction(torch.autograd.Function):
    """
    Autograd function for spherical harmonics.

    Forward: Uses fast CUDA kernel
    Backward: Analytical derivatives dY_lm/d(x,y,z)

    The e3nn convention uses component normalization where Y_lm are
    polynomial functions of (x, y, z) on the unit sphere.
    """

    @staticmethod
    def forward(ctx, edge_vectors: Tensor, lmax: int, normalize: bool = True) -> Tensor:
        """
        Compute spherical harmonics Y_lm for edge vectors.

        Args:
            edge_vectors: [N, 3] input vectors
            lmax: Maximum angular momentum (0 to 6)
            normalize: If True, normalize input vectors to unit length

        Returns:
            [N, (lmax+1)^2] spherical harmonics
        """
        # Store for backward
        ctx.lmax = lmax
        ctx.normalize = normalize

        if normalize:
            # Compute normalized vectors and store for backward
            norms = torch.linalg.norm(edge_vectors, dim=-1, keepdim=True)
            norms = norms.clamp(min=1e-10)
            normalized = edge_vectors / norms
            ctx.save_for_backward(normalized, norms)
            input_vectors = normalized
        else:
            ctx.save_for_backward(edge_vectors, None)
            input_vectors = edge_vectors

        # Use fast CUDA forward
        out = _batteries.spherical_harmonics(input_vectors, lmax)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], None, None]:
        """
        Compute gradient dL/d(edge_vectors) = dL/dY * dY/d(x,y,z)

        Uses fast CUDA kernel for the Jacobian computation.
        When normalize=True, we also apply the chain rule through the normalization.
        """
        lmax = ctx.lmax
        normalize = ctx.normalize
        saved = ctx.saved_tensors

        if normalize:
            vectors, norms = saved
        else:
            vectors = saved[0]
            norms = None

        # Use fast CUDA backward kernel
        grad_output = grad_output.contiguous()
        grad_input = _batteries.spherical_harmonics_backward(vectors, grad_output)

        if normalize and norms is not None:
            # Chain rule through normalization: d(v/|v|)/dv = (I - v v^T / |v|^2) / |v|
            # grad_input = J^T @ grad_input where J = (I - v v^T / |v|^2) / |v|
            x = vectors  # Already normalized, shape [N, 3]
            r = norms    # [N, 1]

            # (I - x x^T) @ g / r
            # = (g - x * (x . g)) / r
            dot = (x * grad_input).sum(dim=-1, keepdim=True)  # [N, 1]
            grad_input = (grad_input - x * dot) / r

        return grad_input, None, None


def _spherical_harmonics_backward(vectors: Tensor, grad_output: Tensor, lmax: int) -> Tensor:
    """
    Compute Jacobian dY_lm/d(x,y,z) contracted with grad_output.

    This computes sum_lm grad_output[lm] * dY_lm/d(x,y,z) for each vector.

    The implementation follows the same polynomial formulas as the forward pass,
    but computes the derivatives analytically.
    """
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]
    N = vectors.shape[0]

    # Accumulator for gradient w.r.t. (x, y, z)
    grad_x = torch.zeros(N, device=vectors.device, dtype=vectors.dtype)
    grad_y = torch.zeros(N, device=vectors.device, dtype=vectors.dtype)
    grad_z = torch.zeros(N, device=vectors.device, dtype=vectors.dtype)

    idx = 0  # Current index into grad_output

    # L=0: Y_0 = 1
    # dY_0/dx = dY_0/dy = dY_0/dz = 0
    idx += 1

    if lmax >= 1:
        # L=1: Y = sqrt(3) * (x, y, z)
        c1 = 1.7320508075688772  # sqrt(3)
        g1_0 = grad_output[:, idx]      # grad for sqrt(3)*x
        g1_1 = grad_output[:, idx + 1]  # grad for sqrt(3)*y
        g1_2 = grad_output[:, idx + 2]  # grad for sqrt(3)*z

        grad_x += c1 * g1_0
        grad_y += c1 * g1_1
        grad_z += c1 * g1_2
        idx += 3

    if lmax >= 2:
        # L=2: 5 components
        sqrt15 = 3.872983346207417
        sqrt5 = 2.23606797749979

        g2_0 = grad_output[:, idx]      # sh_2_0 = sqrt15 * x * z
        g2_1 = grad_output[:, idx + 1]  # sh_2_1 = sqrt15 * x * y
        g2_2 = grad_output[:, idx + 2]  # sh_2_2 = sqrt5 * (y^2 - 0.5*(x^2+z^2))
        g2_3 = grad_output[:, idx + 3]  # sh_2_3 = sqrt15 * y * z
        g2_4 = grad_output[:, idx + 4]  # sh_2_4 = 0.5*sqrt15 * (z^2 - x^2)

        # d(sh_2_0)/dx = sqrt15 * z, d/dy = 0, d/dz = sqrt15 * x
        grad_x += sqrt15 * z * g2_0
        grad_z += sqrt15 * x * g2_0

        # d(sh_2_1)/dx = sqrt15 * y, d/dy = sqrt15 * x, d/dz = 0
        grad_x += sqrt15 * y * g2_1
        grad_y += sqrt15 * x * g2_1

        # d(sh_2_2)/dx = sqrt5 * (-x), d/dy = sqrt5 * 2*y, d/dz = sqrt5 * (-z)
        grad_x += sqrt5 * (-x) * g2_2
        grad_y += sqrt5 * (2 * y) * g2_2
        grad_z += sqrt5 * (-z) * g2_2

        # d(sh_2_3)/dx = 0, d/dy = sqrt15 * z, d/dz = sqrt15 * y
        grad_y += sqrt15 * z * g2_3
        grad_z += sqrt15 * y * g2_3

        # d(sh_2_4)/dx = 0.5*sqrt15 * (-2x), d/dy = 0, d/dz = 0.5*sqrt15 * (2z)
        grad_x += 0.5 * sqrt15 * (-2 * x) * g2_4
        grad_z += 0.5 * sqrt15 * (2 * z) * g2_4

        idx += 5

    if lmax >= 3:
        # L=3: 7 components
        # These use recurrence relations that depend on sh_2 values
        # We compute derivatives using the chain rule

        c3_0 = 1.0801234497346432   # (1/6) * sqrt(42)
        c3_1 = 2.6457513110645907   # sqrt(7)
        c3_2 = 1.6201851746019651   # (1/8) * sqrt(168)
        c3_3 = 1.3228756555322954   # (1/2) * sqrt(7)

        sqrt15 = 3.872983346207417

        x2 = x * x
        y2 = y * y
        z2 = z * z
        x2z2 = x2 + z2

        # sh_2_0 = sqrt15 * x * z
        # sh_2_4 = 0.5 * sqrt15 * (z^2 - x^2)
        sh_2_0 = sqrt15 * x * z
        sh_2_4 = 0.5 * sqrt15 * (z2 - x2)

        g3_0 = grad_output[:, idx]      # sh_3_0 = c3_0 * (sh_2_0 * z + sh_2_4 * x)
        g3_1 = grad_output[:, idx + 1]  # sh_3_1 = c3_1 * sh_2_0 * y
        g3_2 = grad_output[:, idx + 2]  # sh_3_2 = c3_2 * (4*y^2 - x2z2) * x
        g3_3 = grad_output[:, idx + 3]  # sh_3_3 = c3_3 * y * (2*y^2 - 3*x2z2)
        g3_4 = grad_output[:, idx + 4]  # sh_3_4 = c3_2 * z * (4*y^2 - x2z2)
        g3_5 = grad_output[:, idx + 5]  # sh_3_5 = c3_1 * sh_2_4 * y
        g3_6 = grad_output[:, idx + 6]  # sh_3_6 = c3_0 * (sh_2_4 * z - sh_2_0 * x)

        # sh_3_0 = c3_0 * (sqrt15*x*z * z + 0.5*sqrt15*(z^2-x^2) * x)
        #        = c3_0 * sqrt15 * (x*z^2 + 0.5*x*z^2 - 0.5*x^3)
        #        = c3_0 * sqrt15 * (1.5*x*z^2 - 0.5*x^3)
        # d/dx = c3_0 * sqrt15 * (1.5*z^2 - 1.5*x^2) = c3_0 * sqrt15 * 1.5 * (z^2 - x^2)
        # d/dy = 0
        # d/dz = c3_0 * sqrt15 * 3*x*z
        grad_x += c3_0 * sqrt15 * 1.5 * (z2 - x2) * g3_0
        grad_z += c3_0 * sqrt15 * 3 * x * z * g3_0

        # sh_3_1 = c3_1 * sqrt15 * x * z * y
        # d/dx = c3_1 * sqrt15 * z * y
        # d/dy = c3_1 * sqrt15 * x * z
        # d/dz = c3_1 * sqrt15 * x * y
        grad_x += c3_1 * sqrt15 * z * y * g3_1
        grad_y += c3_1 * sqrt15 * x * z * g3_1
        grad_z += c3_1 * sqrt15 * x * y * g3_1

        # sh_3_2 = c3_2 * (4*y^2 - x^2 - z^2) * x
        # d/dx = c3_2 * ((4*y^2 - x^2 - z^2) + x*(-2x)) = c3_2 * (4*y^2 - 3*x^2 - z^2)
        # d/dy = c3_2 * 8*y*x
        # d/dz = c3_2 * (-2z)*x
        grad_x += c3_2 * (4 * y2 - 3 * x2 - z2) * g3_2
        grad_y += c3_2 * 8 * y * x * g3_2
        grad_z += c3_2 * (-2 * z * x) * g3_2

        # sh_3_3 = c3_3 * y * (2*y^2 - 3*(x^2+z^2))
        # d/dx = c3_3 * y * (-6*x)
        # d/dy = c3_3 * ((2*y^2 - 3*x2z2) + y*4*y) = c3_3 * (6*y^2 - 3*x2z2)
        # d/dz = c3_3 * y * (-6*z)
        grad_x += c3_3 * y * (-6 * x) * g3_3
        grad_y += c3_3 * (6 * y2 - 3 * x2z2) * g3_3
        grad_z += c3_3 * y * (-6 * z) * g3_3

        # sh_3_4 = c3_2 * z * (4*y^2 - x^2 - z^2)
        # d/dx = c3_2 * z * (-2*x)
        # d/dy = c3_2 * z * 8*y
        # d/dz = c3_2 * ((4*y^2 - x2z2) + z*(-2z)) = c3_2 * (4*y^2 - x^2 - 3*z^2)
        grad_x += c3_2 * z * (-2 * x) * g3_4
        grad_y += c3_2 * z * 8 * y * g3_4
        grad_z += c3_2 * (4 * y2 - x2 - 3 * z2) * g3_4

        # sh_3_5 = c3_1 * 0.5 * sqrt15 * (z^2 - x^2) * y
        # d/dx = c3_1 * 0.5 * sqrt15 * (-2x) * y
        # d/dy = c3_1 * 0.5 * sqrt15 * (z^2 - x^2)
        # d/dz = c3_1 * 0.5 * sqrt15 * 2*z * y
        grad_x += c3_1 * 0.5 * sqrt15 * (-2 * x) * y * g3_5
        grad_y += c3_1 * 0.5 * sqrt15 * (z2 - x2) * g3_5
        grad_z += c3_1 * 0.5 * sqrt15 * 2 * z * y * g3_5

        # sh_3_6 = c3_0 * (sh_2_4 * z - sh_2_0 * x)
        #        = c3_0 * (0.5*sqrt15*(z^2-x^2)*z - sqrt15*x*z*x)
        #        = c3_0 * sqrt15 * (0.5*z^3 - 0.5*x^2*z - x^2*z)
        #        = c3_0 * sqrt15 * (0.5*z^3 - 1.5*x^2*z)
        # d/dx = c3_0 * sqrt15 * (-3*x*z)
        # d/dy = 0
        # d/dz = c3_0 * sqrt15 * (1.5*z^2 - 1.5*x^2) = c3_0 * sqrt15 * 1.5 * (z^2 - x^2)
        grad_x += c3_0 * sqrt15 * (-3 * x * z) * g3_6
        grad_z += c3_0 * sqrt15 * 1.5 * (z2 - x2) * g3_6

        idx += 7

    # For lmax >= 4, the derivatives become quite complex due to nested recurrences.
    # We'll use PyTorch autograd to compute them via numerical diff or explicit formulas.
    # For now, we implement up to lmax=3 which covers most MACE models.

    if lmax >= 4:
        # Use a fallback: compute forward with autograd enabled, then use PyTorch's grad
        # This is slower but correct
        vectors_grad = vectors.detach().requires_grad_(True)
        sh = _spherical_harmonics_forward_pytorch(vectors_grad, lmax)

        # Only backprop through L>=4 components
        num_l3 = 16  # (3+1)^2 = 16 components for lmax=3
        sh_high = sh[:, num_l3:]
        grad_high = grad_output[:, num_l3:]

        sh_high.backward(grad_high)
        grad_xyz_high = vectors_grad.grad

        grad_x += grad_xyz_high[:, 0]
        grad_y += grad_xyz_high[:, 1]
        grad_z += grad_xyz_high[:, 2]

    return torch.stack([grad_x, grad_y, grad_z], dim=-1)


def _spherical_harmonics_forward_pytorch(vectors: Tensor, lmax: int) -> Tensor:
    """
    Pure PyTorch implementation of spherical harmonics for autograd fallback.
    This is slower than CUDA but supports arbitrary lmax with autograd.
    """
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]
    N = vectors.shape[0]

    outputs = []

    # L=0
    outputs.append(torch.ones(N, device=vectors.device, dtype=vectors.dtype))

    if lmax >= 1:
        c1 = 1.7320508075688772  # sqrt(3)
        outputs.extend([c1 * x, c1 * y, c1 * z])

    if lmax >= 2:
        sqrt15 = 3.872983346207417
        sqrt5 = 2.23606797749979
        x2 = x * x
        y2 = y * y
        z2 = z * z
        x2z2 = x2 + z2

        outputs.extend([
            sqrt15 * x * z,                    # sh_2_0
            sqrt15 * x * y,                    # sh_2_1
            sqrt5 * (y2 - 0.5 * x2z2),        # sh_2_2
            sqrt15 * y * z,                    # sh_2_3
            0.5 * sqrt15 * (z2 - x2),         # sh_2_4
        ])

    if lmax >= 3:
        c3_0 = 1.0801234497346432
        c3_1 = 2.6457513110645907
        c3_2 = 1.6201851746019651
        c3_3 = 1.3228756555322954

        x2 = x * x
        y2 = y * y
        z2 = z * z
        x2z2 = x2 + z2

        sh_2_0 = sqrt15 * x * z
        sh_2_4 = 0.5 * sqrt15 * (z2 - x2)

        outputs.extend([
            c3_0 * (sh_2_0 * z + sh_2_4 * x),   # sh_3_0
            c3_1 * sh_2_0 * y,                   # sh_3_1
            c3_2 * (4 * y2 - x2z2) * x,         # sh_3_2
            c3_3 * y * (2 * y2 - 3 * x2z2),     # sh_3_3
            c3_2 * z * (4 * y2 - x2z2),         # sh_3_4
            c3_1 * sh_2_4 * y,                   # sh_3_5
            c3_0 * (sh_2_4 * z - sh_2_0 * x),   # sh_3_6
        ])

    # For L=4,5,6 use the recurrence relations
    if lmax >= 4:
        # L=4: 9 components using L=3 intermediates
        sh_3 = outputs[9:16]  # L=3 components (indices 9-15)
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6 = sh_3

        outputs.extend([
            1.0606601717798214 * (sh_3_0 * z + sh_3_6 * x),
            0.75 * sh_3_0 * y + 0.9185586535436917 * sh_3_1 * z + 0.9185586535436917 * sh_3_5 * x,
            -0.2004459314343183 * sh_3_0 * z + 0.9819805060619656 * sh_3_1 * y + 0.7763237542601484 * sh_3_2 * z + 0.7763237542601484 * sh_3_4 * x + 0.2004459314343183 * sh_3_6 * x,
            -0.3471825374147068 * sh_3_1 * z + 1.0978875820670997 * sh_3_2 * y + 0.8964214570007952 * sh_3_3 * x + 0.3471825374147068 * sh_3_5 * x,
            -0.6943650748294136 * sh_3_2 * x + 1.1338934190276817 * sh_3_3 * y - 0.6943650748294136 * sh_3_4 * z,
            -0.3471825374147068 * sh_3_1 * x + 0.8964214570007952 * sh_3_3 * z + 1.0978875820670997 * sh_3_4 * y - 0.3471825374147068 * sh_3_5 * z,
            -0.2004459314343183 * sh_3_0 * x - 0.7763237542601484 * sh_3_2 * x + 0.7763237542601484 * sh_3_4 * z + 0.9819805060619656 * sh_3_5 * y - 0.2004459314343183 * sh_3_6 * z,
            -0.9185586535436917 * sh_3_1 * x + 0.9185586535436917 * sh_3_5 * z + 0.75 * sh_3_6 * y,
            1.0606601717798214 * (-sh_3_0 * x + sh_3_6 * z),
        ])

    if lmax >= 5:
        # L=5: 11 components using L=4 intermediates
        sh_4 = outputs[16:25]  # L=4 components
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8 = sh_4

        outputs.extend([
            1.0488088481701516 * (sh_4_0 * z + sh_4_8 * x),
            0.875 * sh_4_0 * y + 0.7395099728874521 * (sh_4_1 * z + sh_4_7 * x),
            -0.13056254903119408 * sh_4_0 * z + 0.9354143466934853 * sh_4_1 * y + 0.6614378277661476 * (sh_4_2 * z + sh_4_6 * x) + 0.13056254903119408 * sh_4_8 * x,
            -0.26691840207059045 * sh_4_1 * z + 0.9819805060619657 * sh_4_2 * y + 0.5807622846498863 * (sh_4_3 * z + sh_4_5 * x) + 0.26691840207059045 * sh_4_7 * x,
            -0.4008918628686366 * sh_4_2 * z + 1.00062322162192 * sh_4_3 * y + 0.5 * (sh_4_4 * z + sh_4_4 * x) + 0.4008918628686366 * sh_4_6 * x,
            -0.5310392846552279 * sh_4_3 * z + 0.5310392846552279 * sh_4_3 * x + 1.0 * sh_4_4 * y - 0.5310392846552279 * sh_4_5 * z + 0.5310392846552279 * sh_4_5 * x,
            -0.4008918628686366 * sh_4_2 * x - 0.5 * sh_4_4 * z + 0.5 * sh_4_4 * x + 1.00062322162192 * sh_4_5 * y - 0.4008918628686366 * sh_4_6 * z,
            -0.26691840207059045 * sh_4_1 * x - 0.5807622846498863 * sh_4_3 * z + 0.5807622846498863 * sh_4_5 * x + 0.9819805060619657 * sh_4_6 * y - 0.26691840207059045 * sh_4_7 * z,
            -0.13056254903119408 * sh_4_0 * x - 0.6614378277661476 * sh_4_2 * x + 0.6614378277661476 * sh_4_6 * z + 0.9354143466934853 * sh_4_7 * y - 0.13056254903119408 * sh_4_8 * z,
            -0.7395099728874521 * sh_4_1 * x + 0.7395099728874521 * sh_4_7 * z + 0.875 * sh_4_8 * y,
            1.0488088481701516 * (-sh_4_0 * x + sh_4_8 * z),
        ])

    if lmax >= 6:
        # L=6: 13 components using L=5 intermediates
        sh_5 = outputs[25:36]  # L=5 components
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10 = sh_5

        outputs.extend([
            1.0408329997330664 * (sh_5_0 * z + sh_5_10 * x),
            0.9166666666666666 * sh_5_0 * y + 0.6454972243679028 * (sh_5_1 * z + sh_5_9 * x),
            -0.09128709291752768 * sh_5_0 * z + 0.9078412775895168 * sh_5_1 * y + 0.5903072291871401 * (sh_5_2 * z + sh_5_8 * x) + 0.09128709291752768 * sh_5_10 * x,
            -0.18257418583505536 * sh_5_1 * z + 0.9569465626879296 * sh_5_2 * y + 0.5345224838248488 * (sh_5_3 * z + sh_5_7 * x) + 0.18257418583505536 * sh_5_9 * x,
            -0.27386127875258315 * sh_5_2 * z + 0.9819805060619657 * sh_5_3 * y + 0.47871355387816913 * (sh_5_4 * z + sh_5_6 * x) + 0.27386127875258315 * sh_5_8 * x,
            -0.36514837167011066 * sh_5_3 * z + 0.9894490507736924 * sh_5_4 * y + 0.42307689461464507 * (sh_5_5 * z + sh_5_5 * x) + 0.36514837167011066 * sh_5_7 * x,
            -0.45643546458763857 * sh_5_4 * z + 0.45643546458763857 * sh_5_4 * x + 0.9819805060619657 * sh_5_5 * y - 0.45643546458763857 * sh_5_6 * z + 0.45643546458763857 * sh_5_6 * x,
            -0.36514837167011066 * sh_5_3 * x - 0.42307689461464507 * sh_5_5 * z + 0.42307689461464507 * sh_5_5 * x + 0.9894490507736924 * sh_5_6 * y - 0.36514837167011066 * sh_5_7 * z,
            -0.27386127875258315 * sh_5_2 * x - 0.47871355387816913 * sh_5_4 * z + 0.47871355387816913 * sh_5_6 * x + 0.9819805060619657 * sh_5_7 * y - 0.27386127875258315 * sh_5_8 * z,
            -0.18257418583505536 * sh_5_1 * x - 0.5345224838248488 * sh_5_3 * z + 0.5345224838248488 * sh_5_7 * x + 0.9569465626879296 * sh_5_8 * y - 0.18257418583505536 * sh_5_9 * z,
            -0.09128709291752768 * sh_5_0 * x - 0.5903072291871401 * sh_5_2 * z + 0.5903072291871401 * sh_5_8 * x + 0.9078412775895168 * sh_5_9 * y - 0.09128709291752768 * sh_5_10 * z,
            -0.6454972243679028 * sh_5_1 * x + 0.6454972243679028 * sh_5_9 * z + 0.9166666666666666 * sh_5_10 * y,
            1.0408329997330664 * (-sh_5_0 * x + sh_5_10 * z),
        ])

    return torch.stack(outputs, dim=-1)


def spherical_harmonics(edge_vectors: Tensor, lmax: int, normalize: bool = True) -> Tensor:
    """
    Compute spherical harmonics with autograd support.

    Args:
        edge_vectors: [N, 3] input vectors
        lmax: Maximum angular momentum (0 to 6)
        normalize: If True, normalize input vectors to unit length

    Returns:
        [N, (lmax+1)^2] spherical harmonics
    """
    return SphericalHarmonicsFunction.apply(edge_vectors, lmax, normalize)


# CG coefficient data for tensor product backward
# Structure: list of (l1, l2, l_out, offset1, offset2, offset_out)
_CG_BLOCK_INFO = [
    (0, 0, 0, 0, 0, 0),
    (0, 1, 1, 0, 1, 1),
    (0, 2, 2, 0, 4, 4),
    (0, 3, 3, 0, 9, 9),
    (1, 0, 1, 1, 0, 1),
    (1, 1, 0, 1, 1, 0),
    (1, 1, 1, 1, 1, 1),
    (1, 1, 2, 1, 1, 4),
    (1, 2, 1, 1, 4, 1),
    (1, 2, 2, 1, 4, 4),
    (1, 2, 3, 1, 4, 9),
    (1, 3, 2, 1, 9, 4),
    (1, 3, 3, 1, 9, 9),
    (2, 0, 2, 4, 0, 4),
    (2, 1, 1, 4, 1, 1),
    (2, 1, 2, 4, 1, 4),
    (2, 1, 3, 4, 1, 9),
    (2, 2, 0, 4, 4, 0),
    (2, 2, 1, 4, 4, 1),
    (2, 2, 2, 4, 4, 4),
    (2, 2, 3, 4, 4, 9),
    (2, 3, 1, 4, 9, 1),
    (2, 3, 2, 4, 9, 4),
    (2, 3, 3, 4, 9, 9),
    (3, 0, 3, 9, 0, 9),
    (3, 1, 2, 9, 1, 4),
    (3, 1, 3, 9, 1, 9),
    (3, 2, 1, 9, 4, 1),
    (3, 2, 2, 9, 4, 4),
    (3, 2, 3, 9, 4, 9),
    (3, 3, 0, 9, 9, 0),
    (3, 3, 1, 9, 9, 1),
    (3, 3, 2, 9, 9, 4),
    (3, 3, 3, 9, 9, 9),
]


def _get_cg_coefficients_for_block(l1: int, l2: int, l_out: int) -> list:
    """
    Return list of (m1, m2, m_out, cg_coeff) for a given (l1, l2, l_out) block.

    Uses the Wigner 3j symbols / CG coefficients from e3nn conventions.
    This is a subset - we compute the full CG coefficients dynamically.
    """
    # Import e3nn to get the CG coefficients - this is only for the backward pass
    # which doesn't need to be as fast as the forward pass
    try:
        from e3nn import o3
        import torch

        # Get the CG matrix for this (l1, l2, l_out) combination
        # e3nn stores them as sparse tensors which we can use
        cg = o3.wigner_3j(l1, l2, l_out)  # Shape: (2*l1+1, 2*l2+1, 2*l_out+1)

        entries = []
        for m1 in range(2 * l1 + 1):
            for m2 in range(2 * l2 + 1):
                for m_out in range(2 * l_out + 1):
                    coeff = cg[m1, m2, m_out].item()
                    if abs(coeff) > 1e-10:
                        entries.append((m1, m2, m_out, coeff))
        return entries
    except ImportError:
        raise ImportError("e3nn is required for tensor product backward pass. "
                         "Install with: pip install e3nn")


# Cache for CG coefficients
_cg_cache = {}


def _get_cached_cg(l1: int, l2: int, l_out: int) -> list:
    """Get CG coefficients with caching."""
    key = (l1, l2, l_out)
    if key not in _cg_cache:
        _cg_cache[key] = _get_cg_coefficients_for_block(l1, l2, l_out)
    return _cg_cache[key]


class TensorProductSimpleFunction(torch.autograd.Function):
    """
    Autograd function for simple tensor product (no channels, no weights).

    Forward: Uses fast CUDA kernel
    Backward: Analytical gradients using CG coefficients

    output[m_out] = sum_{m1, m2} cg[m1,m2,m_out] * input1[m1] * input2[m2]

    Backward:
    dL/dinput1[m1] = sum_{m2, m_out} cg[m1,m2,m_out] * input2[m2] * dL/dout[m_out]
    dL/dinput2[m2] = sum_{m1, m_out} cg[m1,m2,m_out] * input1[m1] * dL/dout[m_out]
    """

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor) -> Tensor:
        """
        Simple tensor product without channels or weights.

        Args:
            input1: [N, 16] first input
            input2: [N, 16] second input (typically spherical harmonics)

        Returns:
            [N, 16] output
        """
        ctx.save_for_backward(input1, input2)

        # Use CUDA forward pass
        output = _batteries.tensor_product_simple(input1, input2)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients w.r.t. input1 and input2 using CUDA kernel."""
        input1, input2 = ctx.saved_tensors

        # Use fast CUDA backward kernel
        grad_output = grad_output.contiguous()
        grad_input1, grad_input2 = _batteries.tensor_product_simple_backward(
            input1, input2, grad_output
        )

        return grad_input1, grad_input2


class TensorProductFunction(torch.autograd.Function):
    """
    Autograd function for full tensor product with channels and weights.

    Forward: Uses fast CUDA kernel
    Backward: Analytical gradients

    output[n, c_out, m_out] = sum_{path} sum_{c_in} weight[path, c_in, c_out] *
                              sum_{m1, m2} cg[m1,m2,m_out] * input1[n, c_in, m1] * input2[n, m2]
    """

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weights: Tensor,
                c_in: int, c_out: int) -> Tensor:
        """
        Full tensor product with channels and weights.

        Args:
            input1: [N, C_in, 16] node features
            input2: [N, 16] edge features (spherical harmonics)
            weights: [num_paths, C_in, C_out] learnable weights
            c_in: Number of input channels
            c_out: Number of output channels

        Returns:
            [N, C_out, 16] output features
        """
        ctx.save_for_backward(input1, input2, weights)
        ctx.c_in = c_in
        ctx.c_out = c_out

        # Use CUDA forward pass (infers c_in/c_out from tensor shapes)
        output = _batteries.tensor_product(input1, input2, weights)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, Tensor, None, None]:
        """Compute gradients w.r.t. input1, input2, and weights using CUDA kernel."""
        input1, input2, weights = ctx.saved_tensors

        # Check which gradients are needed
        needs_input1_grad = ctx.needs_input_grad[0]
        needs_input2_grad = ctx.needs_input_grad[1]
        needs_weights_grad = ctx.needs_input_grad[2]

        # Use fast CUDA backward kernel
        grad_output = grad_output.contiguous()
        grad_input1, grad_input2, grad_weights = _batteries.tensor_product_backward(
            input1, input2, weights, grad_output,
            needs_input1_grad, needs_input2_grad, needs_weights_grad
        )

        return grad_input1, grad_input2, grad_weights, None, None


def tensor_product_simple(input1: Tensor, input2: Tensor) -> Tensor:
    """
    Simple tensor product with autograd support.

    Args:
        input1: [N, 16] first input
        input2: [N, 16] second input (spherical harmonics)

    Returns:
        [N, 16] output
    """
    return TensorProductSimpleFunction.apply(input1, input2)


def tensor_product(input1: Tensor, input2: Tensor, weights: Tensor,
                   c_in: int, c_out: int) -> Tensor:
    """
    Full tensor product with autograd support.

    Args:
        input1: [N, C_in, 16] node features
        input2: [N, 16] edge features (spherical harmonics)
        weights: [num_paths, C_in, C_out] learnable weights
        c_in: Number of input channels
        c_out: Number of output channels

    Returns:
        [N, C_out, 16] output features
    """
    return TensorProductFunction.apply(input1, input2, weights, c_in, c_out)


# Export
__all__ = [
    'spherical_harmonics', 'SphericalHarmonicsFunction',
    'tensor_product_simple', 'TensorProductSimpleFunction',
    'tensor_product', 'TensorProductFunction',
]
