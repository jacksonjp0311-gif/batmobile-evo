"""
Batmobile: Optimized CUDA kernels for equivariant GNNs

High-performance kernels for:
- Spherical harmonics computation (L_max up to 6)
- Tensor products with Clebsch-Gordan coefficients
- Fused SH + tensor product operations
- Message passing for graph neural networks
"""

import torch  # Must import torch first to load shared libraries
import _batteries

# Core operations
from _batteries import (
    # Spherical harmonics
    spherical_harmonics,
    spherical_harmonics_inplace,
    spherical_harmonics_backward,
    # Tensor product
    tensor_product_simple,
    tensor_product,
    tensor_product_simple_backward,
    tensor_product_backward,
    # Path info
    get_tp_path_info,
    get_tp_num_paths,
    # Fused operations
    fused_sh_tp_simple,
    # Graph utilities
    edge_index_to_csr,
    fused_message_passing_simple,
    fused_message_passing,
)

# Autograd-compatible wrappers
from .autograd import (
    SphericalHarmonicsFunction,
    TensorProductSimpleFunction,
    TensorProductFunction,
    spherical_harmonics as spherical_harmonics_autograd,
    tensor_product_simple as tensor_product_simple_autograd,
    tensor_product as tensor_product_autograd,
)

__version__ = "0.1.0"

__all__ = [
    # Core forward operations
    "spherical_harmonics",
    "spherical_harmonics_inplace",
    "tensor_product_simple",
    "tensor_product",
    "fused_sh_tp_simple",
    # Backward operations
    "spherical_harmonics_backward",
    "tensor_product_simple_backward",
    "tensor_product_backward",
    # Path info
    "get_tp_path_info",
    "get_tp_num_paths",
    # Graph utilities
    "edge_index_to_csr",
    "fused_message_passing_simple",
    "fused_message_passing",
    # Autograd classes
    "SphericalHarmonicsFunction",
    "TensorProductSimpleFunction",
    "TensorProductFunction",
]
