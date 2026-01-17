"""
Setup script for batmobile CUDA kernels.
Uses torch.utils.cpp_extension for building.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this file
here = os.path.dirname(os.path.abspath(__file__))

# CUDA architecture for RTX 3090
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')

setup(
    name='batmobile',
    version='0.1.0',
    description='Optimized CUDA kernels for equivariant GNNs',
    ext_modules=[
        CUDAExtension(
            name='_batmobile',
            sources=[
                'python/bindings.cpp',
                'src/spherical_harmonics/spherical_harmonics.cu',
                'src/spherical_harmonics/spherical_harmonics_backward.cu',
                'src/tensor_product/tensor_product.cu',
                'src/tensor_product/tensor_product_backward.cu',
                'src/message_passing/fused_message_passing.cu',
                'src/message_passing/fused_sh_tp.cu',
            ],
            include_dirs=[
                os.path.join(here, 'include'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
    packages=['batmobile'],
    package_dir={'batmobile': 'python/batmobile'},
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0',
        'numpy>=1.20',
    ],
)
