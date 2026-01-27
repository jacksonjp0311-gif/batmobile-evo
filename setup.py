from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="batmobile",
    ext_modules=[
        CUDAExtension(
            name="batmobile._C",
            sources=["batmobile/cuda/tp_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)