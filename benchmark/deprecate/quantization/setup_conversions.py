"""
Setup script for compiling conversion_kernels.cu
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conversion_kernels',
    ext_modules=[
        CUDAExtension(
            name='conversion_kernels',
            sources=['conversion_kernels.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_80',  # Adjust for your GPU (sm_70, sm_75, sm_80, sm_86, sm_89, sm_90)
                    '--expt-relaxed-constexpr',
                    '-lineinfo',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
