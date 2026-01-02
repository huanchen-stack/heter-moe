from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name = "fp4_bench",
    ext_modules = [
        CUDAExtension(
            name = "fp4_bench._C",
            sources = [
                "cpp/bindings.cpp",
                "cpp/gemm_runner.cpp",
                "cpp/fp4_kernels.cu",    
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_100",  # Blackwell
                ],
            },
            libraries=["cublasLt", "cublas"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)