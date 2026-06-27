import os
import sys
import torch
import glob
from setuptools import find_packages, setup  # type: ignore

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)


def get_rwkv_extensions():
    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    # Host-compiler flags differ between MSVC (Windows) and GCC/Clang (Linux/macOS).
    # The upstream flags are GCC-style; MSVC rejects -O3/-fdiagnostics-color.
    # On Windows, CUDA 13.x's CCCL headers require MSVC's conforming preprocessor
    # (/Zc:preprocessor) under -std=c++20, both for host (.cpp) and nvcc-host (.cu).
    if sys.platform == "win32":
        cxx_flags = ["/O2", "/Zc:preprocessor", "/DPy_LIMITED_API=0x03090000"]
        nvcc_flags = ["-O3", "-Xcompiler", "/Zc:preprocessor"]
    else:
        cxx_flags = [
            "-O3",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ]
        nvcc_flags = ["-O3"]
    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    }

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, "rwkv", "model", "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            "rwkv.model.RWKV_CUDA",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=False,
        )
    ]

    return ext_modules


setup(
    name="srs-benchmark",
    packages=find_packages(),
    ext_modules=get_rwkv_extensions(),
    install_requires=[
        "torch",
        "tqdm",
        "lmdb",
        "tomli",
        "pandas",
        "pyarrow",
        "fastparquet",
        "wandb",
        "scikit-learn",
    ],
    cmdclass={"build_ext": BuildExtension},
    options={},
)
