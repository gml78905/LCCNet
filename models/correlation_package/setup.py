#!/usr/bin/env python3
import os
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# [수정 1] PyTorch 1.10 이상은 C++14가 필수입니다.
cxx_args = ['-std=c++14']

# [수정 2] RTX 3080 (sm_86) 및 최신 GPU 아키텍처를 추가했습니다.
nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75', # T4, RTX 20번대
    '-gencode', 'arch=compute_80,code=sm_80', # A100
    '-gencode', 'arch=compute_86,code=sm_86', # RTX 3080 (현재 GPU)
    '-gencode', 'arch=compute_86,code=compute_86'
]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', [
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })