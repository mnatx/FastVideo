import os
import subprocess

from config_vsa import kernels, sources, target
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

target = target.lower()

# Package metadata
PACKAGE_NAME = "vsa"
VERSION = "0.0.3"
AUTHOR = "Hao AI Lab"
DESCRIPTION = "Video Sparse Attention Kernel Used in FastVideo"
URL = "https://github.com/hao-ai-lab/FastVideo/tree/main/csrc/attn/video_sparse_attn"

# Set environment variables
tk_root = os.getenv('THUNDERKITTENS_ROOT', os.path.abspath(os.path.join(os.getcwd(), 'tk/')))
python_include = subprocess.check_output(['python', '-c',
                                          "import sysconfig; print(sysconfig.get_path('include'))"]).decode().strip()
torch_include = subprocess.check_output([
    'python', '-c',
    "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))"
]).decode().strip()
print('vsa root:', tk_root)
print('Python include:', python_include)
print('Torch include directories:', torch_include)

# CUDA flags
cuda_flags = [
    '-DNDEBUG', '-Xcompiler=-Wno-psabi', '-Xcompiler=-fno-strict-aliasing', '--expt-extended-lambda',
    '--expt-relaxed-constexpr', '-forward-unknown-to-host-compiler', '--use_fast_math', '-std=c++20', '-O3',
    '-Xnvlink=--verbose', '-Xptxas=--verbose', '-Xptxas=--warn-on-spills', f'-I{tk_root}/include',
    f'-I{tk_root}/prototype', f'-I{python_include}', '-DTORCH_COMPILE'
] + torch_include.split()
cpp_flags = ['-std=c++20', '-O3']

if target == 'h100':
    cuda_flags.append('-DKITTENS_HOPPER')
    cuda_flags.append('-arch=sm_90a')
elif target == 'rocm':
    # ROCm-specific flags
    cuda_flags.append('-DROCM_BUILD')
    cuda_flags.append('--offload-arch=gfx90a')  # Default to MI200 series
    # Remove CUDA-specific flags for ROCm
    cuda_flags = [flag for flag in cuda_flags if not flag.startswith('-arch=sm_')]
    # For ROCm, we primarily rely on Triton, so we can skip CUDA kernel compilation
    if not os.getenv('VSA_FORCE_CUDA_KERNELS', ''):
        print("ROCm target detected - using Triton implementation only")
        ext_modules = []  # Skip CUDA kernel compilation for ROCm
else:
    raise ValueError(f'Target {target} not supported')

source_files = ['vsa.cpp']
for k in kernels:
    if target not in sources[k]['source_files']:
        raise KeyError(f'Target {target} not found in source files for kernel {k}')
    if isinstance(sources[k]['source_files'][target], list):
        source_files.extend(sources[k]['source_files'][target])
    else:
        source_files.append(sources[k]['source_files'][target])
    cpp_flags.append(f'-DTK_COMPILE_{k.replace(" ", "_").upper()}')


# Only build CUDA extensions if not targeting ROCm or if forced
if target != 'rocm' or os.getenv('VSA_FORCE_CUDA_KERNELS', ''):
    ext_modules = [
        CUDAExtension('vsa_cuda',
            sources=source_files,
            extra_compile_args={
                'cxx': cpp_flags,
                'nvcc': cuda_flags
            },
            libraries=['cuda'])
    ]
else:
    ext_modules = []



setup(name=PACKAGE_NAME,
      version=VERSION,
      author=AUTHOR,
      description=DESCRIPTION,
      url=URL,
      packages=find_packages(),
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExtension},
      classifiers=[
          "Programming Language :: Python :: 3",
          "Environment :: GPU :: NVIDIA CUDA :: 12",
          "License :: OSI Approved :: Apache Software License",
      ],
      python_requires='>=3.10',
      install_requires=["torch>=2.5.0"])
