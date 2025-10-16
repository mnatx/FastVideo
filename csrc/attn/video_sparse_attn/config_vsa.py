### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'block_sparse': {
        'source_files': {
            'h100': 'vsa/block_sparse_h100.cu',
            'rocm': 'vsa/block_sparse_attn_triton.py'  # ROCm uses Triton implementation
        }
    }
}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
kernels = ['block_sparse']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
import os
target = os.getenv('VSA_TARGET', 'h100')  # Default to h100, can be overridden for ROCm
