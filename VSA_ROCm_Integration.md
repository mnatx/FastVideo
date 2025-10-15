# Video Sparse Attention (VSA) ROCm Integration

This document describes the integration of Video Sparse Attention (VSA) with the ROCm platform in FastVideo, using the Triton attention kernel implementation.

## Overview

Video Sparse Attention (VSA) has been enabled as an attention backend for the ROCm platform, leveraging the existing Triton-based implementation for cross-platform compatibility.

## Changes Made

### 1. Platform Backend Selection (`fastvideo/platforms/rocm.py`)
- Added `VIDEO_SPARSE_ATTN` support to the ROCm platform backend selection logic
- Added fallback logic to automatically select VSA when available
- Added proper error handling and logging

### 2. VSA Wrapper Updates (`csrc/attn/video_sparse_attn/vsa/block_sparse_wrapper.py`)
- Extended device type support to include `hip` (ROCm) alongside `cuda`
- Added ROCm device detection logic
- Created unified interface function that automatically selects appropriate implementation
- Added platform-specific logic to use Triton implementation on ROCm

### 3. VSA Backend Enhancements (`fastvideo/attention/backends/video_sparse_attn.py`)
- Added `is_available()` method to check VSA availability on current platform
- Enhanced device detection to support both CUDA and ROCm
- Added fallback import mechanism for the unified wrapper
- Added device type validation in metadata builder

### 4. Build Configuration Updates
- Updated `config_vsa.py` to support ROCm target via environment variable
- Modified `setup.py` to handle ROCm compilation flags
- Added conditional compilation to skip CUDA kernels on ROCm (uses Triton only)

## Usage

### Environment Setup
```bash
# Set target to ROCm
export VSA_TARGET=rocm

# Optional: Force CUDA kernel compilation on ROCm (not recommended)
export VSA_FORCE_CUDA_KERNELS=1
```

### Backend Selection
The VSA backend will be automatically selected on ROCm when:
1. VSA is available and properly installed
2. The current device supports the required head sizes (64, 128)
3. Triton is available (required for ROCm implementation)

### Manual Backend Selection
```python
from fastvideo.attention.selector import global_force_attn_backend
from fastvideo.platforms.interface import AttentionBackendEnum

# Force VSA backend
global_force_attn_backend(AttentionBackendEnum.VIDEO_SPARSE_ATTN)
```

## Implementation Details

### Triton-Based Implementation
- Uses the existing Triton kernels from `block_sparse_attn_triton.py`
- Triton provides native ROCm support, making this approach platform-agnostic
- No additional ROCm-specific kernel development required

### Device Detection
- Automatically detects ROCm vs CUDA platforms
- Falls back to Triton implementation on ROCm
- Uses CUDA kernels only on supported NVIDIA hardware (H100)

### Performance Considerations
- Triton implementation provides good performance on ROCm
- Memory usage optimized through sparse attention patterns
- Supports the same tile sizes and sparsity patterns as CUDA version

## Testing

Run the test script to verify VSA integration:
```bash
python test_vsa_rocm.py
```

The test will:
1. Check ROCm availability
2. Verify VSA backend import
3. Test platform integration
4. Validate backend selection

## Dependencies

- PyTorch with ROCm support
- Triton (for ROCm kernel execution)
- FastVideo VSA package

## Limitations

1. **CUDA Kernels**: ROCm-specific CUDA kernels are not implemented; relies on Triton
2. **Performance**: May not achieve optimal performance compared to native ROCm kernels
3. **Hardware Support**: Tested primarily on MI200 series (gfx90a)

## Future Improvements

1. **Native ROCm Kernels**: Implement ROCm-specific kernels for better performance
2. **Hardware Optimization**: Add support for different ROCm architectures
3. **Performance Tuning**: Optimize Triton kernels specifically for ROCm hardware

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure VSA package is properly installed
2. **Device Detection**: Verify ROCm is properly configured
3. **Triton Issues**: Check Triton installation and ROCm compatibility

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Run your VSA code
```

## Contributing

When contributing to VSA ROCm support:
1. Test on actual ROCm hardware
2. Maintain compatibility with CUDA implementation
3. Update tests and documentation
4. Consider performance implications