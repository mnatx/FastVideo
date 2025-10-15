#!/bin/bash
# Installation script for VSA on ROCm platform

set -e

echo "Installing Video Sparse Attention for ROCm..."

# Set environment variables
export VSA_TARGET=rocm

# Check if we're on ROCm
if ! python -c "import torch; print('ROCm detected' if hasattr(torch.version, 'hip') and torch.version.hip else 'CUDA detected')" | grep -q "ROCm"; then
    echo "Warning: ROCm not detected. This script is designed for ROCm platforms."
    echo "Continuing anyway..."
fi

# Install VSA package
echo "Installing VSA package..."
cd csrc/attn/video_sparse_attn
python setup.py build_ext --inplace
python setup.py install

# Test installation
echo "Testing VSA installation..."
python ../../../test_vsa_rocm.py

echo "VSA ROCm installation completed successfully!"