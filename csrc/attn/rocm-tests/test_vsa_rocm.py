#!/usr/bin/env python3
"""
Simple test script to verify VSA integration with ROCm platform.
This script bypasses the problematic torch.compile issues.
"""

import os
import sys

# Disable torch.compile to avoid compatibility issues
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch

def test_vsa_rocm_integration():
    """Test VSA integration on ROCm platform."""
    print("Testing VSA integration with ROCm...")
    
    # Check if we're on ROCm
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Check if we're on ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            print(f"ROCm detected: {torch.version.hip}")
        else:
            print("CUDA detected (not ROCm)")
    else:
        print("No CUDA/ROCm available")
        return False
    
    # Test VSA package import directly
    try:
        import vsa
        print("✅ VSA package imported successfully")
        
        # Test the video_sparse_attn function
        if hasattr(vsa, 'video_sparse_attn'):
            print("✅ video_sparse_attn function available")
        else:
            print("❌ video_sparse_attn function not available")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import VSA package: {e}")
        return False
    
    # Test Triton import
    try:
        import triton
        print(f"✅ Triton imported successfully (version: {triton.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import Triton: {e}")
        return False
    
    # Test basic VSA functionality
    try:
        # Create test tensors
        batch_size, num_heads, seq_len, head_dim = 1, 8, 64, 64
        device = torch.device('cuda:0')
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        
        # Create variable block sizes
        num_blocks = seq_len // 64  # Assuming 64 is the block size
        variable_block_sizes = torch.full((num_blocks,), 64, device=device, dtype=torch.long)
        
        # Test VSA function
        print("Testing VSA function...")
        # Adjust parameters for the test
        num_blocks = seq_len // 64  # 64 elements per block
        topk = min(2, num_blocks)  # Use at most 2 blocks, but not more than available
        
        output = vsa.video_sparse_attn(
            q, k, v, 
            variable_block_sizes=variable_block_sizes,
            topk=topk,  # Use 2 blocks for sparse attention
            block_size=(4, 4, 4)  # 4x4x4 = 64 elements per block
        )
        
        print(f"✅ VSA function executed successfully")
        print(f"   Input shape: {q.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output device: {output.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ VSA function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vsa_rocm_integration()
    if success:
        print("\n✅ VSA ROCm integration test PASSED")
    else:
        print("\n❌ VSA ROCm integration test FAILED")

