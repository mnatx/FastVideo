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

def detect_rocm_device_type():
    """Detect the specific ROCm device type."""
    if not torch.cuda.is_available():
        return "unknown"
    
    device_name = torch.cuda.get_device_name().lower()
    
    # Check for specific device patterns (order matters for overlapping names)
    if "mi300x" in device_name:
        return "mi300x"
    elif "mi300" in device_name:
        return "mi300x"  # Default MI300 to MI300X
    elif "mi250" in device_name or "m250" in device_name:
        return "mi250"
    elif "mi210" in device_name:
        return "mi210"
    elif "w7800" in device_name or "radeon pro w7800" in device_name:
        return "w7800"
    elif "radeon pro" in device_name:
        return "w7800"
    else:
        return "generic_rocm"

def get_device_optimal_config(device_type):
    """Get optimal configuration for the detected device type."""
    configs = {
        "mi300x": {
            "block_size": (2, 4, 4),  # 32 elements
            "seq_len": 64,
            "head_dim": 64,
            "description": "MI300X: Ultra-high performance with 192GB memory"
        },
        "mi250": {
            "block_size": (4, 4, 4),  # 64 elements
            "seq_len": 128,
            "head_dim": 64,
            "description": "MI250: High performance with 128GB memory and 128KB shared memory"
        },
        "mi210": {
            "block_size": (2, 4, 4),  # 32 elements
            "seq_len": 64,
            "head_dim": 64,
            "description": "MI210: Balanced performance with 64GB memory and 64KB shared memory"
        },
        "w7800": {
            "block_size": (2, 2, 4),  # 16 elements
            "seq_len": 32,
            "head_dim": 64,
            "description": "W7800: Conservative configuration with 30GB memory and 32KB shared memory"
        },
        "generic_rocm": {
            "block_size": (2, 4, 4),  # 32 elements
            "seq_len": 64,
            "head_dim": 64,
            "description": "Generic ROCm: Conservative configuration for compatibility"
        }
    }
    return configs.get(device_type, configs["generic_rocm"])

def test_vsa_rocm_integration():
    """Enhanced VSA integration test with device-specific configurations."""
    print("Testing VSA integration with ROCm...")
    
    # Check if we're on ROCm
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Detect device type
        device_type = detect_rocm_device_type()
        print(f"Detected device type: {device_type.upper()}")
        
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
    
    # Test basic VSA functionality with device-specific configurations
    try:
        # Get device-specific configuration
        device_type = detect_rocm_device_type()
        config = get_device_optimal_config(device_type)
        
        print(f"Using {config['description']}")
        print(f"  Block size: {config['block_size']}")
        print(f"  Sequence length: {config['seq_len']}")
        print(f"  Head dimension: {config['head_dim']}")
        
        # Create test tensors with device-specific parameters
        batch_size, num_heads = 1, 8
        seq_len, head_dim = config['seq_len'], config['head_dim']
        device = torch.device('cuda:0')
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        
        # Calculate block elements from block_size
        block_elements = config['block_size'][0] * config['block_size'][1] * config['block_size'][2]
        num_blocks = seq_len // block_elements
        variable_block_sizes = torch.full((num_blocks,), block_elements, device=device, dtype=torch.long)
        
        # Test VSA function
        print("Testing VSA function...")
        topk = min(2, num_blocks)  # Use at most 2 blocks, but not more than available
        
        output = vsa.video_sparse_attn(
            q, k, v, 
            variable_block_sizes=variable_block_sizes,
            topk=topk,  # Use 2 blocks for sparse attention
            block_size=config['block_size']
        )
        
        print(f"✅ VSA function executed successfully")
        print(f"   Input shape: {q.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output device: {output.device}")
        print(f"   Block elements: {block_elements}")
        print(f"   Number of blocks: {num_blocks}")
        print(f"   Top-k: {topk}")
        
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

