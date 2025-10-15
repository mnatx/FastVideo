#!/usr/bin/env python3
"""
Test script to verify VSA integration with ROCm platform.
"""

import torch
import os

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
    
    # Test VSA backend availability
    try:
        from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionBackend
        print("VSA backend imported successfully")
        
        # Check if VSA is available
        if VideoSparseAttentionBackend.is_available():
            print("VSA backend is available")
        else:
            print("VSA backend is not available")
            return False
            
    except ImportError as e:
        print(f"Failed to import VSA backend: {e}")
        return False
    
    # Test platform integration
    try:
        from fastvideo.platforms import current_platform
        print(f"Current platform: {current_platform.device_name}")
        
        # Test backend selection
        backend_cls = current_platform.get_attn_backend_cls(
            selected_backend=None,  # Let it auto-select
            head_size=64,
            dtype=torch.float16
        )
        print(f"Selected backend: {backend_cls}")
        
        if "video_sparse_attn" in backend_cls.lower():
            print("VSA backend selected successfully!")
            return True
        else:
            print("VSA backend not selected")
            return False
            
    except Exception as e:
        print(f"Failed to test platform integration: {e}")
        return False

if __name__ == "__main__":
    success = test_vsa_rocm_integration()
    if success:
        print("\n✅ VSA ROCm integration test PASSED")
    else:
        print("\n❌ VSA ROCm integration test FAILED")