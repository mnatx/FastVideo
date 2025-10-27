#!/usr/bin/env python3
"""
Comprehensive test script for all supported AMD ROCm devices.
Tests VSA integration with device-specific optimizations for:
- AMD Instinct MI300X
- AMD Instinct MI250  
- AMD Instinct MI210
- AMD Radeon W7800
"""

import os
import sys

# Disable torch.compile to avoid compatibility issues
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
import time

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

def get_device_test_configs(device_type):
    """Get comprehensive test configurations for each device type."""
    configs = {
        "mi300x": [
            {"block_size": (2, 4, 4), "seq_len": 64, "head_dim": 64, "description": "MI300X: 32-element blocks"},
            {"block_size": (2, 4, 4), "seq_len": 128, "head_dim": 64, "description": "MI300X: Longer sequence"},
            {"block_size": (2, 4, 4), "seq_len": 64, "head_dim": 128, "description": "MI300X: Larger head dimension"},
        ],
        "mi250": [
            {"block_size": (4, 4, 4), "seq_len": 128, "head_dim": 64, "description": "MI250: 64-element blocks"},
            {"block_size": (4, 4, 4), "seq_len": 256, "head_dim": 64, "description": "MI250: Longer sequence"},
            {"block_size": (4, 4, 4), "seq_len": 128, "head_dim": 128, "description": "MI250: Larger head dimension"},
            {"block_size": (2, 4, 4), "seq_len": 64, "head_dim": 64, "description": "MI250: 32-element blocks"},
        ],
        "mi210": [
            {"block_size": (2, 4, 4), "seq_len": 64, "head_dim": 64, "description": "MI210: 32-element blocks"},
            {"block_size": (2, 4, 4), "seq_len": 128, "head_dim": 64, "description": "MI210: Longer sequence"},
            {"block_size": (2, 4, 4), "seq_len": 64, "head_dim": 128, "description": "MI210: Larger head dimension"},
        ],
        "w7800": [
            {"block_size": (2, 2, 4), "seq_len": 32, "head_dim": 64, "description": "W7800: 16-element blocks"},
            {"block_size": (2, 2, 4), "seq_len": 64, "head_dim": 64, "description": "W7800: Longer sequence"},
            {"block_size": (2, 4, 4), "seq_len": 32, "head_dim": 64, "description": "W7800: 32-element blocks"},
        ],
        "generic_rocm": [
            {"block_size": (2, 4, 4), "seq_len": 64, "head_dim": 64, "description": "Generic ROCm: 32-element blocks"},
        ]
    }
    return configs.get(device_type, configs["generic_rocm"])

def test_vsa_configuration(block_size, seq_len, head_dim, batch_size=1, num_heads=8):
    """Test VSA with specific configuration."""
    try:
        device = torch.device('cuda:0')
        
        # Create test tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        
        # Calculate block elements and create variable block sizes
        block_elements = block_size[0] * block_size[1] * block_size[2]
        num_blocks = seq_len // block_elements
        variable_block_sizes = torch.full((num_blocks,), block_elements, device=device, dtype=torch.long)
        
        # Test VSA function
        topk = min(2, num_blocks)
        
        start_time = time.time()
        output = vsa.video_sparse_attn(
            q, k, v, 
            variable_block_sizes=variable_block_sizes,
            topk=topk,
            block_size=block_size
        )
        end_time = time.time()
        
        # Verify output
        assert output.shape == q.shape, f"Output shape mismatch: expected {q.shape}, got {output.shape}"
        assert output.dtype == q.dtype, f"Output dtype mismatch: expected {q.dtype}, got {output.dtype}"
        assert output.device == q.device, f"Output device mismatch: expected {q.device}, got {output.device}"
        
        return {
            "success": True,
            "execution_time": end_time - start_time,
            "output_shape": output.shape,
            "block_elements": block_elements,
            "num_blocks": num_blocks,
            "topk": topk
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "block_elements": block_size[0] * block_size[1] * block_size[2] if block_size else 0
        }

def main():
    """Main test function."""
    print("=" * 80)
    print("AMD ROCm Device VSA Integration Test Suite")
    print("=" * 80)
    
    # Check if we're on ROCm
    if not torch.cuda.is_available():
        print("❌ No CUDA/ROCm available")
        return False
    
    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    print(f"Device count: {torch.cuda.device_count()}")
    
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        print(f"ROCm detected: {torch.version.hip}")
    else:
        print("CUDA detected (not ROCm)")
        return False
    
    # Detect device type
    device_type = detect_rocm_device_type()
    print(f"Detected device type: {device_type.upper()}")
    print()
    
    # Test VSA package import
    try:
        import vsa
        print("✅ VSA package imported successfully")
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
    
    print()
    
    # Get device-specific test configurations
    test_configs = get_device_test_configs(device_type)
    print(f"Testing {len(test_configs)} configurations for {device_type.upper()}:")
    print()
    
    # Run tests
    results = []
    for i, config in enumerate(test_configs, 1):
        print(f"Test {i}/{len(test_configs)}: {config['description']}")
        print(f"  Block size: {config['block_size']}")
        print(f"  Sequence length: {config['seq_len']}")
        print(f"  Head dimension: {config['head_dim']}")
        
        result = test_vsa_configuration(
            config['block_size'],
            config['seq_len'],
            config['head_dim']
        )
        
        if result['success']:
            print(f"  ✅ PASSED - Time: {result['execution_time']:.4f}s")
            print(f"     Output shape: {result['output_shape']}")
            print(f"     Block elements: {result['block_elements']}")
            print(f"     Number of blocks: {result['num_blocks']}")
            print(f"     Top-k: {result['topk']}")
        else:
            print(f"  ❌ FAILED - Error: {result['error']}")
            print(f"     Block elements: {result['block_elements']}")
        
        results.append({
            'config': config,
            'result': result
        })
        print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['result']['success'])
    total = len(results)
    
    print(f"Device Type: {device_type.upper()}")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print(f"VSA integration is working correctly on {device_type.upper()}")
        return True
    else:
        print(f"\n❌ {total - passed} TESTS FAILED")
        print("Check the error messages above for details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)