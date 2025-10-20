#!/usr/bin/env python3
"""
Comparison test between original and ROCm-optimized VSA test parameters.
This script demonstrates the difference in test configurations for different platforms.
"""

import torch
import sys
import os

# Add the test directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'csrc', 'attn', 'tests'))

def is_rocm_platform():
    """Check if we're running on ROCm platform."""
    return torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip is not None

def is_mi250_gpu():
    """Check if we're running on AMD Instinct MI250 GPU."""
    if not is_rocm_platform():
        return False
    device_name = torch.cuda.get_device_name().lower()
    return 'mi250' in device_name or 'instinct' in device_name

def get_platform_info():
    """Get platform information."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        device_count = torch.cuda.device_count()
        if is_rocm_platform():
            hip_version = torch.version.hip
            gpu_type = "MI250" if is_mi250_gpu() else "Other ROCm GPU"
            return f"ROCm (HIP {hip_version}) - {device_name} ({device_count} devices) [{gpu_type}]"
        else:
            return f"CUDA - {device_name} ({device_count} devices)"
    else:
        return "CPU only"

def main():
    print("Video Sparse Attention Test Configuration Comparison")
    print("=" * 60)
    print(f"Platform: {get_platform_info()}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    is_rocm = is_rocm_platform()
    is_mi250 = is_mi250_gpu()
    
    if is_rocm:
        if is_mi250:
            print("ROCm Platform Detected - AMD Instinct MI250 GPU:")
            print("-" * 50)
            print("Original Parameters (would cause shared memory error on W7800):")
            print("  h=16, d=128, num_blocks=[16, 32, 53], k=[2, 4, 6]")
            print()
            print("MI250 Optimized Parameters (recommended):")
            print("  h=16, d=128, num_blocks=[16, 32, 48], k=[2, 4, 6]")
            print()
            print("Benefits:")
            print("  ✓ MI250 has 2x shared memory (128KB vs 64KB) compared to W7800")
            print("  ✓ Can use full head dimension (d=128) that failed on W7800")
            print("  ✓ Supports larger batch sizes and sequence lengths")
            print("  ✓ Maintains same test structure and accuracy metrics")
            print("  ✓ Automatically detects MI250 and uses appropriate parameters")
        else:
            print("ROCm Platform Detected - Other GPU (W7800 or similar):")
            print("-" * 50)
            print("Original Parameters (would cause shared memory error):")
            print("  h=16, d=128, num_blocks=[16, 32, 53], k=[2, 4, 6]")
            print()
            print("ROCm Optimized Parameters (current):")
            print("  h=4, d=64, num_blocks=[4, 8, 12], k=[2, 2, 3]")
            print()
            print("Benefits:")
            print("  ✓ Avoids 'OutOfResources: shared memory' errors")
            print("  ✓ Still provides meaningful correctness testing")
            print("  ✓ Maintains same test structure and accuracy metrics")
            print("  ✓ Automatically detects ROCm and adjusts parameters")
    else:
        print("CUDA Platform Detected - Using Original Parameters:")
        print("-" * 50)
        print("Parameters:")
        print("  h=16, d=128, num_blocks=[16, 32, 53], k=[2, 4, 6]")
        print()
        print("These parameters work well on CUDA platforms with sufficient shared memory.")
    
    print()
    print("To run the tests:")
    print("  python csrc/attn/tests/test_vsa.py")
    print()
    print("The test script automatically detects the platform and uses appropriate parameters.")

if __name__ == "__main__":
    main()