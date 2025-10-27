#!/usr/bin/env python3
"""
Device specification database for AMD ROCm GPUs.
Contains shared memory constraints, optimal configurations, and hardware specifications.
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class DeviceSpec:
    """Device specification container."""
    name: str
    shared_memory_kb: int
    total_memory_gb: int
    compute_units: int
    optimal_block_size: Tuple[int, int]
    max_head_dim: int
    max_batch_size: int
    max_sequence_length: int
    memory_bandwidth_gbps: float
    peak_tflops: float
    description: str

# Device specification database
DEVICE_SPECS: Dict[str, DeviceSpec] = {
    "mi300x": DeviceSpec(
        name="AMD Instinct MI300X",
        shared_memory_kb=64,  # 64KB LDS
        total_memory_gb=192,
        compute_units=304,
        optimal_block_size=(32, 32),
        max_head_dim=128,
        max_batch_size=8,
        max_sequence_length=32768,
        memory_bandwidth_gbps=5300,
        peak_tflops=1200,
        description="Ultra-high performance data center GPU with massive memory capacity"
    ),
    "mi250": DeviceSpec(
        name="AMD Instinct MI250",
        shared_memory_kb=128,  # 128KB LDS (2x MI210)
        total_memory_gb=128,
        compute_units=220,
        optimal_block_size=(64, 64),
        max_head_dim=256,
        max_batch_size=16,
        max_sequence_length=16384,
        memory_bandwidth_gbps=3200,
        peak_tflops=400,
        description="High-performance data center GPU with large shared memory"
    ),
    "mi210": DeviceSpec(
        name="AMD Instinct MI210",
        shared_memory_kb=64,  # 64KB LDS
        total_memory_gb=64,
        compute_units=104,
        optimal_block_size=(32, 32),
        max_head_dim=128,
        max_batch_size=4,
        max_sequence_length=8192,
        memory_bandwidth_gbps=1600,
        peak_tflops=200,
        description="Data center GPU with balanced performance and memory"
    ),
    "w7800": DeviceSpec(
        name="AMD Radeon W7800",
        shared_memory_kb=32,  # 32KB LDS (most conservative)
        total_memory_gb=30,
        compute_units=70,
        optimal_block_size=(16, 16),
        max_head_dim=128,
        max_batch_size=2,
        max_sequence_length=4096,
        memory_bandwidth_gbps=576,
        peak_tflops=61,
        description="Professional workstation GPU with conservative shared memory"
    ),
    "generic_rocm": DeviceSpec(
        name="Generic ROCm GPU",
        shared_memory_kb=64,  # Default conservative value
        total_memory_gb=32,
        compute_units=64,
        optimal_block_size=(32, 32),
        max_head_dim=128,
        max_batch_size=4,
        max_sequence_length=4096,
        memory_bandwidth_gbps=1000,
        peak_tflops=100,
        description="Generic ROCm-compatible GPU with conservative settings"
    )
}

def get_device_spec(device_type: str) -> DeviceSpec:
    """Get device specification by type."""
    return DEVICE_SPECS.get(device_type, DEVICE_SPECS["generic_rocm"])

def get_shared_memory_limit(device_type: str) -> int:
    """Get shared memory limit in bytes for device type."""
    return get_device_spec(device_type).shared_memory_kb * 1024

def get_optimal_block_size(device_type: str, head_dim: int) -> Tuple[int, int]:
    """Get optimal block size for device type and head dimension."""
    spec = get_device_spec(device_type)
    base_block_size = spec.optimal_block_size
    
    # Adjust block size based on head dimension
    if head_dim <= 64:
        return base_block_size
    elif head_dim <= 128:
        # Reduce block size for larger head dimensions
        return (max(16, base_block_size[0] // 2), max(16, base_block_size[1] // 2))
    else:
        # Very conservative for head_dim > 128
        return (16, 16)

def get_max_elements(device_type: str, head_dim: int) -> int:
    """Get maximum elements that can be processed based on shared memory constraints."""
    shared_mem_bytes = get_shared_memory_limit(device_type)
    # Account for 4 bytes per element (bfloat16) and some overhead
    max_elements = shared_mem_bytes // (head_dim * 4 * 2)  # 2x for Q and K
    return max(1024, max_elements)  # Minimum 1024 elements

def validate_config_for_device(device_type: str, batch_size: int, num_heads: int, 
                             seq_len: int, head_dim: int) -> bool:
    """Validate if configuration is suitable for device type."""
    spec = get_device_spec(device_type)
    
    # Check basic constraints
    if batch_size > spec.max_batch_size:
        return False
    if head_dim > spec.max_head_dim:
        return False
    if seq_len > spec.max_sequence_length:
        return False
    
    # Check shared memory constraints
    total_elements = batch_size * num_heads * seq_len * head_dim
    max_elements = get_max_elements(device_type, head_dim)
    
    return total_elements <= max_elements

def get_device_triton_configs(device_type: str) -> List[Dict]:
    """Get optimized Triton configurations for device type."""
    spec = get_device_spec(device_type)
    base_block_size = spec.optimal_block_size
    
    if device_type == "w7800":
        # W7800: Most conservative configs for 32KB shared memory
        return [
            {"BLOCK_M": BM, "BLOCK_N": BN, "num_stages": s, "num_warps": w}
            for BM in [16, 32]
            for BN in [16, 32]
            for s in [1, 2]
            for w in [2, 4]
        ]
    elif device_type == "mi250":
        # MI250: Aggressive configs for 128KB shared memory
        return [
            {"BLOCK_M": BM, "BLOCK_N": BN, "num_stages": s, "num_warps": w}
            for BM in [32, 64, 128]
            for BN in [32, 64, 128]
            for s in [2, 3, 4]
            for w in [4, 8, 16]
        ]
    elif device_type == "mi300x":
        # MI300X: High performance configs (same shared memory as MI210)
        return [
            {"BLOCK_M": BM, "BLOCK_N": BN, "num_stages": s, "num_warps": w}
            for BM in [32, 64]
            for BN in [32, 64]
            for s in [1, 2, 3]
            for w in [2, 4, 8]
        ]
    elif device_type == "mi210":
        # MI210: Current optimized configs
        return [
            {"BLOCK_M": BM, "BLOCK_N": BN, "num_stages": s, "num_warps": w}
            for BM in [32, 64]
            for BN in [32, 64]
            for s in [1, 2, 3]
            for w in [2, 4]
        ]
    else:
        # Generic ROCm: Conservative configs
        return [
            {"BLOCK_M": BM, "BLOCK_N": BN, "num_stages": s, "num_warps": w}
            for BM in [32]
            for BN in [32]
            for s in [1, 2]
            for w in [2, 4]
        ]

def get_device_test_configs(device_type: str, quick: bool = False) -> List[Tuple[int, int, int, int, int]]:
    """Get device-specific test configurations (batch, heads, seq_len, head_dim, top_k)."""
    spec = get_device_spec(device_type)
    
    if quick:
        # Quick test configurations
        if device_type == "mi300x":
            return [(1, 32, 4096, 64, 4), (1, 16, 8192, 64, 2)]
        elif device_type == "mi250":
            return [(1, 16, 2048, 64, 2), (2, 8, 4096, 64, 4)]
        elif device_type == "mi210":
            return [(1, 8, 1024, 64, 2), (2, 4, 2048, 64, 4)]
        elif device_type == "w7800":
            return [(1, 8, 1024, 64, 2), (1, 4, 2048, 64, 4)]
        else:
            return [(1, 8, 1024, 64, 2)]
    
    # Full test configurations
    configs = []
    
    if device_type == "mi300x":
        # MI300X: Ultra-aggressive parameters for 192GB memory
        batch_sizes = [1, 2, 4, 8]
        num_heads_list = [4, 8, 16, 24, 32, 48, 64]
        seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        head_dims = [64, 128]
        top_k_list = [1, 2, 4, 8, 16, 32, 64]
    elif device_type == "mi250":
        # MI250: Aggressive parameters for 128GB memory
        batch_sizes = [1, 2, 4, 8, 16]
        num_heads_list = [4, 8, 16, 24, 32, 48, 64]
        seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        head_dims = [64, 128, 256]
        top_k_list = [1, 2, 4, 8, 16, 32, 64]
    elif device_type == "mi210":
        # MI210: Conservative parameters for 64KB shared memory
        batch_sizes = [1, 2, 4]
        num_heads_list = [2, 4, 8, 16, 24, 32]
        seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        head_dims = [64, 128]
        top_k_list = [1, 2, 4, 8, 16, 32]
    elif device_type == "w7800":
        # W7800: Most conservative parameters for 32KB shared memory
        batch_sizes = [1, 2]
        num_heads_list = [2, 4, 8, 16, 24]
        seq_lens = [64, 128, 256, 512, 1024, 2048, 4096]
        head_dims = [64, 128]
        top_k_list = [1, 2, 4, 8, 16]
    else:
        # Generic ROCm: Balanced parameters
        batch_sizes = [1, 2, 4]
        num_heads_list = [4, 8, 16, 32]
        seq_lens = [64, 128, 256, 512, 1024, 2048, 4096]
        head_dims = [64, 128]
        top_k_list = [1, 2, 4, 8, 16]
    
    # Generate all combinations and filter by device constraints
    for batch in batch_sizes:
        for heads in num_heads_list:
            for seq_len in seq_lens:
                for head_dim in head_dims:
                    for top_k in top_k_list:
                        if validate_config_for_device(device_type, batch, heads, seq_len, head_dim):
                            configs.append((batch, heads, seq_len, head_dim, top_k))
    
    return configs

def print_device_info(device_type: str) -> None:
    """Print device information and capabilities."""
    spec = get_device_spec(device_type)
    
    print(f"Device: {spec.name}")
    print(f"  Shared Memory: {spec.shared_memory_kb}KB")
    print(f"  Total Memory: {spec.total_memory_gb}GB")
    print(f"  Compute Units: {spec.compute_units}")
    print(f"  Optimal Block Size: {spec.optimal_block_size}")
    print(f"  Max Head Dimension: {spec.max_head_dim}")
    print(f"  Max Batch Size: {spec.max_batch_size}")
    print(f"  Max Sequence Length: {spec.max_sequence_length}")
    print(f"  Memory Bandwidth: {spec.memory_bandwidth_gbps} GB/s")
    print(f"  Peak TFLOPS: {spec.peak_tflops}")
    print(f"  Description: {spec.description}")

if __name__ == "__main__":
    # Print all device specifications
    for device_type in DEVICE_SPECS.keys():
        print_device_info(device_type)
        print()