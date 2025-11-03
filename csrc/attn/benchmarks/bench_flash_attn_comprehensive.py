#!/usr/bin/env python3
"""
Comprehensive FLASH_ATTN Benchmark for ROCm Platform

GPU-Specific Optimizations:
- AMD Radeon PRO W7800: Conservative configurations optimized for 30GB memory
- AMD Instinct MI250: Aggressive configurations optimized for 128GB memory
- AMD Instinct MI210: Conservative configurations for 64GB memory
- AMD Instinct MI300X: Aggressive configurations optimized for 192GB memory

The benchmark automatically detects the GPU type and selects appropriate parameter ranges.
Reports absolute TFLOPS performance and TFLOPS relative to peak performance from AMD datasheets.
"""

import torch
import argparse
import time
import json
import numpy as np
import random
from typing import Tuple, List, Dict, Optional

try:
    from flash_attn import flash_attn_func as flash_attn_2_func
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_3_func
        # flash_attn 3 no longer have a different API, see following commit:
        # https://github.com/Dao-AILab/flash-attention/commit/ed209409acedbb2379f870bbd03abce31a7a51b7
        flash_attn_func = flash_attn_3_func
    except ImportError:
        flash_attn_func = flash_attn_2_func
except ImportError:
    raise ImportError(
        "flash_attn is not installed. Please install it, e.g., "
        "`pip install flash-attn --no-build-isolation`"
    )

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def detect_gpu_type() -> str:
    """Detect the type of GPU being used."""
    if not torch.cuda.is_available():
        return "unknown"
    
    device_name = torch.cuda.get_device_name().lower()
    if "mi300x" in device_name or "mi300" in device_name:
        return "mi300x"
    elif "mi210" in device_name:
        return "mi210"
    elif "mi250" in device_name:
        return "mi250"
    elif "w7800" in device_name or "radeon pro" in device_name:
        return "w7800"
    else:
        return "generic_rocm"

def get_peak_tflops(gpu_type: str) -> float:
    """
    Get peak FP16/BF16 TFLOPS from AMD product datasheets.
    These are theoretical peak values for reference.
    """
    # Peak TFLOPS values from AMD product datasheets (FP16/BF16)
    peak_tflops = {
        "mi300x": 1200.0,  # AMD Instinct MI300X: ~1200 TFLOPS FP16
        "mi250": 383.0,    # AMD Instinct MI250: ~383 TFLOPS FP16 (per GCD)
        "mi210": 181.0,    # AMD Instinct MI210: ~181 TFLOPS FP16
        "w7800": 149.0,    # AMD Radeon PRO W7800: ~149 TFLOPS FP16
        "generic_rocm": 100.0  # Conservative estimate for unknown ROCm GPUs
    }
    return peak_tflops.get(gpu_type, 100.0)

def validate_config_for_gpu(batch: int, head: int, seq_len: int, headdim: int, gpu_type: str) -> bool:
    """Validate if a configuration is appropriate for the detected GPU."""
    # Estimate memory usage (rough approximation)
    # Q, K, V tensors: 3 * batch * head * seq_len * headdim * 2 bytes (bfloat16)
    # Flash attention uses blocked computation, reducing memory footprint
    # But we still need workspace memory
    tensor_memory = 3 * batch * head * seq_len * headdim * 2
    # Flash attention reduces memory by not storing full attention matrix
    # Estimate workspace memory as fraction of full attention matrix
    workspace_memory = batch * head * seq_len * seq_len * 2  # Conservative estimate
    output_memory = batch * head * seq_len * headdim * 2
    total_memory_bytes = tensor_memory + workspace_memory + output_memory
    
    # Convert to GB
    total_memory_gb = total_memory_bytes / (1024**3)
    
    if gpu_type == "mi300x":
        # MI300X can handle extremely large configurations with 192GB memory
        return total_memory_gb <= 50.0  # Conservative limit
    elif gpu_type == "mi250":
        # MI250 can handle very large configurations with 128GB memory
        return total_memory_gb <= 30.0
    elif gpu_type == "mi210":
        # MI210 has 64GB memory
        return total_memory_gb <= 10.0
    elif gpu_type == "w7800":
        # W7800 has 30GB memory
        return total_memory_gb <= 8.0
    else:
        # Generic ROCm - balanced approach
        return total_memory_gb <= 15.0

def generate_fastwan_configs(gpu_type: str, quick: bool = False) -> List[Tuple[int, int, int, int, bool, float]]:
    """
    Generate realistic configurations for FastWan2.1-T2V-1.3B-Diffusers.
    
    Model architecture:
    - num_attention_heads: 40
    - attention_head_dim: 128
    
    Video parameters for 5 seconds of 480p:
    - Standard: height=480, width=832, num_frames=80 (5 seconds * 16 fps)
    - FastWan: height=448, width=832, num_frames=61
    - Patch size: (1, 2, 2) → seq_len = num_frames * (height/2) * (width/2)
    
    For 480p (480x832), 80 frames: seq_len = 80 * 240 * 416 ≈ 7,987,200
    For FastWan (448x832), 61 frames: seq_len = 61 * 224 * 416 ≈ 5,678,336
    
    However, practical sequence lengths are often smaller due to:
    - Memory constraints
    - Actual model usage patterns
    - VAE compression reducing spatial dimensions
    
    We'll benchmark realistic sequence lengths that represent actual usage.
    """
    configs = []
    
    # FastWan2.1-T2V-1.3B model parameters
    fastwan_heads = 40
    fastwan_head_dim = 128
    
    # Realistic sequence lengths for video generation
    # These represent compressed/processed video tokens after VAE encoding
    # Actual sequence lengths depend on VAE compression ratio and patch size
    realistic_seq_lens = [
        # Small configurations (short videos or compressed)
        1024, 2048, 4096, 8192,
        # Medium configurations (standard video generation)
        16384, 32768, 65536,
        # Large configurations (long videos or high resolution)
        131072, 262144,
    ]
    
    if gpu_type == "mi300x":
        # MI300X can handle the full FastWan configurations
        if quick:
            configs = [
                # Quick test: FastWan architecture with moderate sequence lengths
                (1, fastwan_heads, 16384, fastwan_head_dim, False, 0.0),
                (1, fastwan_heads, 32768, fastwan_head_dim, False, 0.0),
                (1, fastwan_heads, 65536, fastwan_head_dim, True, 0.0),
            ]
        else:
            # FastWan standard configurations
            for seq_len in realistic_seq_lens[:6]:  # Up to 65536
                configs.append((1, fastwan_heads, seq_len, fastwan_head_dim, False, 0.0))
                configs.append((1, fastwan_heads, seq_len, fastwan_head_dim, True, 0.0))
            
            # Batch size variations
            for batch in [1, 2]:
                configs.append((batch, fastwan_heads, 32768, fastwan_head_dim, False, 0.0))
    
    elif gpu_type == "mi250":
        # MI250 configurations for FastWan
        if quick:
            configs = [
                (1, fastwan_heads, 8192, fastwan_head_dim, False, 0.0),
                (1, fastwan_heads, 16384, fastwan_head_dim, False, 0.0),
                (1, fastwan_heads, 32768, fastwan_head_dim, True, 0.0),
            ]
        else:
            for seq_len in realistic_seq_lens[:5]:  # Up to 32768
                configs.append((1, fastwan_heads, seq_len, fastwan_head_dim, False, 0.0))
                configs.append((1, fastwan_heads, seq_len, fastwan_head_dim, True, 0.0))
            
            configs.append((1, fastwan_heads, 32768, fastwan_head_dim, False, 0.0))
    
    elif gpu_type == "mi210":
        # MI210 configurations for FastWan
        if quick:
            configs = [
                (1, fastwan_heads, 4096, fastwan_head_dim, False, 0.0),
                (1, fastwan_heads, 8192, fastwan_head_dim, False, 0.0),
                (1, fastwan_heads, 16384, fastwan_head_dim, True, 0.0),
            ]
        else:
            for seq_len in realistic_seq_lens[:4]:  # Up to 16384
                configs.append((1, fastwan_heads, seq_len, fastwan_head_dim, False, 0.0))
                configs.append((1, fastwan_heads, seq_len, fastwan_head_dim, True, 0.0))
    
    elif gpu_type == "w7800":
        # W7800 configurations for FastWan
        if quick:
            configs = [
                (1, fastwan_heads, 4096, fastwan_head_dim, False, 0.0),
                (1, fastwan_heads, 8192, fastwan_head_dim, False, 0.0),
                (1, fastwan_heads, 16384, fastwan_head_dim, True, 0.0),
            ]
        else:
            for seq_len in realistic_seq_lens[:4]:  # Up to 16384
                configs.append((1, fastwan_heads, seq_len, fastwan_head_dim, False, 0.0))
                configs.append((1, fastwan_heads, seq_len, fastwan_head_dim, True, 0.0))
    
    else:
        # Generic ROCm configurations for FastWan
        if quick:
            configs = [
                (1, fastwan_heads, 4096, fastwan_head_dim, False, 0.0),
                (1, fastwan_heads, 8192, fastwan_head_dim, False, 0.0),
                (1, fastwan_heads, 16384, fastwan_head_dim, True, 0.0),
            ]
        else:
            for seq_len in realistic_seq_lens[:4]:  # Up to 16384
                configs.append((1, fastwan_heads, seq_len, fastwan_head_dim, False, 0.0))
                configs.append((1, fastwan_heads, seq_len, fastwan_head_dim, True, 0.0))
    
    return configs

def generate_gpu_specific_configs(gpu_type: str, quick: bool = False) -> List[Tuple[int, int, int, int, bool, float]]:
    """
    Generate configurations appropriate for the detected GPU type.
    Returns list of (batch, heads, seq_len, head_dim, causal, dropout) tuples.
    """
    configs = []
    
    # First, add FastWan2.1-T2V-1.3B-Diffusers realistic configurations
    fastwan_configs = generate_fastwan_configs(gpu_type, quick)
    configs.extend(fastwan_configs)
    
    if gpu_type == "mi300x":
        # MI300X configurations - extremely aggressive parameters for 192GB memory
        if quick:
            configs = [
                # Quick test configurations for MI300X
                (1, 32, 4096, 64, False, 0.0),
                (2, 32, 8192, 64, False, 0.0),
                (1, 32, 16384, 64, True, 0.0),
            ]
        else:
            # Different sequence lengths
            for seq_len in [2048, 4096, 8192, 16384, 32768, 65536]:
                for causal in [False, True]:
                    for dropout in [0.0, 0.1]:
                        configs.append((1, 32, seq_len, 64, causal, dropout))
            
            # Different head counts
            for num_heads in [16, 24, 32, 48, 64]:
                configs.append((1, num_heads, 16384, 64, False, 0.0))
            
            # Different head dimensions
            for head_dim in [64, 128, 256]:
                configs.append((1, 32, 16384, head_dim, False, 0.0))
            
            # Larger batch sizes
            for batch in [1, 2, 4, 8]:
                configs.append((batch, 32, 16384, 64, False, 0.0))
    
    elif gpu_type == "mi250":
        # MI250 configurations - aggressive parameters for 128GB memory
        if quick:
            configs = [
                # Quick test configurations for MI250
                (1, 16, 2048, 64, False, 0.0),
                (2, 24, 4096, 64, False, 0.0),
                (1, 32, 8192, 64, True, 0.0),
            ]
        else:
            # Different sequence lengths
            for seq_len in [1024, 2048, 4096, 8192, 16384, 32768]:
                for causal in [False, True]:
                    configs.append((1, 16, seq_len, 64, causal, 0.0))
            
            # Different head counts
            for num_heads in [8, 16, 24, 32]:
                configs.append((1, num_heads, 8192, 64, False, 0.0))
            
            # Different head dimensions
            for head_dim in [64, 128, 256]:
                configs.append((1, 16, 8192, head_dim, False, 0.0))
            
            # Batch sizes
            for batch in [1, 2, 4]:
                configs.append((batch, 16, 8192, 64, False, 0.0))
    
    elif gpu_type == "mi210":
        # MI210 configurations - conservative for 64GB memory
        if quick:
            configs = [
                # Quick test configurations for MI210
                (1, 8, 1024, 64, False, 0.0),
                (1, 12, 2048, 64, False, 0.0),
                (1, 16, 4096, 64, True, 0.0),
            ]
        else:
            # Different sequence lengths
            for seq_len in [512, 1024, 2048, 4096, 8192]:
                for causal in [False, True]:
                    configs.append((1, 12, seq_len, 64, causal, 0.0))
            
            # Different head counts
            for num_heads in [4, 8, 12, 16, 24]:
                configs.append((1, num_heads, 2048, 64, False, 0.0))
            
            # Different head dimensions
            for head_dim in [64, 128]:
                configs.append((1, 12, 2048, head_dim, False, 0.0))
            
            # Batch sizes
            for batch in [1, 2]:
                configs.append((batch, 12, 2048, 64, False, 0.0))
    
    elif gpu_type == "w7800":
        # W7800 configurations - conservative parameters for 30GB memory
        if quick:
            configs = [
                # Quick test configurations for W7800
                (1, 8, 1024, 64, False, 0.0),
                (1, 12, 2048, 64, False, 0.0),
                (1, 16, 4096, 64, True, 0.0),
            ]
        else:
            # Different sequence lengths
            for seq_len in [512, 1024, 2048, 4096, 8192]:
                for causal in [False, True]:
                    configs.append((1, 12, seq_len, 64, causal, 0.0))
            
            # Different head counts
            for num_heads in [4, 8, 12, 16]:
                configs.append((1, num_heads, 4096, 64, False, 0.0))
            
            # Different head dimensions
            for head_dim in [64, 128]:
                configs.append((1, 12, 4096, head_dim, False, 0.0))
            
            # Batch sizes
            for batch in [1, 2]:
                configs.append((batch, 12, 4096, 64, False, 0.0))
    
    else:
        # Generic ROCm configurations - balanced approach
        if quick:
            configs = [
                # Quick test configurations for generic ROCm
                (1, 12, 1024, 64, False, 0.0),
                (1, 12, 2048, 64, False, 0.0),
                (1, 12, 4096, 64, True, 0.0),
            ]
        else:
            # Different sequence lengths
            for seq_len in [512, 1024, 2048, 4096, 8192]:
                for causal in [False, True]:
                    configs.append((1, 12, seq_len, 64, causal, 0.0))
            
            # Different head counts
            for num_heads in [4, 8, 12, 16]:
                configs.append((1, num_heads, 4096, 64, False, 0.0))
            
            # Different head dimensions
            for head_dim in [64, 128]:
                configs.append((1, 12, 4096, head_dim, False, 0.0))
    
    # Filter configurations based on GPU memory limits
    validated_configs = []
    for config in configs:
        batch, head, seq_len, headdim, causal, dropout = config
        if validate_config_for_gpu(batch, head, seq_len, headdim, gpu_type):
            validated_configs.append(config)
        else:
            print(f"Skipping config {config} - exceeds memory limits for {gpu_type}")
    
    return validated_configs

def create_input_tensors(batch: int, head: int, seq_len: int, headdim: int, device: str = "cuda"):
    """
    Create random input tensors for flash attention.
    Flash attention expects tensors in format: (batch, seq_len, head, headdim)
    """
    # Flash attention format: (batch, seq_len, head, headdim)
    q = torch.randn(batch, seq_len, head, headdim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, seq_len, head, headdim, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch, seq_len, head, headdim, dtype=torch.bfloat16, device=device)
    return q, k, v

def calculate_attention_flops(batch: int, head: int, seq_len: int, headdim: int, causal: bool = False) -> int:
    """
    Calculate FLOPs for flash attention.
    
    Forward pass:
    - QK^T: batch * head * seq_len * seq_len * headdim (multiply-add)
    - Softmax: batch * head * seq_len * seq_len (approximate, expensive)
    - Attention * V: batch * head * seq_len * seq_len * headdim (multiply-add)
    
    Flash attention uses blocked computation but still performs the same operations,
    just with better memory efficiency.
    
    Total forward: ~2 * batch * head * seq_len^2 * headdim (QK^T and AV)
    Plus softmax: ~batch * head * seq_len^2 operations
    
    For causal attention, we compute only lower triangular, so:
    - FLOPs ≈ batch * head * seq_len * (seq_len + 1) / 2 * headdim
    
    Conservative estimate: 4 * batch * head * seq_len^2 * headdim (includes softmax overhead)
    """
    if causal:
        # Causal attention: only compute lower triangular matrix
        # Approximate: seq_len * (seq_len + 1) / 2 elements
        effective_elements = seq_len * (seq_len + 1) / 2
        flops = 4 * batch * head * effective_elements * headdim
    else:
        # Full attention: seq_len^2 elements
        flops = 4 * batch * head * seq_len * seq_len * headdim
    
    return int(flops)

def benchmark_configuration(
    batch: int,
    head: int,
    seq_len: int,
    headdim: int,
    causal: bool,
    dropout: float,
    num_runs: int = 5
) -> Dict:
    """Benchmark a specific configuration."""
    print(f"Benchmarking: batch={batch}, heads={head}, seq_len={seq_len}, head_dim={headdim}, "
          f"causal={causal}, dropout={dropout}")
    
    # Create tensors
    q, k, v = create_input_tensors(batch, head, seq_len, headdim)
    
    # Calculate theoretical FLOPs
    forward_flops = calculate_attention_flops(batch, head, seq_len, headdim, causal)
    # Backward pass is approximately 2.5x forward FLOPs
    total_flops = forward_flops + int(2.5 * forward_flops)
    
    # Compute softmax scale
    softmax_scale = 1.0 / np.sqrt(headdim)
    
    # Warm-up
    for _ in range(3):
        q_fwd = q.clone().requires_grad_(True)
        k_fwd = k.clone().requires_grad_(True)
        v_fwd = v.clone().requires_grad_(True)
        output = flash_attn_func(
            q_fwd, k_fwd, v_fwd,
            softmax_scale=softmax_scale,
            causal=causal,
            dropout_p=dropout
        )
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        q_fwd = q.clone().requires_grad_(True)
        k_fwd = k.clone().requires_grad_(True)
        v_fwd = v.clone().requires_grad_(True)
        output = flash_attn_func(
            q_fwd, k_fwd, v_fwd,
            softmax_scale=softmax_scale,
            causal=causal,
            dropout_p=dropout
        )
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # Calculate TFLOPS
    tflops = total_flops / avg_time * 1e-12
    
    return {
        'batch': batch,
        'heads': head,
        'seq_len': seq_len,
        'head_dim': headdim,
        'causal': causal,
        'dropout': dropout,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'tflops': tflops,
        'forward_flops': forward_flops,
        'total_flops': total_flops
    }

def main():
    parser = argparse.ArgumentParser(description='Comprehensive FLASH_ATTN Benchmark for ROCm')
    parser.add_argument('--output', type=str, default='flash_attn_benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with fewer configurations')
    parser.add_argument('--gpu-type', type=str,
                       choices=['mi300x', 'mi250', 'mi210', 'w7800', 'auto'],
                       default='auto',
                       help='Force specific GPU type (auto-detects if not specified)')
    parser.add_argument('--num-runs', type=int, default=5,
                       help='Number of benchmark runs per configuration')
    args = parser.parse_args()
    
    set_seed(42)
    
    print("FLASH_ATTN Comprehensive Benchmark for ROCm")
    print("=" * 60)
    
    # Detect GPU type
    if args.gpu_type == 'auto':
        gpu_type = detect_gpu_type()
    else:
        gpu_type = args.gpu_type
    
    # Get device info
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"GPU Type: {gpu_type.upper()}")
    print(f"ROCm Version: {torch.version.hip}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check flash-attn version
    try:
        import flash_attn
        print(f"Flash Attention Version: {flash_attn.__version__}")
    except:
        pass
    
    # Get peak TFLOPS
    peak_tflops = get_peak_tflops(gpu_type)
    print(f"Peak TFLOPS (FP16/BF16): {peak_tflops:.1f}")
    
    # Show GPU-specific optimization info
    if gpu_type == "mi300x":
        print("✅ Using MI300X-optimized configurations (192GB memory, extremely aggressive parameters)")
    elif gpu_type == "mi250":
        print("✅ Using MI250-optimized configurations (128GB memory, aggressive parameters)")
    elif gpu_type == "mi210":
        print("✅ Using MI210-optimized configurations (64GB memory, conservative parameters)")
    elif gpu_type == "w7800":
        print("✅ Using W7800-optimized configurations (30GB memory, conservative parameters)")
    else:
        print("✅ Using generic ROCm configurations (balanced parameters)")
    print()
    
    # Generate GPU-specific configurations
    configs = generate_gpu_specific_configs(gpu_type, args.quick)
    
    # Count FastWan-specific configurations
    fastwan_configs = [c for c in configs if c[1] == 40 and c[3] == 128]
    print(f"Testing {len(configs)} configurations...")
    print(f"  - FastWan2.1-T2V-1.3B-Diffusers configs: {len(fastwan_configs)}")
    print(f"  - General configs: {len(configs) - len(fastwan_configs)}")
    print()
    
    results = []
    
    for i, (batch, head, seq_len, headdim, causal, dropout) in enumerate(configs):
        try:
            result = benchmark_configuration(batch, head, seq_len, headdim, causal, dropout, args.num_runs)
            result['peak_tflops'] = peak_tflops
            result['relative_performance'] = result['tflops'] / peak_tflops * 100.0  # Percentage of peak
            results.append(result)
            print(f"✓ [{i+1:2d}/{len(configs)}] {result['tflops']:.2f} TFLOPS "
                  f"({result['relative_performance']:.1f}% of peak)")
        except Exception as e:
            print(f"✗ [{i+1:2d}/{len(configs)}] Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    if results:
        # Sort by TFLOPS
        results.sort(key=lambda x: x['tflops'], reverse=True)
        
        print(f"GPU Type: {gpu_type.upper()}")
        print(f"Peak TFLOPS (FP16/BF16): {peak_tflops:.1f}")
        print(f"Total configurations tested: {len(results)}")
        print(f"Best performance: {results[0]['tflops']:.2f} TFLOPS "
              f"({results[0]['relative_performance']:.1f}% of peak)")
        print(f"  - Config: batch={results[0]['batch']}, heads={results[0]['heads']}, "
              f"seq_len={results[0]['seq_len']}, head_dim={results[0]['head_dim']}, "
              f"causal={results[0]['causal']}, dropout={results[0]['dropout']}")
        
        # Show configuration ranges used
        if results:
            batch_sizes = sorted(set(r['batch'] for r in results))
            num_heads = sorted(set(r['heads'] for r in results))
            seq_lens = sorted(set(r['seq_len'] for r in results))
            head_dims = sorted(set(r['head_dim'] for r in results))
            causal_modes = sorted(set(r['causal'] for r in results))
            dropout_vals = sorted(set(r['dropout'] for r in results))
            
            print(f"\nConfiguration ranges tested:")
            print(f"  Batch sizes: {batch_sizes}")
            print(f"  Number of heads: {num_heads}")
            print(f"  Sequence lengths: {seq_lens}")
            print(f"  Head dimensions: {head_dims}")
            print(f"  Causal modes: {causal_modes}")
            print(f"  Dropout values: {dropout_vals}")
        
        # Performance by sequence length
        print("\nPerformance by sequence length:")
        seq_perf = {}
        for r in results:
            seq_len = r['seq_len']
            if seq_len not in seq_perf:
                seq_perf[seq_len] = []
            seq_perf[seq_len].append(r['tflops'])
        
        for seq_len in sorted(seq_perf.keys()):
            perfs = seq_perf[seq_len]
            avg_rel = np.mean([r['relative_performance'] for r in results if r['seq_len'] == seq_len])
            print(f"  {seq_len:6d}: {max(perfs):8.2f} TFLOPS (max), {np.mean(perfs):8.2f} TFLOPS (avg), "
                  f"{avg_rel:.1f}% of peak (avg)")
        
        # Performance by causal vs non-causal
        print("\nPerformance by attention mode:")
        causal_perf = [r['tflops'] for r in results if r['causal']]
        non_causal_perf = [r['tflops'] for r in results if not r['causal']]
        if causal_perf:
            avg_rel_causal = np.mean([r['relative_performance'] for r in results if r['causal']])
            print(f"  Causal:     {max(causal_perf):8.2f} TFLOPS (max), {np.mean(causal_perf):8.2f} TFLOPS (avg), "
                  f"{avg_rel_causal:.1f}% of peak (avg)")
        if non_causal_perf:
            avg_rel_non_causal = np.mean([r['relative_performance'] for r in results if not r['causal']])
            print(f"  Non-causal: {max(non_causal_perf):8.2f} TFLOPS (max), {np.mean(non_causal_perf):8.2f} TFLOPS (avg), "
                  f"{avg_rel_non_causal:.1f}% of peak (avg)")
        
        # FastWan-specific performance summary
        fastwan_results = [r for r in results if r['heads'] == 40 and r['head_dim'] == 128]
        if fastwan_results:
            print("\n" + "=" * 60)
            print("FastWan2.1-T2V-1.3B-Diffusers Performance Summary")
            print("=" * 60)
            fastwan_results.sort(key=lambda x: x['tflops'], reverse=True)
            print(f"FastWan configurations tested: {len(fastwan_results)}")
            if fastwan_results:
                print(f"Best FastWan performance: {fastwan_results[0]['tflops']:.2f} TFLOPS "
                      f"({fastwan_results[0]['relative_performance']:.1f}% of peak)")
                print(f"  - Config: batch={fastwan_results[0]['batch']}, heads={fastwan_results[0]['heads']}, "
                      f"seq_len={fastwan_results[0]['seq_len']}, head_dim={fastwan_results[0]['head_dim']}, "
                      f"causal={fastwan_results[0]['causal']}")
                
                # FastWan performance by sequence length
                print("\nFastWan performance by sequence length:")
                fastwan_seq_perf = {}
                for r in fastwan_results:
                    seq_len = r['seq_len']
                    if seq_len not in fastwan_seq_perf:
                        fastwan_seq_perf[seq_len] = []
                    fastwan_seq_perf[seq_len].append(r['tflops'])
                
                for seq_len in sorted(fastwan_seq_perf.keys()):
                    perfs = fastwan_seq_perf[seq_len]
                    avg_rel = np.mean([r['relative_performance'] for r in fastwan_results if r['seq_len'] == seq_len])
                    print(f"  {seq_len:6d}: {max(perfs):8.2f} TFLOPS (max), {np.mean(perfs):8.2f} TFLOPS (avg), "
                          f"{avg_rel:.1f}% of peak (avg)")
        
        # Top 10 configurations
        print("\nTop 10 performing configurations:")
        print(f"{'Rank':<5} {'Batch':<6} {'Heads':<6} {'SeqLen':<7} {'HeadDim':<8} {'Causal':<7} {'TFLOPS':<10} {'% Peak':<8}")
        print("-" * 70)
        for i, r in enumerate(results[:10]):
            print(f"{i+1:<5} {r['batch']:<6} {r['heads']:<6} {r['seq_len']:<7} {r['head_dim']:<8} "
                  f"{str(r['causal']):<7} {r['tflops']:<10.2f} {r['relative_performance']:<8.1f}")
        
        # GPU-specific insights
        print(f"\nGPU-Specific Performance Insights:")
        if gpu_type == "mi300x":
            print("  ✅ MI300X: Massive memory capacity (192GB) enables testing of ultra-long sequences")
            print("  ✅ Optimal for: Ultra-long sequences, massive models, high-throughput processing")
        elif gpu_type == "mi250":
            print("  ✅ MI250: Large memory capacity enables testing of very long sequences")
            print("  ✅ Optimal for: Long sequences, high-throughput processing, large models")
        elif gpu_type == "mi210":
            print("  ✅ MI210: Balanced memory capacity for medium-scale workloads")
            print("  ✅ Optimal for: Medium sequences, efficient resource usage")
        elif gpu_type == "w7800":
            print("  ✅ W7800: Conservative memory usage optimized for professional workloads")
            print("  ✅ Optimal for: Standard workloads, moderate sequence lengths")
        else:
            print("  ✅ Generic ROCm: Balanced configurations for various ROCm-compatible GPUs")
    
    print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
