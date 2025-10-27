#!/usr/bin/env python3
"""
Comprehensive Video Sparse Attention Benchmark for ROCm Platform

GPU-Specific Optimizations:
- AMD Radeon PRO W7800: Conservative configurations optimized for 30GB memory
- AMD Instinct MI250: Aggressive configurations optimized for 128GB memory
- Generic ROCm: Balanced configurations for other ROCm-compatible GPUs

The benchmark automatically detects the GPU type and selects appropriate parameter ranges.
"""

import torch
import argparse
import triton.testing
import time
import json
from vsa import block_sparse_attn
from vsa import BLOCK_M, BLOCK_N
import numpy as np
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def detect_gpu_type():
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

def validate_config_for_gpu(batch, head, seq_len, headdim, gpu_type):
    """Validate if a configuration is appropriate for the detected GPU."""
    total_elements = batch * head * seq_len * headdim
    
    if gpu_type == "mi300x":
        # MI300X can handle extremely large configurations with 192GB memory
        return total_elements <= 100000000  # 100M elements
    elif gpu_type == "mi250":
        # MI250 can handle very large configurations
        return total_elements <= 50000000  # 50M elements
    elif gpu_type == "mi210":
        # MI210 has limited shared memory (64KB vs 128KB on MI250)
        # Conservative limits to avoid shared memory errors
        # Also restrict head_dim to 64 to avoid shared memory issues
        if headdim > 64:
            return False
        return total_elements <= 2000000   # 2M elements (slightly more permissive)
    elif gpu_type == "w7800":
        # W7800 has more conservative limits
        return total_elements <= 5000000   # 5M elements
    else:
        # Generic ROCm - balanced approach
        return total_elements <= 10000000  # 10M elements

def generate_gpu_specific_configs(gpu_type, quick=False):
    """Generate configurations appropriate for the detected GPU type."""
    if gpu_type == "mi300x":
        # MI300X configurations - extremely aggressive parameters for 192GB memory
        if quick:
            configs = [
                # Quick test configurations for MI300X
                (1, 32, 4096, 64, 4),
                (2, 32, 8192, 64, 8),
                (1, 32, 16384, 64, 16),
            ]
        else:
            configs = [
                # Different sequence lengths with MI300X-optimized parameters
                (1, 32, 2048, 64, 1),
                (1, 32, 2048, 64, 2),
                (1, 32, 2048, 64, 4),
                (1, 32, 2048, 64, 8),
                (1, 32, 2048, 64, 16),
                
                (1, 32, 4096, 64, 1),
                (1, 32, 4096, 64, 2),
                (1, 32, 4096, 64, 4),
                (1, 32, 4096, 64, 8),
                (1, 32, 4096, 64, 16),
                
                (1, 32, 8192, 64, 1),
                (1, 32, 8192, 64, 2),
                (1, 32, 8192, 64, 4),
                (1, 32, 8192, 64, 8),
                (1, 32, 8192, 64, 16),
                
                (1, 32, 16384, 64, 1),
                (1, 32, 16384, 64, 2),
                (1, 32, 16384, 64, 4),
                (1, 32, 16384, 64, 8),
                (1, 32, 16384, 64, 16),
                
                # Very long sequences for MI300X
                (1, 32, 32768, 64, 1),
                (1, 32, 32768, 64, 2),
                (1, 32, 32768, 64, 4),
                (1, 32, 32768, 64, 8),
                
                # Ultra-long sequences
                (1, 32, 65536, 64, 1),
                (1, 32, 65536, 64, 2),
                (1, 32, 65536, 64, 4),
                
                # Different head counts
                (1, 16, 16384, 64, 8),
                (1, 24, 16384, 64, 8),
                (1, 40, 16384, 64, 8),
                (1, 48, 16384, 64, 8),
                (1, 64, 16384, 64, 8),
                
                # Different head dimensions
                (1, 32, 16384, 128, 8),
                (1, 32, 16384, 256, 8),
                (1, 32, 16384, 512, 8),
                
                # Large batch sizes
                (2, 32, 16384, 64, 8),
                (4, 32, 16384, 64, 8),
                (8, 32, 16384, 64, 8),
                (16, 32, 16384, 64, 8),
                
                # High sparsity configurations
                (1, 32, 32768, 64, 1),
                (1, 32, 65536, 64, 1),
                (1, 64, 16384, 64, 1),
                
                # Memory-intensive configurations
                (1, 32, 131072, 64, 1),  # 131K sequence length
                (1, 32, 131072, 64, 2),
                (1, 32, 131072, 64, 4),
            ]
    
    elif gpu_type == "mi250":
        # MI250 configurations - aggressive parameters for 128GB memory
        if quick:
            configs = [
                # Quick test configurations for MI250
                (1, 16, 2048, 64, 2),
                (2, 24, 4096, 64, 4),
                (1, 32, 8192, 64, 8),
            ]
        else:
            configs = [
                # Different sequence lengths with MI250-optimized parameters
                (1, 16, 1024, 64, 1),
                (1, 16, 1024, 64, 2),
                (1, 16, 1024, 64, 4),
                (1, 16, 1024, 64, 8),
                
                (1, 16, 2048, 64, 1),
                (1, 16, 2048, 64, 2),
                (1, 16, 2048, 64, 4),
                (1, 16, 2048, 64, 8),
                
                (1, 16, 4096, 64, 1),
                (1, 16, 4096, 64, 2),
                (1, 16, 4096, 64, 4),
                (1, 16, 4096, 64, 8),
                
                (1, 16, 8192, 64, 1),
                (1, 16, 8192, 64, 2),
                (1, 16, 8192, 64, 4),
                (1, 16, 8192, 64, 8),
                
                # Very long sequences for MI250
                (1, 16, 16384, 64, 1),
                (1, 16, 16384, 64, 2),
                (1, 16, 16384, 64, 4),
                
                # Different head counts
                (1, 8, 4096, 64, 4),
                (1, 24, 4096, 64, 4),
                (1, 32, 4096, 64, 4),
                
                # Different head dimensions
                (1, 16, 4096, 128, 4),
                (1, 16, 4096, 256, 4),
                
                # Larger batch sizes
                (2, 16, 4096, 64, 4),
                (4, 16, 4096, 64, 4),
                (8, 16, 4096, 64, 4),
                
                # High sparsity configurations
                (1, 16, 8192, 64, 1),
                (1, 16, 16384, 64, 1),
                (1, 32, 8192, 64, 1),
            ]
    
    elif gpu_type == "mi210":
        # MI210 configurations - very conservative for shared memory limitations
        # MI210 has limited shared memory (64KB vs 128KB on MI250)
        if quick:
            configs = [
                # Quick test configurations for MI210
                (1, 4, 512, 64, 2),
                (1, 8, 1024, 64, 4),
                (1, 8, 2048, 64, 8),
            ]
        else:
            configs = [
                # Very conservative configurations for MI210
                (1, 4, 512, 64, 1),
                (1, 4, 512, 64, 2),
                (1, 4, 512, 64, 4),
                (1, 4, 512, 64, 8),
                
                (1, 4, 1024, 64, 1),
                (1, 4, 1024, 64, 2),
                (1, 4, 1024, 64, 4),
                (1, 4, 1024, 64, 8),
                
                (1, 8, 512, 64, 1),
                (1, 8, 512, 64, 2),
                (1, 8, 512, 64, 4),
                (1, 8, 512, 64, 8),
                
                (1, 8, 1024, 64, 1),
                (1, 8, 1024, 64, 2),
                (1, 8, 1024, 64, 4),
                (1, 8, 1024, 64, 8),
                
                # Slightly larger sequences that still work
                (1, 8, 1536, 64, 1),
                (1, 8, 1536, 64, 2),
                (1, 8, 1536, 64, 4),
                (1, 8, 1536, 64, 8),
                
                (1, 4, 1536, 64, 1),
                (1, 4, 1536, 64, 2),
                (1, 4, 1536, 64, 4),
                (1, 4, 1536, 64, 8),
                
                # Different head counts
                (1, 2, 1024, 64, 4),
                (1, 6, 1024, 64, 4),
                (1, 12, 1024, 64, 4),
                (1, 12, 1536, 64, 4),
                
                # Small batch sizes
                (2, 4, 1024, 64, 4),
                (2, 8, 512, 64, 4),
                (2, 4, 1536, 64, 4),
            ]
    
    elif gpu_type == "w7800":
        # W7800 configurations - conservative parameters for 30GB memory
        if quick:
            configs = [
                # Quick test configurations for W7800
                (1, 8, 1024, 64, 2),
                (1, 12, 2048, 64, 4),
                (1, 12, 4096, 64, 8),
            ]
        else:
            configs = [
                # Conservative configurations for W7800
                (1, 8, 1024, 64, 1),
                (1, 8, 1024, 64, 2),
                (1, 8, 1024, 64, 4),
                (1, 8, 1024, 64, 8),
                
                (1, 8, 2048, 64, 1),
                (1, 8, 2048, 64, 2),
                (1, 8, 2048, 64, 4),
                (1, 8, 2048, 64, 8),
                
                (1, 8, 4096, 64, 1),
                (1, 8, 4096, 64, 2),
                (1, 8, 4096, 64, 4),
                (1, 8, 4096, 64, 8),
                
                (1, 8, 8192, 64, 1),
                (1, 8, 8192, 64, 2),
                (1, 8, 8192, 64, 4),
                (1, 8, 8192, 64, 8),
                
                # Different head counts
                (1, 4, 4096, 64, 4),
                (1, 12, 4096, 64, 4),
                (1, 16, 4096, 64, 4),
                
                # Different head dimensions
                (1, 8, 4096, 128, 4),
                
                # Small batch sizes
                (2, 8, 4096, 64, 4),
            ]
    
    else:
        # Generic ROCm configurations - balanced approach
        if quick:
            configs = [
                # Quick test configurations for generic ROCm
                (1, 12, 1024, 64, 2),
                (1, 12, 2048, 64, 4),
                (1, 12, 4096, 64, 8),
            ]
        else:
            configs = [
                # Balanced configurations for generic ROCm
                (1, 12, 1024, 64, 1),
                (1, 12, 1024, 64, 2),
                (1, 12, 1024, 64, 4),
                (1, 12, 1024, 64, 8),
                
                (1, 12, 2048, 64, 1),
                (1, 12, 2048, 64, 2),
                (1, 12, 2048, 64, 4),
                (1, 12, 2048, 64, 8),
                
                (1, 12, 4096, 64, 1),
                (1, 12, 4096, 64, 2),
                (1, 12, 4096, 64, 4),
                (1, 12, 4096, 64, 8),
                
                (1, 12, 8192, 64, 1),
                (1, 12, 8192, 64, 2),
                (1, 12, 8192, 64, 4),
                (1, 12, 8192, 64, 8),
                
                # Different head counts
                (1, 8, 4096, 64, 4),
                (1, 16, 4096, 64, 4),
                (1, 24, 4096, 64, 4),
                
                # Different head dimensions
                (1, 12, 4096, 128, 4),
            ]
    
    # Filter configurations based on GPU memory limits
    validated_configs = []
    for config in configs:
        batch, head, seq_len, headdim, topk = config
        if validate_config_for_gpu(batch, head, seq_len, headdim, gpu_type):
            validated_configs.append(config)
        else:
            print(f"Skipping config {config} - exceeds memory limits for {gpu_type}")
    
    return validated_configs

def create_input_tensors(batch, head, seq_len, headdim):
    """Create random input tensors for attention."""
    q = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    return q, k, v

def create_block_map_and_sizes(batch, head, seq_len, topk, device="cuda"):
    """Create block map and variable block sizes for VSA."""
    num_q_blocks = seq_len // BLOCK_M
    num_kv_blocks = seq_len // BLOCK_N
    
    # Create random block map
    block_map = torch.zeros(batch, head, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device)
    
    # For each batch and head, create sparse pattern
    for b in range(batch):
        for h in range(head):
            scores = torch.rand(num_q_blocks, num_kv_blocks, device=device)
            _, topk_indices = torch.topk(scores, min(topk, num_kv_blocks), dim=-1)
            for q_idx in range(num_q_blocks):
                kv_indices = topk_indices[q_idx]
                block_map[b, h, q_idx, kv_indices] = True
    
    variable_block_sizes = torch.full((num_kv_blocks,), BLOCK_N, dtype=torch.int32, device=device)
    return block_map, variable_block_sizes

def benchmark_configuration(batch, head, seq_len, headdim, topk, num_runs=5):
    """Benchmark a specific configuration."""
    print(f"Benchmarking: batch={batch}, heads={head}, seq_len={seq_len}, head_dim={headdim}, topk={topk}")
    
    # Create tensors
    q, k, v = create_input_tensors(batch, head, seq_len, headdim)
    block_map, variable_block_sizes = create_block_map_and_sizes(batch, head, seq_len, topk)
    
    # Calculate theoretical FLOPs
    flops = 4 * batch * head * headdim * seq_len * seq_len
    
    # Warm-up
    for _ in range(3):
        q_fwd = q.clone().requires_grad_(True)
        k_fwd = k.clone().requires_grad_(True)
        v_fwd = v.clone().requires_grad_(True)
        o, M = block_sparse_attn(q_fwd, k_fwd, v_fwd, block_map, variable_block_sizes)
        grad_output = torch.randn_like(o)
        o.backward(grad_output)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        q_fwd = q.clone().requires_grad_(True)
        k_fwd = k.clone().requires_grad_(True)
        v_fwd = v.clone().requires_grad_(True)
        o, M = block_sparse_attn(q_fwd, k_fwd, v_fwd, block_map, variable_block_sizes)
        grad_output = torch.randn_like(o)
        o.backward(grad_output)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # Calculate TFLOPS
    total_flops = flops + 2.5 * flops  # Forward + backward
    tflops = total_flops / avg_time * 1e-12
    
    # Calculate sparsity
    sparsity = 1.0 - (topk * (seq_len // BLOCK_M)) / (seq_len // BLOCK_N * seq_len // BLOCK_M)
    
    return {
        'batch': batch,
        'heads': head,
        'seq_len': seq_len,
        'head_dim': headdim,
        'topk': topk,
        'sparsity': sparsity,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'tflops': tflops,
        'flops': total_flops
    }

def main():
    parser = argparse.ArgumentParser(description='Comprehensive VSA Benchmark for ROCm')
    parser.add_argument('--output', type=str, default='vsa_benchmark_results.json', help='Output file for results')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark with fewer configurations')
    parser.add_argument('--gpu-type', type=str, choices=['mi300x', 'mi250', 'mi210', 'w7800', 'auto'], default='auto', 
                       help='Force specific GPU type (auto-detects if not specified)')
    args = parser.parse_args()
    
    set_seed(42)
    
    print("Video Sparse Attention Comprehensive Benchmark for ROCm")
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
    print(f"Triton Version: {triton.__version__}")
    
    # Show GPU-specific optimization info
    if gpu_type == "mi300x":
        print("✅ Using MI300X-optimized configurations (192GB memory, extremely aggressive parameters)")
    elif gpu_type == "mi250":
        print("✅ Using MI250-optimized configurations (128GB memory, aggressive parameters)")
    elif gpu_type == "mi210":
        print("✅ Using MI210-optimized configurations (64KB shared memory, very conservative parameters)")
    elif gpu_type == "w7800":
        print("✅ Using W7800-optimized configurations (30GB memory, conservative parameters)")
    else:
        print("✅ Using generic ROCm configurations (balanced parameters)")
    print()
    
    # Generate GPU-specific configurations
    configs = generate_gpu_specific_configs(gpu_type, args.quick)
    
    print(f"Testing {len(configs)} configurations...")
    print()
    
    results = []
    
    for i, (batch, head, seq_len, headdim, topk) in enumerate(configs):
        try:
            result = benchmark_configuration(batch, head, seq_len, headdim, topk)
            results.append(result)
            print(f"✓ [{i+1:2d}/{len(configs)}] {result['tflops']:.2f} TFLOPS (sparsity: {result['sparsity']:.1%})")
        except Exception as e:
            print(f"✗ [{i+1:2d}/{len(configs)}] Failed: {e}")
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
        print(f"Total configurations tested: {len(results)}")
        print(f"Best performance: {results[0]['tflops']:.2f} TFLOPS")
        print(f"  - Config: batch={results[0]['batch']}, heads={results[0]['heads']}, seq_len={results[0]['seq_len']}, head_dim={results[0]['head_dim']}, topk={results[0]['topk']}")
        print(f"  - Sparsity: {results[0]['sparsity']:.1%}")
        
        # Show configuration ranges used
        if results:
            batch_sizes = sorted(set(r['batch'] for r in results))
            num_heads = sorted(set(r['heads'] for r in results))
            seq_lens = sorted(set(r['seq_len'] for r in results))
            head_dims = sorted(set(r['head_dim'] for r in results))
            topk_vals = sorted(set(r['topk'] for r in results))
            
            print(f"\nConfiguration ranges tested:")
            print(f"  Batch sizes: {batch_sizes}")
            print(f"  Number of heads: {num_heads}")
            print(f"  Sequence lengths: {seq_lens}")
            print(f"  Head dimensions: {head_dims}")
            print(f"  Top-k values: {topk_vals}")
        
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
            print(f"  {seq_len:4d}: {max(perfs):6.2f} TFLOPS (max), {np.mean(perfs):6.2f} TFLOPS (avg)")
        
        # Performance by sparsity
        print("\nPerformance by sparsity level:")
        sparsity_perf = {}
        for r in results:
            sparsity_range = f"{int(r['sparsity']*100)//10*10}-{int(r['sparsity']*100)//10*10+9}%"
            if sparsity_range not in sparsity_perf:
                sparsity_perf[sparsity_range] = []
            sparsity_perf[sparsity_range].append(r['tflops'])
        
        for sparsity_range in sorted(sparsity_perf.keys()):
            perfs = sparsity_perf[sparsity_range]
            print(f"  {sparsity_range:>8}: {max(perfs):6.2f} TFLOPS (max), {np.mean(perfs):6.2f} TFLOPS (avg)")
        
        # GPU-specific insights
        print(f"\nGPU-Specific Performance Insights:")
        if gpu_type == "mi300x":
            print("  ✅ MI300X: Massive memory capacity (192GB) enables testing of ultra-long sequences and very large batch sizes")
            print("  ✅ Optimal for: Ultra-long video sequences, massive models, high-throughput processing, research applications")
        elif gpu_type == "mi250":
            print("  ✅ MI250: Large memory capacity enables testing of very long sequences and large batch sizes")
            print("  ✅ Optimal for: Long video sequences, high-throughput processing, large models")
        elif gpu_type == "mi210":
            print("  ✅ MI210: Limited shared memory (64KB) requires very conservative configurations")
            print("  ✅ Optimal for: Short to medium video sequences, memory-constrained applications")
        elif gpu_type == "w7800":
            print("  ✅ W7800: Conservative memory usage optimized for professional workloads")
            print("  ✅ Optimal for: Standard video processing, moderate sequence lengths, efficient resource usage")
        else:
            print("  ✅ Generic ROCm: Balanced configurations for various ROCm-compatible GPUs")
            print("  ✅ Optimal for: General-purpose video processing, compatibility testing")
    
    print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()