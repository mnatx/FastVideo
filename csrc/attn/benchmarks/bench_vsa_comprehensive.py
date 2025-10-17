#!/usr/bin/env python3
"""
Comprehensive Video Sparse Attention Benchmark for ROCm Platform
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
    parser = argparse.ArgumentParser(description='Comprehensive VSA Benchmark')
    parser.add_argument('--output', type=str, default='vsa_benchmark_results.json', help='Output file for results')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark with fewer configurations')
    args = parser.parse_args()
    
    set_seed(42)
    
    print("Video Sparse Attention Comprehensive Benchmark for ROCm")
    print("=" * 60)
    
    # Get device info
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"ROCm Version: {torch.version.hip}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Triton Version: {triton.__version__}")
    print()
    
    # Define test configurations
    if args.quick:
        configs = [
            # (batch, heads, seq_len, head_dim, topk)
            (1, 12, 1024, 64, 1),
            (1, 12, 2048, 64, 2),
            (1, 12, 4096, 64, 4),
        ]
    else:
        configs = [
            # Different sequence lengths
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
            
            # Different head dimensions (if supported)
            (1, 8, 2048, 64, 2),
            (1, 16, 2048, 64, 2),
            (1, 24, 2048, 64, 2),
        ]
    
    results = []
    
    for i, (batch, head, seq_len, headdim, topk) in enumerate(configs):
        try:
            result = benchmark_configuration(batch, head, seq_len, headdim, topk)
            results.append(result)
            print(f"✓ {result['tflops']:.2f} TFLOPS (sparsity: {result['sparsity']:.1%})")
        except Exception as e:
            print(f"✗ Failed: {e}")
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
        
        print(f"Total configurations tested: {len(results)}")
        print(f"Best performance: {results[0]['tflops']:.2f} TFLOPS")
        print(f"  - Config: batch={results[0]['batch']}, heads={results[0]['heads']}, seq_len={results[0]['seq_len']}, head_dim={results[0]['head_dim']}, topk={results[0]['topk']}")
        print(f"  - Sparsity: {results[0]['sparsity']:.1%}")
        
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
    
    print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()