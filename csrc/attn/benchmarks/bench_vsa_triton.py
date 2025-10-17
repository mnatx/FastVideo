import torch
import argparse
import triton.testing
from vsa import block_sparse_attn
from vsa import BLOCK_M, BLOCK_N

import numpy as np
import random

def set_seed(seed: int = 42):
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark Block Sparse Attention')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--topk', type=int, default=None, help='Number of kv blocks each q block attends to')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[49152], help='Sequence lengths to benchmark')
    return parser.parse_args()

def create_input_tensors(batch, head, seq_len, headdim):
    """Create random input tensors for attention."""
    q = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    return q, k, v

def create_block_map_and_sizes(batch, head, seq_len, topk, device="cuda"):
    """
    Create block map and variable block sizes for VSA.
    
    Args:
        batch: batch size
        head: number of heads
        seq_len: sequence length
        topk: number of kv blocks each q block attends to
        device: device to create tensors on
        
    Returns:
        block_map: [batch, head, num_q_blocks, num_kv_blocks] binary mask
        variable_block_sizes: [num_kv_blocks] block sizes
    """
    num_q_blocks = seq_len // BLOCK_M
    num_kv_blocks = seq_len // BLOCK_N
    
    # Create random block map
    block_map = torch.zeros(batch, head, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device)
    
    # For each batch and head, create sparse pattern
    for b in range(batch):
        for h in range(head):
            # Create random scores for each q block
            scores = torch.rand(num_q_blocks, num_kv_blocks, device=device)
            # Get top-k indices for each q block
            _, topk_indices = torch.topk(scores, min(topk, num_kv_blocks), dim=-1)
            # Set the mask
            for q_idx in range(num_q_blocks):
                kv_indices = topk_indices[q_idx]
                block_map[b, h, q_idx, kv_indices] = True
    
    # Create variable block sizes (all blocks are full size for now)
    variable_block_sizes = torch.full((num_kv_blocks,), BLOCK_N, dtype=torch.int32, device=device)
    
    return block_map, variable_block_sizes

def benchmark_block_sparse_attention(q, k, v, block_map, variable_block_sizes, flops):
    """Benchmark block sparse attention forward+backward pass."""
    print("\n=== BLOCK SPARSE ATTENTION FORWARD+BACKWARD BENCHMARK ===")
    
    # Combined forward+backward pass
    # Warm-up run
    q_fwd = q.clone().requires_grad_(True)
    k_fwd = k.clone().requires_grad_(True)
    v_fwd = v.clone().requires_grad_(True)
    o, M = block_sparse_attn(q_fwd, k_fwd, v_fwd, block_map, variable_block_sizes)
    grad_output = torch.randn_like(o)
    o.backward(grad_output)
    torch.cuda.synchronize()
    
    # Benchmark forward+backward
    def forward_backward_fn():
        q_fwd = q.clone().requires_grad_(True)
        k_fwd = k.clone().requires_grad_(True)
        v_fwd = v.clone().requires_grad_(True)
        o, M = block_sparse_attn(q_fwd, k_fwd, v_fwd, block_map, variable_block_sizes)
        grad_output = torch.randn_like(o)
        o.backward(grad_output)
    
    total_time = triton.testing.do_bench(
        forward_backward_fn,
        warmup=25,
        rep=100,
        return_mode='mean'
    )
    
    # Total flops for forward + backward (forward + 2.5x backward approximation)
    total_flops = flops + 2.5 * flops  # 3.5x the forward flops
    sparse_tflops = total_flops / total_time * 1e-12 * 1e3
    print(f"Block Sparse Forward+Backward - TFLOPS: {sparse_tflops:.2f}")
    
    return sparse_tflops

def main():
    args = parse_arguments()

    set_seed(42)
    
    # Extract parameters
    batch = args.batch_size
    head = args.num_heads
    headdim = args.head_dim
    
    print(f"Block Sparse Attention Benchmark")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}")
    
    # Test with different sequence lengths
    for seq_len in args.seq_lengths:
        # Skip very long sequences if they might cause OOM
        if seq_len > 16384 and batch > 1:
            continue
            
        print("="*100)
        print(f"\nSequence length: {seq_len}")
        
        # Calculate theoretical FLOPs for attention
        flops = 4 * batch * head * headdim * seq_len * seq_len
        
        # Create input tensors
        q, k, v = create_input_tensors(batch, head, seq_len, headdim)
        
        # Setup block sparse parameters
        num_q_blocks = seq_len // BLOCK_M
        num_kv_blocks = seq_len // BLOCK_N
        
        # Determine k value (number of kv blocks per q block)
        topk = args.topk
        if topk is None:
            topk = num_kv_blocks // 10  # Default to ~90% sparsity if k is not specified
        topk = max(1, topk)       
        print(f"Using topk={topk} kv blocks per q block (out of {num_kv_blocks} total kv blocks)")
        
        # Generate block map and variable block sizes
        block_map, variable_block_sizes = create_block_map_and_sizes(
            batch, head, seq_len, topk, device="cuda")
        
        # Benchmark block sparse attention
        sparse_fwd = benchmark_block_sparse_attention(
            q, k, v, block_map, variable_block_sizes, flops
        )
        
        # Print results
        print("\n=== PERFORMANCE RESULTS ===")
        print(f"Block Sparse Forward+Backward - TFLOPS: {sparse_fwd:.2f}")

if __name__ == "__main__":
    main()