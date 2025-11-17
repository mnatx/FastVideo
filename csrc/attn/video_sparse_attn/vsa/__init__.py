import torch
from typing import Tuple
import warnings
block_sparse_attn=None
major, minor = torch.cuda.get_device_capability(0)
is_h100 = major == 9 and minor == 0
is_rocm = torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip is not None
if is_h100 and not is_rocm:# check if H100
    from vsa_cuda import block_sparse_fwd, block_sparse_bwd
    from vsa.block_sparse_wrapper import block_sparse_attn_SM90
    block_sparse_attn = block_sparse_attn_SM90
else:
    from vsa.block_sparse_wrapper import block_sparse_attn_triton
    block_sparse_fwd = None
    block_sparse_bwd = None
    block_sparse_attn = block_sparse_attn_triton

BLOCK_M = 64
BLOCK_N = 64


def torch_attention(q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1)**0.5)

    # Causal mask removed since causal is always false

    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)
    return output, QK


def video_sparse_attn(q, k, v, variable_block_sizes, topk, block_size, compress_attn_weight=None):
    """
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_heads, seq_len, head_dim]
    v: [batch_size, num_heads, seq_len, head_dim]
    topk: int
    block_size: int or tuple of 3 ints
    video_shape: tuple of (T, H, W)
    compress_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    select_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    NOTE: We assume q, k, v is zero padded!!
    V1 of sparse attention. Include compress attn and sparse attn branch, use average pooling to compress. 
    Assume q, k, v is flattened in this way: [batch_size, num_heads, T//block_size[0], H//block_size[1], W//block_size[2], block_size[0], block_size[1], block_size[2]]
    """

    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    assert block_elements == 64
    
    # Ensure tensors are contiguous before reshaping
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Get shapes for all tensors
    batch_size_q, num_heads_q, seq_len_q, head_dim_q = q.shape
    batch_size_k, num_heads_k, seq_len_k, head_dim_k = k.shape
    batch_size_v, num_heads_v, seq_len_v, head_dim_v = v.shape
    
    # Validate that all tensors have compatible shapes
    # For attention, q, k, v should typically have the same batch_size, num_heads, and head_dim
    # But seq_len might differ in some cases (e.g., cross-attention)
    if batch_size_q != batch_size_k or batch_size_q != batch_size_v:
        raise RuntimeError(
            f"Batch size mismatch: q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}"
        )
    if num_heads_q != num_heads_k or num_heads_q != num_heads_v:
        raise RuntimeError(
            f"Number of heads mismatch: q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}"
        )
    if head_dim_q != head_dim_k or head_dim_q != head_dim_v:
        raise RuntimeError(
            f"Head dimension mismatch: q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}"
        )
    
    # Use q's sequence length as the primary reference, but validate each tensor separately
    batch_size = batch_size_q
    num_heads = num_heads_q
    head_dim = head_dim_q
    seq_len = seq_len_q
    
    # Validate each tensor's seq_len is divisible by block_elements
    for tensor_name, tensor_seq_len, tensor_shape in [("q", seq_len_q, q.shape), 
                                                       ("k", seq_len_k, k.shape), 
                                                       ("v", seq_len_v, v.shape)]:
        if tensor_seq_len % block_elements != 0:
            raise RuntimeError(
                f"{tensor_name} tensor seq_len ({tensor_seq_len}) must be divisible by "
                f"block_elements ({block_elements}). Tensor shape: {tensor_shape}"
            )
    
    # compress attn - use reshape instead of view for better error handling
    # Use each tensor's own sequence length for reshaping
    num_blocks_q = seq_len_q // block_elements
    num_blocks_k = seq_len_k // block_elements
    num_blocks_v = seq_len_v // block_elements
    
    # Adjust variable_block_sizes for each tensor if needed
    # Auto-corrects size mismatches with warnings
    def adjust_variable_block_sizes(vbs, num_blocks_needed, tensor_name):
        if vbs.numel() != num_blocks_needed:
            # Auto-correct: truncate if too large, pad if too small
            if vbs.numel() > num_blocks_needed:
                warnings.warn(
                    f"variable_block_sizes has {vbs.numel()} elements but {tensor_name} tensor needs {num_blocks_needed} blocks. "
                    f"Truncating variable_block_sizes from {vbs.numel()} to {num_blocks_needed}.",
                    UserWarning
                )
                vbs = vbs[:num_blocks_needed]
            elif vbs.numel() < num_blocks_needed:
                warnings.warn(
                    f"variable_block_sizes has {vbs.numel()} elements but {tensor_name} tensor needs {num_blocks_needed} blocks. "
                    f"Padding variable_block_sizes from {vbs.numel()} to {num_blocks_needed} with default value {block_elements}.",
                    UserWarning
                )
                padding = torch.full(
                    (num_blocks_needed - vbs.numel(),),
                    block_elements,
                    dtype=vbs.dtype,
                    device=vbs.device
                )
                vbs = torch.cat([vbs, padding])
        return vbs
    
    # Adjust variable_block_sizes for q (primary reference)
    variable_block_sizes_q = adjust_variable_block_sizes(variable_block_sizes, num_blocks_q, "q")
    
    try:
        target_shape_q = (batch_size, num_heads, num_blocks_q, block_elements, head_dim)
        q_reshaped = q.reshape(*target_shape_q)
        q_compress = (q_reshaped.float().sum(dim=3) / variable_block_sizes_q.view(1, 1, -1, 1)).to(q.dtype)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to reshape q tensor: shape {q.shape} -> {target_shape_q}. "
            f"Tensor has {q.numel()} elements. "
            f"seq_len={seq_len_q}, block_elements={block_elements}, num_blocks={num_blocks_q}, "
            f"variable_block_sizes.shape={variable_block_sizes.shape}. Original error: {e}"
        ) from e
    
    # For k and v, if they have different seq_len, we need to handle them separately
    # But typically in attention, k and v should match q's seq_len
    if seq_len_k != seq_len_q:
        # k has different length - adjust variable_block_sizes for k
        variable_block_sizes_k = adjust_variable_block_sizes(variable_block_sizes, num_blocks_k, "k")
        try:
            target_shape_k = (batch_size, num_heads, num_blocks_k, block_elements, head_dim)
            k_reshaped = k.reshape(*target_shape_k)
            k_compress = (k_reshaped.float().sum(dim=3) / variable_block_sizes_k.view(1, 1, -1, 1)).to(k.dtype)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to reshape k tensor: shape {k.shape} -> {target_shape_k}. "
                f"Tensor has {k.numel()} elements. "
                f"seq_len={seq_len_k}, block_elements={block_elements}, num_blocks={num_blocks_k}, "
                f"variable_block_sizes.shape={variable_block_sizes.shape}. Original error: {e}"
            ) from e
    else:
        # k has same length as q, use same variable_block_sizes
        try:
            target_shape_k = (batch_size, num_heads, num_blocks_k, block_elements, head_dim)
            k_reshaped = k.reshape(*target_shape_k)
            k_compress = (k_reshaped.float().sum(dim=3) / variable_block_sizes_q.view(1, 1, -1, 1)).to(k.dtype)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to reshape k tensor: shape {k.shape} -> {target_shape_k}. "
                f"Tensor has {k.numel()} elements. "
                f"seq_len={seq_len_k}, block_elements={block_elements}, num_blocks={num_blocks_k}, "
                f"variable_block_sizes.shape={variable_block_sizes.shape}. Original error: {e}"
            ) from e
    
    if seq_len_v != seq_len_q:
        # v has different length - adjust variable_block_sizes for v
        variable_block_sizes_v = adjust_variable_block_sizes(variable_block_sizes, num_blocks_v, "v")
        try:
            target_shape_v = (batch_size, num_heads, num_blocks_v, block_elements, head_dim)
            v_reshaped = v.reshape(*target_shape_v)
            v_compress = (v_reshaped.float().sum(dim=3) / variable_block_sizes_v.view(1, 1, -1, 1)).to(v.dtype)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to reshape v tensor: shape {v.shape} -> {target_shape_v}. "
                f"Tensor has {v.numel()} elements. "
                f"seq_len={seq_len_v}, block_elements={block_elements}, num_blocks={num_blocks_v}, "
                f"variable_block_sizes.shape={variable_block_sizes.shape}. Original error: {e}"
            ) from e
    else:
        # v has same length as q, use same variable_block_sizes
        try:
            target_shape_v = (batch_size, num_heads, num_blocks_v, block_elements, head_dim)
            v_reshaped = v.reshape(*target_shape_v)
            v_compress = (v_reshaped.float().sum(dim=3) / variable_block_sizes_q.view(1, 1, -1, 1)).to(v.dtype)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to reshape v tensor: shape {v.shape} -> {target_shape_v}. "
                f"Tensor has {v.numel()} elements. "
                f"seq_len={seq_len_v}, block_elements={block_elements}, num_blocks={num_blocks_v}, "
                f"variable_block_sizes.shape={variable_block_sizes.shape}. Original error: {e}"
            ) from e

    output_compress, block_attn_score = torch_attention(q_compress, k_compress,
                                                        v_compress)

    # output_compress shape should match q_compress, which is based on q's sequence length
    output_compress = output_compress.reshape(batch_size, num_heads,
                                               num_blocks_q, 1,
                                               head_dim)
    output_compress = output_compress.repeat(1, 1, 1, block_elements,
                                             1).reshape(batch_size, num_heads,
                                                        seq_len_q, head_dim)

    # block_attn_score has shape [batch, heads, num_blocks_q, num_blocks_k]
    # We select topk along the last dimension (num_blocks_k)
    # Clamp topk to be at most the size of the last dimension to avoid "k out of range" error
    num_blocks_k_actual = block_attn_score.shape[-1]
    topk_clamped = max(1, min(topk, num_blocks_k_actual))  # Ensure at least 1
    
    if topk_clamped < topk:
        warnings.warn(
            f"topk ({topk}) is larger than the number of key blocks ({num_blocks_k_actual}). "
            f"Clamping topk to {topk_clamped}.",
            UserWarning
        )
    
    topK_indices = torch.topk(block_attn_score, topk_clamped, dim=-1).indices
    block_mask = torch.zeros_like(block_attn_score, dtype=torch.bool).scatter_(-1, topK_indices, True)
    output_select, _ = block_sparse_attn(q, k, v, block_mask, variable_block_sizes)

    if compress_attn_weight is not None:
        final_output = output_compress * compress_attn_weight + output_select
    else:
        final_output = output_compress + output_select
    return final_output

