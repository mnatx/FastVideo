# SPDX-License-Identifier: Apache-2.0
import functools
import math
from dataclasses import dataclass

import torch

try:
    from vsa import video_sparse_attn
except ImportError:
    try:
        from vsa.block_sparse_wrapper import video_sparse_attn
    except ImportError:
        video_sparse_attn = None

from typing import Any

from fastvideo.attention.backends.abstract import (AttentionBackend,
                                                   AttentionImpl,
                                                   AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.distributed import get_sp_group
from fastvideo.logger import init_logger

logger = init_logger(__name__)
VSA_TILE_SIZE = (4, 4, 4)


@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device,
                           dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(indices[t * ts:min(t * ts + ts, T),
                                  h * hs:min(h * hs + hs, H),
                                  w * ws:min(w * ws + ws, W)].flatten())
    index = torch.cat(ls, dim=0)
    return index


@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    return torch.argsort(
        get_tile_partition_indices(dit_seq_shape, tile_size, device))


@functools.lru_cache(maxsize=10)
def construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    num_tiles: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """
    Compute the number of valid (non‑padded) tokens inside every
    (ts_t × ts_h × ts_w) tile after padding ‑‑ flattened in the order
    (t‑tile, h‑tile, w‑tile) that `rearrange` uses.

    Returns
    -------
    torch.LongTensor  # shape: [∏ full_window_size]
    """
    # unpack
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = VSA_TILE_SIZE
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.LongTensor:
        """Vector with the size of each tile along one dimension."""
        sizes = torch.full((n_tiles, ), tile, dtype=torch.int, device=device)
        # size of last (possibly partial) tile
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)  # [n_t]
    h_sizes = _sizes(h, ts_h, n_h)  # [n_h]
    w_sizes = _sizes(w, ts_w, n_w)  # [n_w]

    # broadcast‑multiply to get voxels per tile, then flatten
    block_sizes = (
        t_sizes[:, None, None]  # [n_t, 1,   1]
        * h_sizes[None, :, None]  # [1,   n_h, 1]
        * w_sizes[None, None, :]  # [1,   1,   n_w]
    ).reshape(-1)  # [n_t * n_h * n_w]

    return block_sizes


@functools.lru_cache(maxsize=10)
def get_non_pad_index(
    variable_block_sizes: torch.LongTensor,
    max_block_size: int,
):
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size,
                                                   device=device)[None, :]
    index_mask = torch.arange(
        max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


class VideoSparseAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]
    
    @staticmethod
    def is_available() -> bool:
        """Check if VSA is available on the current platform."""
        if video_sparse_attn is None:
            return False
        
        # Check if we're on a supported platform (CUDA or ROCm)
        if not torch.cuda.is_available():
            return False
            
        # For ROCm, we need to check if Triton is available
        try:
            import triton
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "VIDEO_SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["VideoSparseAttentionImpl"]:
        return VideoSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["VideoSparseAttentionMetadata"]:
        return VideoSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["VideoSparseAttentionMetadataBuilder"]:
        return VideoSparseAttentionMetadataBuilder


@dataclass
class VideoSparseAttentionMetadata(AttentionMetadata):
    current_timestep: int
    dit_seq_shape: list[int]
    VSA_sparsity: float
    num_tiles: list[int]
    total_seq_length: int
    tile_partition_indices: torch.LongTensor
    reverse_tile_partition_indices: torch.LongTensor
    variable_block_sizes: torch.LongTensor
    non_pad_index: torch.LongTensor


class VideoSparseAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(  # type: ignore
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        device: torch.device,
        **kwargs: dict[str, Any],
    ) -> VideoSparseAttentionMetadata:
        # Ensure we're on a supported device
        if not device.type in ['cuda', 'hip']:
            raise ValueError(f"VSA requires CUDA or ROCm device, got {device.type}")
        patch_size = patch_size
        dit_seq_shape = (raw_latent_shape[0] // patch_size[0],
                         raw_latent_shape[1] // patch_size[1],
                         raw_latent_shape[2] // patch_size[2])

        num_tiles = (math.ceil(dit_seq_shape[0] / VSA_TILE_SIZE[0]),
                     math.ceil(dit_seq_shape[1] / VSA_TILE_SIZE[1]),
                     math.ceil(dit_seq_shape[2] / VSA_TILE_SIZE[2]))
        total_seq_length = math.prod(dit_seq_shape)

        tile_partition_indices = get_tile_partition_indices(
            dit_seq_shape, VSA_TILE_SIZE, device)
        reverse_tile_partition_indices = get_reverse_tile_partition_indices(
            dit_seq_shape, VSA_TILE_SIZE, device)
        variable_block_sizes = construct_variable_block_sizes(
            dit_seq_shape, num_tiles, device)
        non_pad_index = get_non_pad_index(variable_block_sizes,
                                          math.prod(VSA_TILE_SIZE))

        return VideoSparseAttentionMetadata(
            current_timestep=current_timestep,
            dit_seq_shape=dit_seq_shape,  # type: ignore
            VSA_sparsity=VSA_sparsity,  # type: ignore
            num_tiles=num_tiles,  # type: ignore
            total_seq_length=total_seq_length,  # type: ignore
            tile_partition_indices=tile_partition_indices,  # type: ignore
            reverse_tile_partition_indices=reverse_tile_partition_indices,
            variable_block_sizes=variable_block_sizes,
            non_pad_index=non_pad_index)


class VideoSparseAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        self.softmax_scale = softmax_scale
        self.head_size = head_size
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size

    def tile(self, x: torch.Tensor, num_tiles: list[int],
             tile_partition_indices: torch.LongTensor,
             non_pad_index: torch.LongTensor) -> torch.Tensor:
        t_padded_size = num_tiles[0] * VSA_TILE_SIZE[0]
        h_padded_size = num_tiles[1] * VSA_TILE_SIZE[1]
        w_padded_size = num_tiles[2] * VSA_TILE_SIZE[2]
        
        expected_seq_len = tile_partition_indices.shape[0]
        actual_seq_len = x.shape[1]
        
        # Validate that the sequence length matches what the indices expect
        if actual_seq_len != expected_seq_len:
            raise RuntimeError(
                f"Sequence length mismatch in tile(): "
                f"expected {expected_seq_len} (from tile_partition_indices), "
                f"got {actual_seq_len} (from x.shape[1]). "
                f"x.shape={x.shape}. "
                f"This may indicate an issue with sequence parallelism or tiling indices computation."
            )

        x_padded = torch.zeros(
            (x.shape[0], t_padded_size * h_padded_size * w_padded_size,
             x.shape[-2], x.shape[-1]),
            device=x.device,
            dtype=x.dtype)
        x_padded[:, non_pad_index] = x[:, tile_partition_indices]
        return x_padded

    def untile(self, x: torch.Tensor,
               reverse_tile_partition_indices: torch.LongTensor,
               non_pad_index: torch.LongTensor) -> torch.Tensor:
        x = x[:, non_pad_index][:, reverse_tile_partition_indices]
        return x

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        # qkv has shape [3 or 4, seq_len, num_heads, head_dim] when stacked
        # The first dimension is the number of tensors (q, k, v, and optionally gate_compress)
        # The tile() function will apply the same tiling to all tensors in the stack
        num_tensors = qkv.shape[0]
        seq_len = qkv.shape[1]
        expected_seq_len = attn_metadata.total_seq_length
        
        # Account for sequence parallelism: after all_to_all_4D with scatter_dim=2, gather_dim=1,
        # the sequence length may be multiplied by sp_size (number of heads per rank)
        # When sp_size=1, seq_len should equal expected_seq_len
        # When sp_size > 1, seq_len = expected_seq_len * sp_size (approximately)
        if seq_len != expected_seq_len:
            # Check if this is due to sequence parallelism
            if self.sp_size > 1 and seq_len == expected_seq_len * self.sp_size:
                # Sequence parallelism case: tile each chunk separately
                # Split the sequence into sp_size chunks, tile each, then concatenate
                chunk_size = expected_seq_len
                qkv_chunks = []
                for i in range(self.sp_size):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, seq_len)
                    chunk = qkv[:, start_idx:end_idx, :, :]
                    # Pad chunk to expected_seq_len if needed (for last chunk)
                    if chunk.shape[1] < expected_seq_len:
                        padding = torch.zeros(
                            (num_tensors, expected_seq_len - chunk.shape[1], chunk.shape[2], chunk.shape[3]),
                            device=chunk.device,
                            dtype=chunk.dtype
                        )
                        chunk = torch.cat([chunk, padding], dim=1)
                    tiled_chunk = self.tile(chunk, attn_metadata.num_tiles,
                                          attn_metadata.tile_partition_indices,
                                          attn_metadata.non_pad_index)
                    qkv_chunks.append(tiled_chunk)
                # Concatenate along sequence dimension
                return torch.cat(qkv_chunks, dim=1)
            else:
                # Unexpected sequence length mismatch
                logger.error(
                    f"Sequence length mismatch in preprocess_qkv: "
                    f"expected {expected_seq_len}, got {seq_len}, sp_size={self.sp_size}. "
                    f"qkv.shape={qkv.shape}. "
                    f"This may indicate an issue with all_to_all_4D or sequence parallelism configuration."
                )
                # Try to tile anyway - this will raise an error in tile() if it fails
                return self.tile(qkv, attn_metadata.num_tiles,
                               attn_metadata.tile_partition_indices,
                               attn_metadata.non_pad_index)
        
        # Normal case: sequence length matches expected (sp_size=1)
        return self.tile(qkv, attn_metadata.num_tiles,
                         attn_metadata.tile_partition_indices,
                         attn_metadata.non_pad_index)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output, attn_metadata.reverse_tile_partition_indices,
                           attn_metadata.non_pad_index)

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
        gate_compress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        if gate_compress is not None:
            gate_compress = gate_compress.transpose(1, 2).contiguous()

        VSA_sparsity = attn_metadata.VSA_sparsity

        cur_topk = math.ceil(
            (1 - VSA_sparsity) *
            (attn_metadata.total_seq_length / math.prod(VSA_TILE_SIZE)))

        if video_sparse_attn is None:
            raise NotImplementedError("video_sparse_attn is not installed")
        
        # Account for sequence parallelism: when sequence parallelism is active,
        # the sequence length may be multiplied by sp_size after all_to_all_4D.
        # We need to ensure variable_block_sizes matches the actual sequence length.
        # Check all three tensors to ensure they have consistent shapes
        variable_block_sizes = attn_metadata.variable_block_sizes
        block_elements = math.prod(VSA_TILE_SIZE)
        
        # Get sequence lengths from all tensors after transpose
        # Shape after transpose: [batch_size, num_heads, seq_len, head_dim]
        seq_len_q = query.shape[2]
        seq_len_k = key.shape[2]
        seq_len_v = value.shape[2]
        
        # Check if this is cross-attention (q and k/v have different sequence lengths)
        # In cross-attention, q comes from video latents (may be tiled) and k/v come from 
        # text encoder (typically not tiled). VSA tiling is designed for video latents.
        is_cross_attention = (seq_len_q != seq_len_k) or (seq_len_q != seq_len_v)
        
        if is_cross_attention:
            # Cross-attention: k and v should have the same length, but q can be different
            if seq_len_k != seq_len_v:
                raise RuntimeError(
                    f"Sequence length mismatch in cross-attention: "
                    f"k.shape={key.shape}, v.shape={value.shape} must match. "
                    f"q.shape={query.shape} can differ."
                )
            
            # Check if q is tiled (sequence length is much larger than k/v)
            # If q is tiled but k/v are not, VSA (both CUDA and Triton) won't work correctly
            # In this case, fall back to standard PyTorch attention
            if seq_len_q > seq_len_k * 10:  # Heuristic: q is likely tiled if much larger
                # logger.info(
                #     f"Cross-attention detected with tiled q and untiled k/v: "
                #     f"q.shape={query.shape} (seq_len={seq_len_q}, likely tiled), "
                #     f"k.shape={key.shape} (seq_len={seq_len_k}, not tiled), "
                #     f"v.shape={value.shape} (seq_len={seq_len_v}, not tiled). "
                #     f"Falling back to standard PyTorch attention (SDPA) for cross-attention."
                # )
                # Fall back to standard PyTorch scaled dot-product attention
                # This can handle cross-attention with different sequence lengths
                # Note: We need to handle the case where q might be tiled - we'll use
                # standard attention which will work correctly even if q has a different shape
                # Use softmax_scale if provided, otherwise use default (1/sqrt(head_dim))
                scale = self.softmax_scale if self.softmax_scale is not None else (1.0 / math.sqrt(self.head_size))
                attn_kwargs = {
                    "attn_mask": None,
                    "dropout_p": 0.0,
                    "is_causal": False,  # Cross-attention is not causal
                    "scale": scale
                }
                if query.shape[1] != key.shape[1]:
                    attn_kwargs["enable_gqa"] = True
                output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, **attn_kwargs)
                # Transpose back to [batch_size, seq_len, num_heads, head_dim]
                output = output.transpose(1, 2)
                return output
            else:
                # Cross-attention but q is not heavily tiled - might still work with VSA
                # but log a warning
                logger.warning(
                    f"Cross-attention detected with VSA backend: "
                    f"q.shape={query.shape} (seq_len={seq_len_q}), "
                    f"k.shape={key.shape} (seq_len={seq_len_k}), "
                    f"v.shape={value.shape} (seq_len={seq_len_v}). "
                    f"Attempting to use VSA, but this may not work correctly."
                )
                actual_seq_len = seq_len_q
        else:
            # Self-attention: all tensors should have the same sequence length
            actual_seq_len = seq_len_q
        expected_num_blocks = variable_block_sizes.shape[0]
        expected_seq_len = expected_num_blocks * block_elements
        
        # Validate that sequence length is divisible by block_elements
        if actual_seq_len % block_elements != 0:
            raise RuntimeError(
                f"Sequence length {actual_seq_len} is not divisible by block_elements {block_elements}. "
                f"q.shape={query.shape}, k.shape={key.shape}, v.shape={value.shape}. "
                f"This indicates a mismatch in tiling or padding."
            )
        
        # Calculate the actual number of blocks needed
        actual_num_blocks = actual_seq_len // block_elements
        
        # Check if actual sequence length matches what variable_block_sizes expects
        if actual_seq_len != expected_seq_len:
            logger.debug(
                f"Sequence length mismatch detected: actual_seq_len={actual_seq_len} "
                f"({actual_num_blocks} blocks), expected_seq_len={expected_seq_len} "
                f"({expected_num_blocks} blocks), sp_size={self.sp_size}. "
                f"Adjusting variable_block_sizes."
            )
            
            # Adjust variable_block_sizes to match actual_num_blocks
            if actual_num_blocks == expected_num_blocks:
                # Sizes match, no adjustment needed
                pass
            elif actual_num_blocks > expected_num_blocks:
                # Need to repeat variable_block_sizes
                if actual_num_blocks % expected_num_blocks == 0:
                    # Evenly divisible - repeat the pattern
                    repeat_factor = actual_num_blocks // expected_num_blocks
                    variable_block_sizes = variable_block_sizes.repeat(repeat_factor)
                    logger.info(
                        f"Repeated variable_block_sizes {repeat_factor}x: "
                        f"{expected_num_blocks} -> {actual_num_blocks} blocks"
                    )
                else:
                    # Not evenly divisible - repeat and truncate
                    repeat_factor = (actual_num_blocks + expected_num_blocks - 1) // expected_num_blocks
                    variable_block_sizes = variable_block_sizes.repeat(repeat_factor)[:actual_num_blocks]
                    logger.warning(
                        f"variable_block_sizes adjusted (not evenly divisible): "
                        f"expected {expected_num_blocks} blocks, got {actual_num_blocks} blocks "
                        f"(sp_size={self.sp_size}). Repeated {repeat_factor}x and truncated."
                    )
            else:
                # actual_num_blocks < expected_num_blocks - truncate
                variable_block_sizes = variable_block_sizes[:actual_num_blocks]
                logger.warning(
                    f"variable_block_sizes truncated: expected {expected_num_blocks} blocks, "
                    f"got {actual_num_blocks} blocks (sp_size={self.sp_size}). "
                    f"Truncated to match actual sequence length."
                )
        
        # Final validation: ensure variable_block_sizes matches the actual sequence length
        # This is critical because the old VSA code may calculate num_blocks from variable_block_sizes.shape[0]
        if variable_block_sizes.shape[0] != actual_num_blocks:
            raise RuntimeError(
                f"variable_block_sizes adjustment failed: "
                f"variable_block_sizes has {variable_block_sizes.shape[0]} blocks, "
                f"but actual sequence length {actual_seq_len} requires {actual_num_blocks} blocks. "
                f"q.shape={query.shape}, k.shape={key.shape}, v.shape={value.shape}, "
                f"sp_size={self.sp_size}."
            )
        
        # Use the unified interface that automatically selects the appropriate implementation
        hidden_states = video_sparse_attn(
            query,
            key,
            value,
            variable_block_sizes=variable_block_sizes,
            topk=cur_topk,
            block_size=VSA_TILE_SIZE,
            compress_attn_weight=gate_compress).transpose(1, 2)

        return hidden_states
