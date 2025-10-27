# SPDX-License-Identifier: Apache-2.0
# Adapted from rocm/vllm: https://github.com/ROCm/vllm/blob/v0.7.3%2Brocm/vllm/platforms/rocm.py
"""
This file is a platform abstraction for ROCm GPUs,
adjusted to match the structure and interface of `cuda.py`.
"""

import torch

import fastvideo.envs as envs
from fastvideo.logger import init_logger
from fastvideo.platforms.interface import (AttentionBackendEnum,
                                           DeviceCapability, Platform,
                                           PlatformEnum)

logger = init_logger(__name__)


# ROCm uses the same torch.cuda interface
class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"  # torch uses 'cuda' backend string
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)
    
    @classmethod
    def detect_rocm_device_type(cls, device_id: int = 0) -> str:
        """Detect the specific ROCm device type for optimal configuration."""
        if not torch.cuda.is_available():
            return "unknown"
        
        try:
            device_name = torch.cuda.get_device_name(device_id).lower()
            
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
        except:
            return "generic_rocm"
    
    @classmethod
    def get_device_shared_memory_limit(cls, device_id: int = 0) -> int:
        """Get device-specific shared memory limit in bytes."""
        device_type = cls.detect_rocm_device_type(device_id)
        
        # Device-specific shared memory limits (in bytes)
        shared_memory_limits = {
            "mi300x": 65536,   # 64KB
            "mi250": 131072,   # 128KB
            "mi210": 65536,    # 64KB
            "w7800": 32768,    # 32KB
            "generic_rocm": 65536  # 64KB default
        }
        
        return shared_memory_limits.get(device_type, 65536)
    
    @classmethod
    def get_optimal_block_size(cls, device_id: int = 0) -> tuple:
        """Get optimal block size for the detected device."""
        device_type = cls.detect_rocm_device_type(device_id)
        
        # Device-specific optimal block sizes
        block_sizes = {
            "mi300x": (32, 32),
            "mi250": (64, 64),
            "mi210": (32, 32),
            "w7800": (16, 16),
            "generic_rocm": (32, 32)
        }
        
        return block_sizes.get(device_type, (32, 32))

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA graph. "
                "Since enforce-eager is enabled, async output processor cannot be used"
            )
            return False
        return True

    @classmethod
    def log_warnings(cls) -> None:
        pass  # ROCm-specific warnings can be added here

    @classmethod
    def get_current_memory_usage(cls,
                                 device: torch.device | None = None) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.max_memory_allocated(device))

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: AttentionBackendEnum | None,
                             head_size: int, dtype: torch.dtype) -> str:
        logger.info("Trying FASTVIDEO_ATTENTION_BACKEND=%s",
                    envs.FASTVIDEO_ATTENTION_BACKEND)

        if selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            return "fastvideo.attention.backends.sdpa.SDPABackend"

        elif selected_backend in (AttentionBackendEnum.FLASH_ATTN, None):
            pass

        elif selected_backend == AttentionBackendEnum.VIDEO_SPARSE_ATTN:
            logger.info("Using Video Sparse Attention backend.")
            return "fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend"
        elif selected_backend in (AttentionBackendEnum.SLIDING_TILE_ATTN,
                                  AttentionBackendEnum.SAGE_ATTN):
            raise ValueError(
                f"{selected_backend.name} is not supported on {cls.device_name}."
            )
        elif selected_backend:
            raise ValueError(
                f"Invalid attention backend for {cls.device_name}: {selected_backend}"
            )

        target_backend = AttentionBackendEnum.FLASH_ATTN
        if dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention backend for dtype other than "
                "torch.float16 or torch.bfloat16.")
            target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.FLASH_ATTN:
            try:
                import flash_attn  # noqa: F401

                from fastvideo.attention.backends.flash_attn import (  # noqa: F401
                    FlashAttentionBackend)

                supported_sizes = \
                    FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info(
                        "Cannot use FlashAttention-2 backend for head size %d.",
                        head_size)
                    target_backend = AttentionBackendEnum.TORCH_SDPA
            except ImportError:
                logger.info("Cannot use FlashAttention backend because the "
                            "flash_attn package is not found. "
                            "Make sure that flash_attn was built and installed "
                            "(on by default).")
                target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            return "fastvideo.attention.backends.sdpa.SDPABackend"

        # Try Video Sparse Attention as a fallback for ROCm
        try:
            from fastvideo.attention.backends.video_sparse_attn import (  # noqa: F401
                VideoSparseAttentionBackend)
            
            supported_sizes = VideoSparseAttentionBackend.get_supported_head_sizes()
            if head_size in supported_sizes:
                logger.info("Using Video Sparse Attention backend.")
                return "fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend"
        except ImportError:
            logger.info("Video Sparse Attention backend not available.")

        logger.info("Using Flash Attention backend.")
        return "fastvideo.attention.backends.flash_attn.FlashAttentionBackend"

    @classmethod
    def get_torch_device(cls):
        """
        Return torch.cuda (ROCm uses CUDA interface)
        """
        return torch.cuda

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # works for ROCm too
