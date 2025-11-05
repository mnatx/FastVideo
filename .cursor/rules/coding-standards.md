# FastVideo Coding Standards

## Type Annotations
- **Always use type hints** for function parameters, return values, and class attributes
- Use `typing` module imports: `from typing import Any, Dict, List, Optional, Union, TypeVar`
- Use modern Python syntax: `str | None` instead of `Optional[str]` when possible
- Use `TypeVar` for generic types: `T = TypeVar("T")`

## Code Organization
- Follow the **modular design pattern** with clear separation of concerns
- Use **abstract base classes** for interfaces and common functionality
- Implement **registry patterns** for dynamic component loading
- Group related functionality in dedicated modules (models/, pipelines/, attention/, etc.)
- Organize attention backends in `fastvideo/attention/backends/` directory
- Use **pipeline stages** (`fastvideo/pipelines/stages/`) for modular pipeline components
- Group embedding utilities in `fastvideo/layers/` (rotary_embedding.py, visual_embedding.py)

## Function and Class Design
- Use **dataclasses** for configuration objects with `@dataclass` decorator
- Implement **context managers** for resource management
- Use **class methods** for factory patterns (e.g., `from_pretrained`, `from_kwargs`)
- Follow **single responsibility principle** - one class/function per purpose
- Use **static methods** for utility functions (e.g., `StageValidators` methods)
- Implement **abstract methods** in base classes (e.g., `PipelineStage.forward()`)
- Use **weak references** (`weakref.ref`) for circular dependencies that should be garbage collected

## Error Handling
- Use **specific exception types**: `FileNotFoundError`, `ValueError`, `TypeError`, `RuntimeError`
- Always **validate inputs** at function boundaries
- Provide **meaningful error messages** with context information
- Use **try-except-finally** blocks for resource cleanup
- Implement **graceful degradation** in batch processing scenarios

## Performance Considerations
- Use **lazy evaluation** for expensive operations
- Implement **caching** with `@lru_cache` for repeated computations
- Consider **memory usage** and implement activation checkpointing when needed
- Use **efficient data structures** and avoid unnecessary copies
- Cache **rotary embedding** computations using module-level dictionaries (`_ROPE_DICT`)
- Use **register_buffer** for non-trainable cached tensors (e.g., `cos_sin_cache`)
- Avoid external dependencies when possible (e.g., `ImageProcessor` uses only PyTorch/NumPy/PIL)

## Documentation
- Write **comprehensive docstrings** for all public functions and classes
- Include **type information** in docstrings
- Document **side effects** and **resource usage**
- Provide **usage examples** for complex APIs
- Include **SPDX license identifiers** at the top of files (`# SPDX-License-Identifier: Apache-2.0`)
- Document **sequence parallelism** parameters when applicable (`shard_dim`, `sp_rank`, `sp_world_size`)
- Include **adapter/attribution notes** when code is adapted from other projects (e.g., vLLM, transformers)

## Image Processing Patterns
- Use **ImageProcessor** class for lightweight image preprocessing (avoids heavy dependencies)
- Support **multiple input types**: `PIL.Image.Image`, `np.ndarray`, `torch.Tensor`
- Handle **grayscale images** by expanding dimensions appropriately
- Ensure dimensions are **aligned to VAE scale factor** (typically 8)
- Normalize to **[-1, 1] range** for diffusion model compatibility
- Use **LANCZOS resampling** for high-quality image resizing

## Embedding Patterns
- Use **rotary embeddings** for positional encoding in attention layers
- Support **1D rotary embeddings** (`get_1d_rotary_pos_embed`) for sequence dimensions
- Support **nD rotary embeddings** (`get_nd_rotary_pos_embed`) for spatial/temporal dimensions
- Cache rotary embeddings using **RotaryEmbedding** class with `register_buffer`
- Support **sequence parallelism** by accepting `shard_dim`, `sp_rank`, `sp_world_size` parameters
- Use **get_rotary_pos_embed** helper for common 3D (temporal/spatial) cases
- Support **different rotary styles** (NeoX vs GPT-J) via `is_neox_style` parameter
- Use **visual embeddings** (`Timesteps`, `TimestepEmbedder`) for timestep conditioning
- Support **patch embeddings** for converting images/videos to patch sequences