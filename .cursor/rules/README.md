# FastVideo Cursor Rules

This directory contains project-specific rules and guidelines for the FastVideo codebase. These rules are designed to help maintain code quality, consistency, and best practices throughout the project.

## Rule Files

### Core Development Rules
- **[coding-standards.md](./coding-standards.md)** - Type annotations, code organization, and function design
- **[error-handling.md](./error-handling.md)** - Exception handling, input validation, and error recovery
- **[logging-patterns.md](./logging-patterns.md)** - Logging standards and process-aware logging

### Architecture and Performance
- **[architecture-patterns.md](./architecture-patterns.md)** - Pipeline architecture, model loading, and distributed computing
- **[performance-guidelines.md](./performance-guidelines.md)** - Memory optimization, attention mechanisms, and performance monitoring

### Quality Assurance
- **[testing-standards.md](./testing-standards.md)** - Testing framework, fixtures, and test organization
- **[documentation-standards.md](./documentation-standards.md)** - Docstring format, API documentation, and examples

## Key Principles

### 1. **Type Safety**
- Always use type hints for function parameters and return values
- Use modern Python syntax (`str | None` instead of `Optional[str]`)
- Implement comprehensive input validation

### 2. **Process-Aware Development**
- Use process-aware logging for distributed environments
- Handle multi-GPU scenarios appropriately
- Implement proper resource cleanup

### 3. **Performance First**
- Consider memory usage and computational efficiency
- Use appropriate attention mechanisms for video generation
- Implement caching and optimization strategies

### 4. **Error Resilience**
- Provide meaningful error messages with context
- Implement graceful degradation in batch processing
- Handle distributed computing failures appropriately

### 5. **Modular Design**
- Follow the registry pattern for dynamic component loading
- Use abstract base classes for extensibility
- Maintain clear separation of concerns
- Use **pipeline stages** for modular pipeline components
- Implement **stage verification** for robust error handling

## Quick Reference

### Logging
```python
from fastvideo.logger import init_logger
logger = init_logger(__name__)
logger.info("Message", local_main_process_only=True)  # Default
```

### Error Handling
```python
if not isinstance(prompt, str):
    raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
```

### Type Annotations
```python
def generate_video(self, prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate video with type hints and docstring."""
```

### Testing
```python
@pytest.fixture
def sample_batch():
    return ForwardBatch(prompt="test", height=256, width=256)
```

### Pipeline Stage Verification
```python
from fastvideo.pipelines.stages.validators import StageValidators as V, VerificationResult

def verify_input(self, batch, fastvideo_args):
    result = VerificationResult()
    result.add_check("height", batch.height, V.positive_int_divisible(8))
    result.add_check("width", batch.width, V.positive_int_divisible(8))
    return result
```

### Image Processing
```python
from fastvideo.image_processor import ImageProcessor

image_processor = ImageProcessor(vae_scale_factor=8)
processed_tensor = image_processor.preprocess(image, height=256, width=256)
```

### Rotary Embeddings
```python
from fastvideo.layers.rotary_embedding import get_nd_rotary_pos_embed

cos, sin = get_nd_rotary_pos_embed(
    rope_dim_list=[64, 64, 64],
    rope_sizes=(16, 32, 32),
    theta=10000.0,
    shard_dim=0,
    sp_rank=sp_rank,
    sp_world_size=sp_world_size
)
```

## Recent Updates (Commit 188e872)

This ruleset has been updated to include patterns from commit 188e87242ebc6abed189e421ecf8f7a834cc499d, which includes:

- **Pipeline Stage Architecture**: New modular stage-based pipeline system with verification
- **Image Processing**: New lightweight `ImageProcessor` class for preprocessing
- **Rotary Embeddings**: New rotary embedding utilities with sequence parallelism support
- **Visual Embeddings**: New visual embedding patterns for timestep conditioning
- **Cosmos Pipeline**: Support for Cosmos video diffusion pipeline
- **Attention Backends**: Reorganized attention backend structure
- **Documentation**: Migration from Sphinx to MkDocs

## Usage

These rules should be followed when:
- Writing new code
- Refactoring existing code
- Reviewing pull requests
- Setting up new development environments

For questions or clarifications about these rules, please refer to the individual rule files or consult with the development team.