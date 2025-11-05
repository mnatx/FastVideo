# FastVideo Documentation Standards

## Code Documentation
- Write **comprehensive docstrings** for all public functions, classes, and methods
- Use **Google-style docstrings** with clear sections
- Include **type information** in docstrings
- Document **side effects** and **resource usage**
- Include **SPDX license identifiers** at the top of files (`# SPDX-License-Identifier: Apache-2.0`)
- Add **adapter notes** when code is adapted from other projects (e.g., "Adapted from vllm", "Adapted from transformers")

## Docstring Format
```python
def generate_video(self, prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate a video based on the given prompt.
    
    Args:
        prompt: The text prompt for video generation
        **kwargs: Additional parameters (height, width, num_frames, etc.)
        
    Returns:
        Dictionary containing generated video data and metadata
        
    Raises:
        ValueError: If prompt is empty or invalid
        RuntimeError: If generation fails
        
    Example:
        >>> generator = VideoGenerator.from_pretrained("model_name")
        >>> result = generator.generate_video("A cat playing")
        >>> print(result["frames"].shape)
    """
```

## API Documentation
- Document **all public APIs** with examples
- Include **parameter descriptions** with types and constraints
- Document **return values** with structure details
- Provide **usage examples** for common scenarios

## Configuration Documentation
- Document **all configuration options** with descriptions
- Include **default values** and **valid ranges**
- Explain **environment variables** and their effects
- Provide **configuration examples** for different use cases

## Error Documentation
- Document **all possible exceptions** that functions can raise
- Include **error conditions** and **recovery suggestions**
- Explain **error codes** and **status values**
- Provide **troubleshooting guides** for common issues

## Performance Documentation
- Document **performance characteristics** and **scaling behavior**
- Include **memory requirements** and **GPU specifications**
- Explain **optimization options** and **tuning parameters**
- Provide **benchmark results** and **performance tips**

## Example Documentation
- Provide **complete working examples** for main features
- Include **step-by-step tutorials** for complex workflows
- Show **best practices** and **common patterns**
- Demonstrate **error handling** and **recovery procedures**

## Documentation System
- Use **MkDocs** for project documentation (replaces Sphinx)
- Configuration file: `mkdocs.yml` in project root
- Documentation source files in `docs/` directory (not `docs/source/`)
- Use **Markdown** format for documentation files (`.md`)
- API documentation generated via `docs/api/` directory
- Custom assets (CSS, JS, images) in `docs/assets/` directory
- Run `mkdocs serve` for local preview or `mkdocs build` for production

## README Standards
- Include **quick start** instructions
- Provide **installation** and **setup** guides
- Show **basic usage** examples
- List **requirements** and **dependencies**

## Code Comments
- Use **inline comments** for complex logic
- Explain **non-obvious** implementation details
- Document **algorithm choices** and **optimizations**
- Include **TODO comments** for future improvements
- Document **sequence parallelism** parameters when applicable
- Explain **rotary embedding** parameter choices (e.g., `use_real_unbind_dim`)

## Example Patterns
```python
class VideoGenerator:
    """A unified class for generating videos using diffusion models.
    
    This class provides a simple interface for video generation with rich
    customization options, similar to popular frameworks like HF Diffusers.
    
    Attributes:
        fastvideo_args: The inference arguments
        executor: The executor for running inference
        
    Example:
        >>> generator = VideoGenerator.from_pretrained("FastWan2.1-T2V-1.3B")
        >>> video = generator.generate_video("A dog running in the park")
        >>> print(f"Generated {len(video['frames'])} frames")
    """
    
    def __init__(self, fastvideo_args: FastVideoArgs, executor_class: type[Executor]):
        """Initialize the video generator.
        
        Args:
            fastvideo_args: The inference arguments
            executor_class: The executor class to use for inference
        """
        self.fastvideo_args = fastvideo_args
        self.executor = executor_class(fastvideo_args)
```