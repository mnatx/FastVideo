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

## Usage

These rules should be followed when:
- Writing new code
- Refactoring existing code
- Reviewing pull requests
- Setting up new development environments

For questions or clarifications about these rules, please refer to the individual rule files or consult with the development team.