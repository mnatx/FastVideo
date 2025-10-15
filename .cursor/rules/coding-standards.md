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

## Function and Class Design
- Use **dataclasses** for configuration objects with `@dataclass` decorator
- Implement **context managers** for resource management
- Use **class methods** for factory patterns (e.g., `from_pretrained`, `from_kwargs`)
- Follow **single responsibility principle** - one class/function per purpose

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

## Documentation
- Write **comprehensive docstrings** for all public functions and classes
- Include **type information** in docstrings
- Document **side effects** and **resource usage**
- Provide **usage examples** for complex APIs