# FastVideo Architecture Patterns

## Pipeline Architecture
- Use **ComposedPipelineBase** as the base class for all pipelines
- Implement **LoRAPipeline** for LoRA-enabled pipelines
- Use **PipelineWithLoRA** type for pipelines with both capabilities
- Follow **registry pattern** for dynamic pipeline selection

## Model Loading
- Use **factory methods** for model instantiation (`from_pretrained`, `from_fastvideo_args`)
- Implement **lazy loading** for expensive model components
- Use **component loaders** for modular model parts
- Support **multiple model formats** (Diffusers, custom checkpoints)

## Attention Backends
- Implement **abstract attention backends** with common interface
- Use **backend selector** for dynamic attention selection
- Support **multiple attention types** (VSA, STA, Sage, Flash Attention)
- Implement **distributed attention** for multi-GPU setups

## Configuration Management
- Use **environment variables** for runtime configuration
- Implement **JSON configuration files** for model-specific settings
- Use **dataclasses** for configuration objects
- Support **configuration inheritance** and **overrides**

## Distributed Computing
- Use **process-aware logging** for distributed environments
- Implement **rank-based execution** for multi-GPU operations
- Support **FSDP2** for model parallelism
- Use **sequence parallelism** for long sequences

## Memory Management
- Implement **activation checkpointing** for memory efficiency
- Use **gradient checkpointing** during training
- Support **mixed precision** training and inference
- Implement **memory monitoring** and **cleanup**

## Testing Patterns
- Use **pytest** for testing framework
- Implement **fixtures** for common test setup
- Use **parametrized tests** for multiple configurations
- Support **distributed testing** with proper setup/teardown

## Example Patterns
```python
# Pipeline factory pattern
@classmethod
def from_pretrained(cls, model_path: str, **kwargs) -> "VideoGenerator":
    fastvideo_args = FastVideoArgs.from_kwargs(**kwargs)
    return cls.from_fastvideo_args(fastvideo_args)

# Registry pattern
pipeline_registry = get_pipeline_registry(pipeline_type)
pipeline_cls = pipeline_registry.resolve_pipeline_cls(
    pipeline_name, pipeline_type, workload_type)

# Component loading
def load_component(component_type: str, config: Dict[str, Any]):
    loader = ComponentLoader(component_type)
    return loader.load(config)

# Attention backend selection
attention_backend = get_attn_backend(backend_name)
attention_layer = attention_backend.create_layer(config)
```