# FastVideo Architecture Patterns

## Pipeline Architecture
- Use **ComposedPipelineBase** as the base class for all pipelines
- Implement **LoRAPipeline** for LoRA-enabled pipelines
- Use **PipelineWithLoRA** type for pipelines with both capabilities
- Follow **registry pattern** for dynamic pipeline selection
- Use **modular stage-based architecture** with discrete pipeline stages
- Implement **PipelineStage** subclasses for each stage (input validation, text encoding, latent preparation, denoising, decoding)
- Use **stage dependency injection** for passing modules between stages
- Support **Cosmos pipeline** architecture for Cosmos models

## Model Loading
- Use **factory methods** for model instantiation (`from_pretrained`, `from_fastvideo_args`)
- Implement **lazy loading** for expensive model components
- Use **component loaders** for modular model parts
- Support **multiple model formats** (Diffusers, custom checkpoints)

## Attention Backends
- Implement **abstract attention backends** with common interface
- Use **backend selector** for dynamic attention selection
- Support **multiple attention types** (VSA, STA, Sage, Flash Attention, VMOBA)
- Implement **distributed attention** for multi-GPU setups
- Organize attention backends in `fastvideo/attention/backends/` directory
- Use **attention backend enumeration** (`AttentionBackendEnum`) for type safety

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

## Pipeline Stage Verification
- Implement **verify_input** and **verify_output** methods for all pipeline stages
- Use **StageValidators** (`V`) for common validation checks
- Return **VerificationResult** objects from verification methods
- Enable verification via `fastvideo_args.enable_stage_verification`
- Use validators like `V.positive_int`, `V.is_tensor`, `V.tensor_with_dims`, `V.divisible_by`
- Add validation checks with `result.add_check(field_name, value, validator)`
- Raise **StageVerificationError** when verification fails

## Image Processing
- Use **ImageProcessor** class for lightweight image preprocessing
- Support **PIL.Image**, **numpy.ndarray**, and **torch.Tensor** inputs
- Handle **VAE scale factor** alignment for dimensions
- Normalize images to **[-1, 1] range** for diffusion models
- Use **ImageProcessor.preprocess()** method for preprocessing

## Embedding Patterns
- Use **rotary embeddings** (`apply_rotary_emb`) for positional encoding
- Support **1D and nD rotary positional embeddings** (`get_1d_rotary_pos_embed`, `get_nd_rotary_pos_embed`)
- Use **RotaryEmbedding** class with caching for efficiency
- Support **sequence parallelism** in rotary embeddings via `shard_dim` parameter
- Use **visual embeddings** (`Timesteps`, `TimestepEmbedder`) for timestep conditioning
- Support **patch embeddings** (`PatchEmbed`) for image/video patchification

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

# Pipeline stage implementation
class LatentPreparationStage(PipelineStage):
    def __init__(self, scheduler, transformer):
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer
    
    def verify_input(self, batch, fastvideo_args):
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int_divisible(8))
        result.add_check("width", batch.width, V.positive_int_divisible(8))
        return result
    
    def forward(self, batch, fastvideo_args):
        # Stage implementation
        return batch

# Stage creation in pipeline
def create_pipeline_stages(self, fastvideo_args):
    self.add_stage(
        stage_name="latent_preparation_stage",
        stage=LatentPreparationStage(
            scheduler=self.get_module("scheduler"),
            transformer=self.get_module("transformer")
        )
    )

# Image processing
image_processor = ImageProcessor(vae_scale_factor=8)
processed_tensor = image_processor.preprocess(
    image, height=256, width=256
)

# Rotary embeddings
cos, sin = get_nd_rotary_pos_embed(
    rope_dim_list=[64, 64, 64],
    rope_sizes=(16, 32, 32),
    theta=10000.0,
    shard_dim=0,
    sp_rank=sp_rank,
    sp_world_size=sp_world_size
)
```