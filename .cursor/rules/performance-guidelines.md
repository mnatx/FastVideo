# FastVideo Performance Guidelines

## Memory Optimization
- Use **activation checkpointing** for memory efficiency during training
- Implement **gradient checkpointing** to reduce memory usage
- Use **mixed precision** (fp16/bf16) for training and inference
- Monitor **GPU memory usage** and implement cleanup procedures

## Attention Optimization
- Use **Video Sparse Attention (VSA)** for video generation tasks
- Implement **Sliding Tile Attention (STA)** for memory efficiency
- Use **Sage Attention** for advanced optimization
- Support **Flash Attention** for standard attention patterns

## Distributed Computing
- Use **FSDP2** for model parallelism with near-linear scaling
- Implement **sequence parallelism** for long video sequences
- Use **data parallelism** for batch processing
- Support **tensor parallelism** for large models

## Caching Strategies
- Implement **TeaCache** for attention computation caching
- Use **LRU caching** for repeated computations with `@lru_cache`
- Cache **model weights** and **embeddings** when appropriate
- Implement **incremental caching** for batch processing

## Batch Processing
- **Align batch sizes** to GPU memory and model requirements
- Use **dynamic batching** for variable-length sequences
- Implement **efficient data loading** with proper prefetching
- Support **gradient accumulation** for large effective batch sizes

## Profiling and Monitoring
- Use **PyTorch profiler** for performance analysis
- Monitor **GPU utilization** and **memory usage**
- Log **timing information** for performance-critical operations
- Implement **performance metrics** collection

## Data Loading
- Use **efficient data loaders** with proper num_workers
- Implement **data prefetching** for GPU utilization
- Use **memory-mapped files** for large datasets
- Support **streaming data loading** for very large datasets

## Model Optimization
- Use **torch.compile** for model optimization when appropriate
- Implement **kernel fusion** for custom operations
- Use **optimized CUDA kernels** for attention operations
- Support **quantization** for inference optimization

## Example Patterns
```python
# Memory-efficient training
@torch.no_grad()
def forward_with_checkpointing(self, x):
    return checkpoint(self._forward_impl, x, use_reentrant=False)

# Efficient attention
def create_attention_layer(self, config):
    if config.attention_type == "vsa":
        return VideoSparseAttention(config)
    elif config.attention_type == "sta":
        return SlidingTileAttention(config)
    else:
        return StandardAttention(config)

# Batch processing optimization
def process_batch(self, batch):
    # Align batch size to GPU memory
    aligned_batch_size = align_to(batch.size(0), self.gpu_memory_limit)
    return self.model(batch[:aligned_batch_size])

# Performance monitoring
start_time = time.perf_counter()
result = self.execute_forward(batch)
gen_time = time.perf_counter() - start_time
logger.info("Generated successfully in %.2f seconds", gen_time)
```