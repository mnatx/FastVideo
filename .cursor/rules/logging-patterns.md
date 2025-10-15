# FastVideo Logging Patterns

## Logger Initialization
- Always use `from fastvideo.logger import init_logger` to get logger instances
- Initialize loggers with module name: `logger = init_logger(__name__)`
- Use the **process-aware logging** system for distributed environments

## Log Levels
- **DEBUG**: Detailed information for debugging (variable values, execution flow)
- **INFO**: General information about program execution (startup, completion, key metrics)
- **WARNING**: Something unexpected happened but program continues
- **ERROR**: Serious problem occurred, some functionality may be affected

## Process-Aware Logging
- Use `logger.info(..., local_main_process_only=True)` for local process logging (default)
- Use `logger.info(..., main_process_only=True)` for global main process logging
- Use `logger.info(..., local_main_process_only=False, main_process_only=False)` for all processes
- **Never log from all processes by default** - this causes log flooding in multi-GPU setups

## Logging Best Practices
- Use **structured logging** with format strings: `logger.info("Processing %d/%d: %s", i, total, name)`
- Include **context information** in log messages (file paths, dimensions, counts)
- Use **once-only logging** for repeated messages: `logger.info_once("Message")`
- **Log important state changes** and **error conditions**
- Use **appropriate log levels** - don't use ERROR for expected conditions

## Performance Logging
- Log **timing information** for performance-critical operations
- Use `time.perf_counter()` for accurate timing measurements
- Log **memory usage** and **GPU utilization** when relevant
- Include **batch processing progress** with current/total counts

## Error Logging
- Always **log exceptions** with full context
- Use **stacklevel=2** to show the original caller's location
- Include **input parameters** in error logs for debugging
- Log **recovery actions** when errors are handled gracefully

## Example Patterns
```python
# Good logging examples
logger.info("Starting video generation with %d frames", num_frames)
logger.info("Processing prompt %d/%d: %s...", i + 1, total, prompt[:100])
logger.error("Failed to generate video for prompt %d: %s", i + 1, str(e))
logger.warning("No output path provided, video not saved")
logger.info_once("Using attention backend: %s", attention_backend)
```