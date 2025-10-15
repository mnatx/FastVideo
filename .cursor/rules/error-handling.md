# FastVideo Error Handling Guidelines

## Exception Types
- **FileNotFoundError**: When required files don't exist
- **ValueError**: Invalid input values or parameters
- **TypeError**: Wrong data types passed to functions
- **RuntimeError**: Runtime conditions that prevent execution
- **AttributeError**: Missing required attributes or methods

## Input Validation
- **Validate all inputs** at function boundaries
- Check **data types** before processing
- Validate **ranges and constraints** (positive numbers, valid paths, etc.)
- Use **early returns** for validation failures

## Error Context
- Include **meaningful error messages** with context
- Provide **suggestions for fixes** when possible
- Include **input values** that caused the error
- Use **f-strings** for error message formatting

## Resource Management
- Use **try-except-finally** blocks for resource cleanup
- Implement **graceful shutdown** procedures
- Handle **interrupt signals** (KeyboardInterrupt) appropriately
- Clean up **temporary files** and **GPU memory**

## Batch Processing
- **Continue processing** other items when one fails
- Log **individual failures** without stopping the batch
- Provide **summary statistics** of successes/failures
- Use **try-except** around individual batch items

## Distributed Computing
- Handle **process-specific errors** appropriately
- Use **process-aware logging** for error reporting
- Implement **timeout handling** for distributed operations
- Handle **communication failures** between processes

## Error Recovery
- Implement **fallback mechanisms** when possible
- Use **default values** for optional parameters
- Provide **alternative execution paths** for common failures
- Log **recovery actions** taken

## Example Patterns
```python
# Input validation
if not isinstance(prompt, str):
    raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")

if sampling_param.height <= 0 or sampling_param.width <= 0:
    raise ValueError(f"Height and width must be positive integers, got "
                   f"height={sampling_param.height}, width={sampling_param.width}")

# File handling
if not os.path.exists(prompt_txt_path):
    raise FileNotFoundError(f"Prompt text file not found: {prompt_txt_path}")

# Batch processing with error handling
try:
    result = self._generate_single_video(batch_prompt, sampling_param, **kwargs)
    results.append(result)
    logger.info("Successfully generated video for prompt %d", i + 1)
except Exception as e:
    logger.error("Failed to generate video for prompt %d: %s", i + 1, e)
    continue  # Continue with next item

# Resource cleanup
try:
    # Resource usage
    pass
except Exception as e:
    logger.error("Error during processing: %s", e)
    raise
finally:
    # Cleanup resources
    if hasattr(self, 'executor'):
        self.executor.shutdown()
```