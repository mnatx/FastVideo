# FastVideo Error Handling Guidelines

## Exception Types
- **FileNotFoundError**: When required files don't exist
- **ValueError**: Invalid input values or parameters
- **TypeError**: Wrong data types passed to functions
- **RuntimeError**: Runtime conditions that prevent execution
- **AttributeError**: Missing required attributes or methods
- **StageVerificationError**: When pipeline stage input/output verification fails

## Input Validation
- **Validate all inputs** at function boundaries
- Check **data types** before processing
- Validate **ranges and constraints** (positive numbers, valid paths, etc.)
- Use **early returns** for validation failures
- Implement **stage verification** for pipeline stages using `verify_input` and `verify_output` methods
- Use **StageValidators** (`V`) for common validation checks
- Return **VerificationResult** objects from verification methods
- Enable verification via `fastvideo_args.enable_stage_verification` flag

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

## Pipeline Stage Verification
- Implement **verify_input** and **verify_output** methods for all pipeline stages
- Use **StageValidators** static methods for validation checks:
  - `V.positive_int`: Check for positive integers
  - `V.is_tensor`: Check if value is a tensor without NaN values
  - `V.tensor_with_dims`: Check tensor has specific dimensions
  - `V.tensor_min_dims`: Check tensor has at least N dimensions
  - `V.divisible_by`: Check if value is divisible by a number
  - `V.positive_int_divisible`: Check positive integer divisible by a number
  - `V.not_none`: Check if value is not None
- Create **VerificationResult** objects and add checks with `result.add_check(field_name, value, validator)`
- Verification runs automatically when `fastvideo_args.enable_stage_verification` is True
- Raise **StageVerificationError** with detailed failure information when verification fails
- Include **failed field names** and **detailed failure summaries** in error messages

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

# Pipeline stage verification
from fastvideo.pipelines.stages.validators import StageValidators as V, VerificationResult

def verify_input(self, batch, fastvideo_args):
    result = VerificationResult()
    result.add_check("height", batch.height, V.positive_int_divisible(8))
    result.add_check("width", batch.width, V.positive_int_divisible(8))
    result.add_check("latents", batch.latents, V.tensor_with_dims(5))  # [B, C, T, H, W]
    result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
    return result

def verify_output(self, batch, fastvideo_args):
    result = VerificationResult()
    result.add_check("latents", batch.latents, V.is_tensor)
    result.add_check("latents", batch.latents, V.tensor_min_dims(4))
    return result

# Stage verification error handling
try:
    stage(batch, fastvideo_args)
except StageVerificationError as e:
    logger.error("Stage verification failed: %s", e)
    # Error message includes failed fields and detailed summary
    raise
```