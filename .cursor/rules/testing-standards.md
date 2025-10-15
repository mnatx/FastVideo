# FastVideo Testing Standards

## Test Organization
- Use **pytest** as the testing framework
- Organize tests in `fastvideo/tests/` directory
- Use **descriptive test names** that explain what is being tested
- Group related tests in **test classes** or **test modules**

## Test Structure
- Use **fixtures** for common test setup and teardown
- Implement **parametrized tests** for multiple configurations
- Use **pytest.mark** for test categorization and skipping
- Support **distributed testing** with proper environment setup

## Test Categories
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test performance characteristics
- **Nightly Tests**: Long-running or resource-intensive tests

## Fixtures and Setup
- Use **conftest.py** for shared fixtures
- Implement **distributed setup fixtures** for multi-GPU tests
- Use **temporary directories** for file-based tests
- Clean up **resources** after each test

## Mocking and Stubbing
- Use **unittest.mock** for external dependencies
- Mock **expensive operations** (model loading, GPU operations)
- Use **pytest fixtures** for dependency injection
- Implement **test doubles** for complex dependencies

## Assertions
- Use **pytest assertions** for clear error messages
- Test **both success and failure cases**
- Verify **side effects** and **state changes**
- Use **approximate equality** for floating-point comparisons

## Test Data
- Use **minimal test data** that covers edge cases
- Create **synthetic data** for testing when appropriate
- Use **fixtures** for test data generation
- Avoid **hardcoded paths** in tests

## Performance Testing
- Use **pytest-benchmark** for performance regression testing
- Test **memory usage** and **GPU utilization**
- Implement **timeout handling** for long-running tests
- Use **pytest.mark.slow** for performance tests

## Example Patterns
```python
# Test fixture
@pytest.fixture
def sample_batch():
    return ForwardBatch(
        prompt="test prompt",
        height=256,
        width=256,
        num_frames=16
    )

# Parametrized test
@pytest.mark.parametrize("attention_type", ["vsa", "sta", "flash"])
def test_attention_backends(attention_type):
    config = AttentionConfig(type=attention_type)
    backend = get_attn_backend(attention_type)
    assert backend is not None

# Distributed test
@pytest.mark.usefixtures("distributed_setup")
def test_distributed_inference():
    # Test distributed inference logic
    pass

# Performance test
@pytest.mark.slow
def test_inference_performance():
    start_time = time.perf_counter()
    result = generator.generate_video("test prompt")
    elapsed = time.perf_counter() - start_time
    assert elapsed < 10.0  # Should complete within 10 seconds

# Mocking example
@patch('fastvideo.models.loader.load_model')
def test_model_loading(mock_load):
    mock_load.return_value = MockModel()
    generator = VideoGenerator.from_pretrained("test_model")
    assert generator is not None
```