# Test Suite

## Running Tests

Install pytest:
```bash
pip install pytest
```

Run all tests:
```bash
pytest tests/
```

Run with verbose output:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_skill_extractor.py
```

## Test Coverage

- `test_skill_extractor.py` - Tests for skill extraction module
  - Initialization tests
  - Empty response structure tests
  - Input validation tests

## Future Tests

- [ ] Mock LLM responses for faster tests
- [ ] Integration tests with actual API calls
- [ ] Performance benchmarks
- [ ] Edge case coverage
