# Yggdrasil Test Suite

This directory contains the test suite for the Yggdrasil project. The tests are written using pytest and are organized to mirror the structure of the source code.

## Running Tests

To run the entire test suite:

```bash
# From the project root
pytest

# Or with more details
pytest -v
```

To run specific test files or directories:

```bash
# Run a specific test file
pytest python/tests/libutils/test_arrow_utils.py

# Run all tests in a directory
pytest python/tests/libutils/
```

To run tests with coverage reports:

```bash
pytest --cov=yggdrasil
```

## Test Structure

The tests are organized to match the structure of the source code:

- `tests/data/` - Tests for the data module
- `tests/utils/` - Tests for the utility modules
- `tests/types/` - Tests for the types module

## Handling Optional Dependencies

The test suite uses pytest's skip markers to handle optional dependencies:

- Tests that require optional dependencies like NumPy, Pandas, Polars, or PySpark use `@pytest.mark.skipif`
- When a dependency is not available, related tests are automatically skipped
- This allows the test suite to run in environments with minimal dependencies

Example:

```python
import pytest

# Skip the test if pandas is not installed
has_pandas = False
try:
    import pandas as pd
    has_pandas = True
except ImportError:
    pass

@pytest.mark.skipif(not has_pandas, reason="Pandas not installed")
def test_pandas_functionality():
    # Test code that requires pandas
    ...
```

## Test Files Overview

### Utils Module Tests

- `test_arrow_utils.py` - Tests for PyArrow utility functions
- `test_fake_module.py` - Tests for the fake module implementation
- `test_numpy_pandas_utils.py` - Tests for NumPy and Pandas utilities
- `test_polars_utils.py` - Tests for Polars DataFrame utilities
- `test_py_utils.py` - Tests for Python utility functions
- `test_spark_utils.py` - Tests for PySpark utility functions

### Types Module Tests

- `test_field.py` - Tests for the DataField class
- `test_schema.py` - Tests for schema functionality
- `test_py_hints.py` - Tests for Python type hint conversions