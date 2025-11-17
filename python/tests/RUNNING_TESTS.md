# Running Yggdrasil Tests Safely

This guide provides instructions for running tests in the Yggdrasil project, especially when dealing with potential memory issues or missing dependencies.

## Initial Setup

1. Make sure you have the core dependencies installed:

```bash
pip install -e .
```

2. If you want to run the full test suite, install all optional dependencies:

```bash
pip install -e ".[dev,spark,pandas]"
```

## Running Tests Safely

### Option 1: Run Only Specific Test Files

The safest approach is to run specific test files one at a time:

```bash
# Run a specific test file
pytest python/tests/libutils/test_py_utils.py -v

# Run specific test cases
pytest python/tests/libutils/test_arrow_utils.py::test_python_to_arrow_type_map_mappings -v
```

### Option 2: Run with Limited xdist Workers

Use pytest-xdist to run tests in parallel with a limited number of workers:

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests with 2 workers (adjust based on available memory)
pytest -n 2
```

### Option 3: Run with Memory Constraints

If you're experiencing memory issues:

```bash
# Limit memory usage with xvfb-run (Linux)
xvfb-run --server-args="-screen 0 1024x768x24" pytest

# On Windows, try running with limited memory using job objects or similar tools
# or simply run tests in smaller batches
```

## Test Categories

Tests are organized by dependency requirements:

1. **Core Tests**: Don't require optional dependencies
   - `python/tests/utils/test_py_utils.py`
   - `python/tests/utils/test_fake_module.py`

2. **Arrow-dependent Tests**: Require PyArrow
   - `python/tests/utils/test_arrow_utils.py`
   - `python/tests/types/test_field.py`

3. **Optional Dependency Tests**: Skip automatically when dependencies are missing
   - `python/tests/utils/test_polars_utils.py` (requires Polars)
   - `python/tests/utils/test_spark_utils.py` (requires PySpark)
   - `python/tests/utils/test_numpy_pandas_utils.py` (requires NumPy/Pandas)

## Troubleshooting

### Access Violation Errors

If you encounter "access violation" errors on Windows:

1. Run tests one file at a time
2. Ensure you have sufficient memory (at least 4GB available)
3. Close other memory-intensive applications
4. Try running tests in a fresh command prompt or terminal

### ImportError or ModuleNotFoundError

These are expected for optional dependencies and the tests should automatically skip. If you see unexpected import errors:

1. Verify your Python environment is activated
2. Ensure the package is installed in development mode (`pip install -e .`)
3. Check for path issues in your environment

### Other Memory Issues

If you're still experiencing memory problems:

1. Run the test with increased verbosity to identify problematic tests:
   ```bash
   pytest -vv
   ```

2. Skip problematic test modules:
   ```bash
   pytest --ignore=python/tests/libutils/test_spark_utils.py
   ```

## Maintaining Test Safety

When adding new tests:

1. Always use `pytest.mark.skipif` for tests that require optional dependencies
2. Handle imports safely with try/except blocks
3. Check for the existence of required modules before accessing their functionality
4. Add appropriate dependencies to the relevant `[extras]` section in `pyproject.toml`