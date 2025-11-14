# Yggdrasil

Yggdrasil is a multi-language research playground focused on data processing and interoperability. The Python package provides utilities for working with PyArrow, Polars, type conversions, and efficient data handling.

## Overview

This project aims to bridge the gap between Python's type system and Arrow-based data processing frameworks, creating a seamless development experience for data engineering tasks. It leverages modern Python features like type annotations and dataclasses to provide type safety while enabling high-performance data processing.

## Features

### Arrow DataClass

The `arrow_dataclass` decorator extends Python's dataclasses with PyArrow schema support, enabling seamless integration between Python's type annotations and Arrow's type system. This makes it easy to define structured data that can be efficiently serialized and deserialized with PyArrow.

Key features:
- Automatic schema inference from Python type hints
- Non-nullable fields by default
- Nullable support for Optional types and Union with None
- Support for collections (List, Dict, Set, Tuple)
- Support for nested dataclasses
- Customizable field metadata
- Type overrides for fine-grained control

Example usage:

```python
from typing import List, Optional
from yggdrasil.types import arrow_dataclass, field

@arrow_dataclass
class User:
    user_id: int
    username: str
    email: str
    active: bool = True
    bio: Optional[str] = None  # Optional fields are nullable
    age: int = field(
        default=0,
        arrow_metadata={"description": "User age in years"}
    )

# Access the generated Arrow schema
schema = User.__arrow_schema__
print(schema)
```

## Project Structure

The project is organized as follows:

```
yggdrasil/
├── python/               # Python package
│   ├── src/              # Source code
│   │   └── yggdrasil/    # Main package
│   │       ├── types/    # Type definition and conversion
│   │       │   ├── field.py   # DataField implementation
│   │       │   └── schema.py  # DataSchema implementation
│   │       ├── data/     # Data handling utilities
│   │       └── logging.py # Logging configuration
│   └── tests/            # Test suite
└── README.md            # This file
```

## Key Components

### Types Module

The `yggdrasil.types` module provides:

- **DataField**: A representation of a field in an Arrow schema, with support for converting Python type hints to Arrow data types
- **DataSchema**: A collection of DataFields forming a complete schema, created from Python classes or existing Arrow schemas

### Logging

The package includes a configurable logging system that sets up appropriate handlers and formatters for consistent log output across the application.

## Quick Start

### Installation

**Option 1: Direct from GitHub (recommended)**
```bash
pip install "yggdrasil @ git+https://github.com/Platob/Yggdrasil.git"
```

**Option 2: Local Development Setup**

For Windows:
```powershell
# Clone the repository (if you haven't already)
# git clone https://github.com/Platob/Yggdrasil.git
# cd Yggdrasil

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install with development dependencies
pip install -e .[dev]

# Run tests to verify installation
pytest
```

For macOS/Linux:
```bash
# Clone the repository (if you haven't already)
# git clone https://github.com/Platob/Yggdrasil.git
# cd Yggdrasil

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e .[dev]

# Run tests to verify installation
pytest
```

## Dependencies

The core dependencies of Yggdrasil include:

- **PyArrow**: Core library for working with Arrow data format
- **Polars**: High-performance DataFrame library built on Arrow
- **tzdata**: Time zone database

Optional dependencies:

- **deltalake**: Support for working with Delta Lake tables (included in `[delta]` and `[dev]` extras)

## Development

### Testing

Tests are written using pytest and can be run with:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=yggdrasil

# Run specific tests
pytest python/tests/data/types/test_field.py
```

### Code Style

Code formatting and linting is handled by Ruff:

```bash
# Check style
ruff check .

# Automatically fix style issues
ruff check --fix .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Roadmap

The project is under active development, with planned features including:

- Extended polars integration for schema validation and conversion
- Additional Arrow data type support
- Delta Lake schema evolution
- Advanced serialization/deserialization options
- Performance optimizations for large datasets

## License

This project is licensed under the terms of the included LICENSE file.
