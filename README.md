# Yggdrasil

Yggdrasil is a multi-language research playground focused on data processing and interoperability. The Python package provides utilities for working with PyArrow, Polars, type conversions, and efficient data handling.

## Overview

This project bridges the gap between Python's type system and Arrow-based data processing frameworks, creating a seamless development experience for data engineering tasks. It leverages modern Python features like type annotations and dataclasses to provide type safety while enabling high-performance data processing.

## Features

- **Type-Safe Data Processing**: Convert between Python type hints and Arrow schemas with complete type safety
- **Dataclass Integration**: Easily work with typed dataclasses that automatically map to Arrow schemas
- **Framework Interoperability**: Seamless conversions between PyArrow, Polars, and PySpark (optional)
- **Declarative Schema Definitions**: Define schemas using familiar Python type annotations
- **Support for Nested Types**: Handle complex nested structures with lists, maps, and nested objects
- **Metadata Annotations**: Add rich metadata to fields using Python's Annotated type
- **Delta Lake Integration**: Optional support for working with Delta Lake tables

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

### Optional Dependencies

Choose the appropriate extras based on your needs:

```bash
# For Spark integration
pip install "yggdrasil @ git+https://github.com/Platob/Yggdrasil.git#egg=yggdrasil[spark]"

# For Delta Lake integration
pip install "yggdrasil @ git+https://github.com/Platob/Yggdrasil.git#egg=yggdrasil[delta]"

# For both Spark and Delta Lake
pip install "yggdrasil @ git+https://github.com/Platob/Yggdrasil.git#egg=yggdrasil[spark,delta]"

# For development (includes testing tools and Delta Lake)
pip install "yggdrasil @ git+https://github.com/Platob/Yggdrasil.git#egg=yggdrasil[dev]"
```

## Usage Examples

### Basic Type Conversion

Convert Python types to PyArrow schema using DataField:

```python
from dataclasses import dataclass
from typing import List, Optional
from yggdrasil.types.field import DataField

# Define a simple Python dataclass
@dataclass
class Person:
    name: str
    age: int
    email: Optional[str] = None

# Create a DataField from the Python type
field = DataField.from_py_hint("person", Person)

# Convert to Arrow schema
arrow_schema = field.to_arrow_schema()
print(arrow_schema)
```

### Using Metadata with Annotated Types

Enrich your schemas with metadata using Python's Annotated type:

```python
from dataclasses import dataclass
from typing import Annotated
from yggdrasil.types.field import DataField, Annotated

@dataclass
class Product:
    id: Annotated[int, "primary_key", ("indexed", True)]
    name: Annotated[str, {"description": "Product name"}]
    price: Annotated[float, {"precision": "high", "currency": "USD"}]
    stock: int

# Create a DataField with metadata
field = DataField.from_py_hint("product", Product)

# Access the enriched schema
arrow_field = field.to_arrow_field()
print(arrow_field)
```

### Working with Nested Structures

Easily handle complex nested data:

```python
from dataclasses import dataclass
from typing import List, Dict
from yggdrasil.types.field import DataField

@dataclass
class Address:
    street: str
    city: str
    postal_code: str

@dataclass
class Customer:
    id: int
    name: str
    addresses: List[Address]
    preferences: Dict[str, str]

# Create a DataField from the nested structure
field = DataField.from_py_hint("customer", Customer)

# Convert to Arrow schema
arrow_schema = field.to_arrow_schema()
print(arrow_schema)
```

### Integration with PySpark (optional)

Convert between PyArrow and PySpark schemas:

```python
from dataclasses import dataclass
from yggdrasil.types.field import DataField

@dataclass
class Record:
    id: int
    name: str
    value: float

# Create a DataField
field = DataField.from_py_hint("record", Record)

# Convert to Spark schema
spark_field = field.to_spark_field()
print(spark_field)

# Create Spark DataFrame with the schema
spark_df = spark.createDataFrame(data, schema=spark_field.dataType)
```

## Core Components

### DataField

The `DataField` class is the central component of Yggdrasil, providing conversion between Python types, PyArrow, and optionally PySpark. Key methods include:

- `from_py_hint(name, hint, nullable=None, metadata=None)`: Create a DataField from a Python type hint
- `from_arrow_field(field)`: Create a DataField from a PyArrow field
- `from_spark_field(spark_field)`: Create a DataField from a PySpark StructField
- `to_arrow_field()`: Convert to a PyArrow field
- `to_arrow_schema()`: Convert to a PyArrow schema (for struct types with children)
- `to_spark_field()`: Convert to a PySpark StructField

## Architecture

Yggdrasil follows a modular design with these core components:

- **types module**: Contains the core type conversion functionality
  - `field.py`: Implements the DataField class for type conversions
  - `spark_utils.py`: Optional utilities for PySpark integration
- **data module**: Provides higher-level data handling capabilities
- **utils module**: Common utilities and helper functions

## Dependencies

The core dependencies of Yggdrasil include:

- **PyArrow**: Core library for working with Arrow data format
- **Polars**: High-performance DataFrame library built on Arrow
- **tzdata**: Time zone database

Optional dependencies:

- **PySpark**: For Spark integration (included in `[spark]` extras)
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