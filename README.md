# Yggdrasil

Yggdrasil is a multi-language research playground focused on data processing and interoperability. The Python package provides utilities for working with Apache Arrow, type conversions, and data handling.

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

## Features

### Arrow Data Utilities

Yggdrasil provides tools for working with Apache Arrow data types:

```python
import pyarrow as pa
from yggdrasil.data.arrow import ARROW_CAST_REGISTRY

# Example: Cast an array from int32 to int64
source_field = pa.field("values", pa.list_(pa.int32()))
target_field = pa.field("values", pa.list_(pa.int64()))

caster = ARROW_CAST_REGISTRY.get_or_build(source_field, target_field)
array = pa.array([[1, 2, 3]], type=source_field.type)
cast_array = caster.cast_array(array)

print(f"Source type: {source_field.type}")
print(f"Target type: {target_field.type}")
print(f"Values: {cast_array.to_pylist()}")
```

### CLI Commands

Yggdrasil comes with useful command line tools:

```bash
# Get version info
python -m yggdrasil --version

# Arrow casting example
python -m yggdrasil arrow-cast 1 2 3 4 5
```

## Requirements

- Python 3.9 or later
- Dependencies:
  - pyarrow >= 13.0
  - polars
  - tzdata

## Project Structure

```
Yggdrasil/
├── python/
│   ├── src/
│   │   └── yggdrasil/
│   │       ├── data/
│   │       │   └── arrow/
│   │       │       ├── arrow_cast.py
│   │       │       └── init.py
│   │       └── cli.py
│   └── tests/
└── README.md
```

## License

This project is licensed under the terms of the included LICENSE file.