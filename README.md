# Yggdrasil

Yggdrasil is a multi-language research playground focused on data processing and interoperability. The Python package provides utilities for working with Polars, type conversions, and data handling.

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

### Data Casting Utilities

Yggdrasil provides tools for working with Polars data types:

```python
import polars as pl
from yggdrasil.data import DATA_CAST_REGISTRY

# Example: Cast a series from int32 to int64
source_dtype = pl.Int32
target_dtype = pl.Int64

caster = DATA_CAST_REGISTRY.get_or_build(source_dtype, target_dtype)
series = pl.Series("values", [1, 2, 3], dtype=source_dtype)
cast_series = caster.cast_series(series)

print(f"Source type: {source_dtype}")
print(f"Target type: {target_dtype}")
print(f"Values: {cast_series.to_list()}")
```

### Optional Arrow Support

Arrow interoperability is available as an optional dependency:

```bash
pip install "yggdrasil[arrow] @ git+https://github.com/Platob/Yggdrasil.git"
```

With Arrow support installed, the `SeriesLike` type can accept Arrow arrays:

```python
import pyarrow as pa
import polars as pl
from yggdrasil.data import DataCaster

# Create an Arrow array
arr = pa.array([1, 2, 3], type=pa.int32())

# Set up the caster
caster = DataCaster(source_dtype=pl.Int32, target_dtype=pl.Int64)

# Cast the Arrow array (automatically converts to Polars)
cast_series = caster.cast_series(arr)
```

### CLI Commands

Yggdrasil comes with useful command line tools:

```bash
# Get version info
python -m yggdrasil --version

# Data casting example
python -m yggdrasil data-cast 1 2 3 4 5

# Show a demo table
python -m yggdrasil demo-table
```

## Requirements

- Python 3.9 or later
- Dependencies:
  - polars >= 0.20.0
  - tzdata

- Optional Dependencies:
  - pyarrow >= 13.0 (for Arrow interoperability)

## Project Structure

```
Yggdrasil/
├── python/
│   ├── src/
│   │   └── yggdrasil/
│   │       ├── data/
│   │       │   ├── data_cast.py
│   │       │   └── __init__.py
│   │       └── cli.py
│   └── tests/
└── README.md
```

## License

This project is licensed under the terms of the included LICENSE file.