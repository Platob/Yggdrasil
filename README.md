# Yggdrasil

Yggdrasil is a multi-language research playground. Every language-specific
implementation lives in its own top-level directory. The Python package resides
in `python/` and is designed to be installed directly from this repository using
`pip` and the `git+https` scheme.

## Install

To install the Python helpers straight from the repository, point `pip` at the
Git URL:

```bash
pip install "yggdrasil @ git+https://github.com/Platob/Yggdrasil.git"
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- `pip` (comes bundled with Python)

### Installation (macOS/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

### Installation (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
```

## License

This project is licensed under the terms of the included LICENSE file.
