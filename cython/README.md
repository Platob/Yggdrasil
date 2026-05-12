# yggcy

Cython acceleration kernels for [`ygg`](https://pypi.org/project/ygg/).

This package ships compiled wheels carrying performance-critical kernels
that `ygg` picks up automatically at import time (`yggdrasil/cy.py`).
Environments without `yggcy` keep working on the pure-Python fallback
paths in `ygg` itself — every kernel here is mirrored by an equivalent
Python implementation in the matching `yggdrasil.*` module.

## Layout

```
cython/
  pyproject.toml      # PEP 621 + setuptools/Cython build backend
  setup.py            # cythonize() entry point
  src/_yggcy/
    __init__.py
    io/
      __init__.py
      url.pyx         # URL parsing + percent-encoding kernels
```

The top-level compiled package is `_yggcy` (not `yggdrasil`) so two
wheels never compete for the same `yggdrasil/` directory — `ygg` ships
the Python package, `yggcy` ships the C extensions, and the bridge
re-exports `_yggcy.<sub>` under `yggdrasil.cy.<sub>`.

## Building locally

```bash
cd cython
pip install -e .[dev]
# or for a fresh wheel:
pip install build && python -m build
```
