# Rust acceleration (experimental)

This folder contains optional Rust extensions for hot-path operations used by
`yggdrasil`.

## Why

- Keep Python APIs stable while accelerating tight loops.
- Reuse Arrow-compatible memory ideas for future schema and metadata kernels.
- Allow graceful fallback to pure Python when native builds are unavailable.

## Module layout

Rust source files mirror the Python package hierarchy so every kernel is easy
to locate.  Add new kernels to the matching submodule, then register them in
`lib.rs`.

The compiled extension is placed at `yggdrasil/rust.abi3.so` (or `.pyd` on
Windows), extending the `yggdrasil` namespace directly.  No top-level package
collision — `yggrs` (PyPI) slots into the existing `yggdrasil` namespace:

| Rust file              | Python package           | Exposed as                  |
|------------------------|--------------------------|-----------------------------|
| `rust/src/data.rs`     | `yggdrasil/data/`        | `yggdrasil.rust.data`       |
| `rust/src/io.rs`       | `yggdrasil/io/`          | `yggdrasil.rust.io` (future)|
| `rust/src/arrow.rs`    | `yggdrasil/arrow/`       | `yggdrasil.rust.arrow` (future) |

Current kernels:

- `yggdrasil.rust.data.utf8_len(values)` — Unicode character counts for string batches.

## Build locally

`rust/pyproject.toml` uses **maturin** as the PEP 517 build backend, so the
Rust extension is compiled automatically whenever you build or install the
package.

```bash
# Full editable install (compiles Rust extension):
cd rust
maturin develop --extras dev      # recommended during Rust development
# or from the repo root:
pip install -e rust/              # also works via PEP 517

# Release wheel for the current platform (output → rust/dist/):
cd rust
maturin build --release
```

### abi3-py310 stable ABI

`Cargo.toml` enables `pyo3/abi3-py310`.  This means maturin emits a
`cp310-abi3` wheel that is **compatible with Python 3.10, 3.11, 3.12, and
3.13** — one wheel per OS/arch, not per Python version.

### CI / release wheels

The GitHub workflow (`.github/workflows/publish-native.yml`) builds five
platform wheels in parallel using `PyO3/maturin-action@v1` with
`working-directory: rust`:

| Artifact name           | Target triple                  | Runner           |
|-------------------------|--------------------------------|------------------|
| `linux-x86_64`          | `x86_64-unknown-linux-gnu`     | ubuntu-latest    |
| `linux-aarch64`         | `aarch64-unknown-linux-gnu`    | ubuntu-latest + QEMU |
| `windows-x86_64`        | `x86_64-pc-windows-msvc`       | windows-latest   |
| `macos-arm64`           | `aarch64-apple-darwin`         | macos-latest     |
| `macos-x86_64`          | `x86_64-apple-darwin`          | macos-13         |

All five wheels plus the sdist are merged and uploaded to PyPI in a single
`publish` job.  The GitHub Release receives all of them as downloadable assets.



## Python bridge

`yggdrasil/rs.py` is the single entry point.  It imports from the matching
`yggdrasil.rust.<submodule>` and falls back to pure Python when the native
wheel is absent.  Import it as:

```python
from yggdrasil.rs import HAS_RS, utf8_len
```

Under the hood, when `yggrs` is installed, `rs.py` uses:

```python
from yggdrasil.rust.data import utf8_len  # fast path
```

