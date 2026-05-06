# `yggrs` — Rust acceleration for Yggdrasil

Optional native kernels for `yggdrasil`. Built with **PyO3 + maturin** on the **`abi3-py310`** stable ABI — one wheel per OS/arch covers Python 3.10, 3.11, 3.12, 3.13.

- **PyPI:** [`yggrs`](https://pypi.org/project/yggrs/)
- **Crate root:** [`rust/`](.) · **Bridge:** [`python/src/yggdrasil/rs.py`](../python/src/yggdrasil/rs.py)
- Pulled in automatically by `pip install ygg`.

---

## Design rules

1. **Python is canonical.** Rust must match Python behavior, not diverge.
2. **Pure-Python fallback always works.** The `ygg` test suite must pass with and without `yggrs` installed.
3. **`yggdrasil/rs.py` is the only bridge.** Feature code never imports from `yggdrasil.rust.*` directly.
4. **Add Rust only to a hot, semantically stable path.**

---

## Use it from Python

```python
from yggdrasil.rs import HAS_RS, utf8_len

print(HAS_RS)                    # True if yggrs is installed
print(utf8_len(["héllo", "🦀"])) # uses Rust if available, falls back otherwise
```

Under the hood, when `yggrs` is installed:

```python
from yggdrasil.rust.data import utf8_len   # fast path
```

---

## Module map

Rust source files mirror the Python package hierarchy. The compiled extension installs as `_yggrs` and is rebound under `yggdrasil.rust.<submodule>` by `yggdrasil/rs.py`.

| Rust file | Python namespace | Status |
|---|---|---|
| `rust/src/data.rs` | `yggdrasil.rust.data` | `utf8_len` shipped |
| `rust/src/io.rs`   | `yggdrasil.rust.io`   | reserved |
| `rust/src/arrow.rs`| `yggdrasil.rust.arrow`| reserved |

Add new kernels in the matching submodule, register them in `lib.rs`, and expose them through `yggdrasil/rs.py` with a Python fallback.

> Why a top-level `_yggrs` import name? `ygg` already ships `yggdrasil/__init__.py` as a regular package, so two wheels can't both own the directory. Shipping the extension as `_yggrs` and rebinding through `yggdrasil/rs.py` keeps both wheels installable side by side.

---

## Build locally

```bash
cd rust
maturin develop --release          # editable build into the active venv
# or compile a release wheel for this platform:
maturin build --release             # → rust/dist/
```

`maturin develop` rebuilds the cdylib, packages it as a wheel, and installs it as `yggrs` in the current venv. The compiled module lands at `_yggrs.abi3.so` (Linux/macOS) or `_yggrs.pyd` (Windows).

`importlib.reload` does **not** swap a compiled extension — restart the interpreter after `maturin develop`.

### Daily loop

| Editing… | Command |
|---|---|
| Python only (`python/src/**`) | nothing — editable install picks up changes on next `import` |
| Rust (`rust/src/**`) | `maturin develop --release` (then restart the interpreter) |
| Cargo deps | `cargo update`, then `maturin develop --release` |

### Toolchain prerequisites

`maturin` shells out to `cargo`, which needs a working C linker for the host target.

- **Linux**: `gcc` + `pkg-config` (`apt install build-essential pkg-config`).
- **macOS**: Xcode Command Line Tools (`xcode-select --install`).
- **Windows**: [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/downloads/?q=build+tools) with the **"Desktop development with C++"** workload (provides `link.exe`). The published Windows wheel on PyPI is MSVC-built.

  Fallback for development only — GNU toolchain:

  ```bash
  rustup target add x86_64-pc-windows-gnu
  maturin develop --release --target x86_64-pc-windows-gnu
  ```

---

## CI / release wheels

[`.github/workflows/publish-native.yml`](../.github/workflows/publish-native.yml) builds five platform wheels in parallel using `PyO3/maturin-action@v1` with `working-directory: rust`:

| Artifact | Target triple | Runner |
|---|---|---|
| `linux-x86_64`  | `x86_64-unknown-linux-gnu`  | `ubuntu-latest` |
| `linux-aarch64` | `aarch64-unknown-linux-gnu` | `ubuntu-latest` + QEMU |
| `windows-x86_64`| `x86_64-pc-windows-msvc`    | `windows-latest` |
| `macos-arm64`   | `aarch64-apple-darwin`      | `macos-latest` |
| `macos-x86_64`  | `x86_64-apple-darwin`       | `macos-13` |

All five wheels plus the sdist are merged and uploaded to PyPI in a single `publish` job. The GitHub Release receives all of them as downloadable assets. The version is read from `python/pyproject.toml` and stamped into `rust/pyproject.toml` and `rust/Cargo.toml` before building.

---

## Adding a kernel — checklist

1. Add the function in `rust/src/<module>.rs` and register it in `lib.rs`.
2. Re-run `maturin develop --release`.
3. Add a Python fallback in `python/src/yggdrasil/<module>/` and import it in `yggdrasil/rs.py`. Decide what the Rust path replaces.
4. Add tests under `python/tests/test_yggdrasil/test_<module>/`. They must pass both with and without `yggrs` installed.
5. Update the module map above.

---

## License

[Apache-2.0](LICENSE).
