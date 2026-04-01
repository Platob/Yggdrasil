/// `yggrs` — optional Rust acceleration for `yggdrasil`.
///
/// Source layout mirrors the Python package so each Rust submodule is easy to
/// locate:
///
/// | Rust module                   | Python package      |
/// |-------------------------------|---------------------|
/// | `yggdrasil.rust.data`         | `yggdrasil.data`    |
/// | `yggdrasil.rust.io`  (future) | `yggdrasil.io`      |
/// | `yggdrasil.rust.arrow` (future)| `yggdrasil.arrow`  |
///
/// The compiled extension is placed at `yggdrasil/rust.abi3.so` (or `.pyd`
/// on Windows) so it is importable as `yggdrasil.rust`.  The function must be
/// named `rust` so the linker exports `PyInit_rust`, which Python looks up
/// when importing `yggdrasil.rust`.
///
/// Add new accelerated kernels inside the matching submodule, then call its
/// `register()` from the `#[pymodule]` below.
use pyo3::prelude::*;

mod data;

#[pymodule]
fn rust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── data submodule (mirrors yggdrasil.data) ───────────────────────────
    let data_mod = PyModule::new(py, "data")?;
    data::register(py, &data_mod)?;
    m.add_submodule(&data_mod)?;

    Ok(())
}
