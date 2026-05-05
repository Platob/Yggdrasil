/// `yggrs` — Rust acceleration for `yggdrasil`.
///
/// Ships as the top-level extension module `_yggrs`. The Python side
/// re-exports its submodules through `yggdrasil.rs` (the bridge) and
/// hangs the public surface off `yggdrasil.rust.<name>` so callers can
/// keep writing `from yggdrasil.rust.io import url`.
///
/// | Rust module                   | Python re-export    |
/// |-------------------------------|---------------------|
/// | `_yggrs.data`                 | `yggdrasil.rust.data` |
/// | `_yggrs.io`                   | `yggdrasil.rust.io`   |
///
/// Top-level naming was chosen over the previous `yggdrasil.rust`
/// dotted module-name because `yggdrasil` is a regular package (with
/// its own `__init__.py`) shipped by `ygg`. Two wheels cannot both
/// own that package directory; a separate compiled module avoids the
/// namespace-package vs. regular-package conflict and lets `ygg` and
/// `yggrs` install cleanly side by side.
///
/// Add new accelerated kernels inside the matching submodule, then call
/// its `register()` from the `#[pymodule]` below.
use pyo3::prelude::*;

mod data;
mod io;

#[pymodule]
fn _yggrs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── data submodule (mirrors yggdrasil.data) ───────────────────────────
    let data_mod = PyModule::new(py, "data")?;
    data::register(py, &data_mod)?;
    m.add_submodule(&data_mod)?;

    // ── io submodule (mirrors yggdrasil.io) ───────────────────────────────
    let io_mod = PyModule::new(py, "io")?;
    io::register(py, &io_mod)?;
    m.add_submodule(&io_mod)?;

    Ok(())
}
