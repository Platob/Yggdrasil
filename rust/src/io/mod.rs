/// Rust acceleration kernels for `yggdrasil.io`.
///
/// Mirrors the Python package `yggdrasil/io/`. Submodules registered
/// here become importable as `yggdrasil.rust.io.<name>`.
use pyo3::prelude::*;

pub mod url;

pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let url_mod = PyModule::new(py, "url")?;
    url::register(py, &url_mod)?;
    module.add_submodule(&url_mod)?;
    Ok(())
}
