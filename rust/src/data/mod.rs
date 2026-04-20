/// Rust acceleration kernels for `yggdrasil.data`.
///
/// Mirrors the Python package `yggdrasil/data/`.  Add new data-layer kernels
/// here; register them in `register()` so they appear under
/// `yggdrasil.rust.data`.
use pyo3::prelude::*;

pub mod cast;
pub mod constants;
pub mod field;
pub mod schema;
pub mod types;

/// Return the Unicode character count for each element.
///
/// `None` values pass through as `None`.  Equivalent to
/// `[None if v is None else len(v) for v in values]` but faster for
/// large batches because the entire iteration runs in Rust without
/// per-element Python overhead.
#[pyfunction]
pub fn utf8_len(values: Vec<Option<String>>) -> Vec<Option<usize>> {
    values
        .into_iter()
        .map(|v| v.map(|s| s.chars().count()))
        .collect()
}

/// Register all `data` kernels on *module*.
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(utf8_len, module)?)?;

    constants::register(py, module)?;
    field::register(py, module)?;
    schema::register(py, module)?;

    let types_mod = PyModule::new(py, "types")?;
    types::register(py, &types_mod)?;
    module.add_submodule(&types_mod)?;

    let cast_mod = PyModule::new(py, "cast")?;
    cast::register(py, &cast_mod)?;
    module.add_submodule(&cast_mod)?;

    Ok(())
}
