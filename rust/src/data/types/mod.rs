/// Rust-side `yggdrasil.data.types`.
///
/// Layout mirrors the Python package:
///
/// | Rust module    | Python module                  |
/// |----------------|--------------------------------|
/// | `id`           | `yggdrasil.data.types.id`      |
/// | `base`         | `yggdrasil.data.types.base`    |
/// | `primitive`    | `yggdrasil.data.types.primitive` |
///
/// Only the shapes that are cheap to express without pyarrow / polars /
/// spark live on the Rust side.  Engine conversions stay in Python as
/// the canonical source of truth (see `yggdrasil/rs.py`); Rust provides
/// fast, allocation-light metadata carriers the Python package can
/// bridge through once the fast path is needed.
use pyo3::prelude::*;

pub mod base;
pub mod id;
pub mod nested;
pub mod object;
pub mod primitive;
pub mod temporal;

use crate::data::engine;

/// `from_arrow(pyarrow_dt)` — reverse bridge from PyArrow DataType to
/// our PyDataType subclasses.
#[pyfunction]
fn from_arrow(py: Python<'_>, dtype: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    engine::from_pyarrow(py, &dtype)
}

/// `from_polars(polars_dt)` — reverse bridge from polars DataType.
#[pyfunction]
fn from_polars(py: Python<'_>, dtype: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    engine::from_polars(py, &dtype)
}

pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    id::register(py, module)?;
    base::register(py, module)?;
    primitive::register(py, module)?;
    object::register(py, module)?;
    temporal::register(py, module)?;
    nested::register(py, module)?;
    module.add_function(wrap_pyfunction!(from_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(from_polars, module)?)?;
    Ok(())
}
