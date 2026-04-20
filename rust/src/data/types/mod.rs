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
pub mod primitive;

pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    id::register(py, module)?;
    base::register(py, module)?;
    primitive::register(py, module)?;
    Ok(())
}
