/// Cast subsystem — mirrors `yggdrasil.data.cast`.
///
/// Structure:
///
/// | Rust module    | Python module                     |
/// |----------------|-----------------------------------|
/// | `options`      | `yggdrasil.data.cast.options`     |
/// | `registry`     | `yggdrasil.data.cast.registry`    |
/// | `primitive`    | Rust-only primitive-to-primitive converters |
///
/// The Rust registry dispatches on `(DataTypeId, DataTypeId)` pairs.
/// That's a pragmatic subset of the Python dispatch (which keys on
/// arbitrary type hints with MRO fallback): primitive-to-primitive
/// casts are the hot path, nested/engine dispatch stays in Python.
use pyo3::prelude::*;

pub mod options;
pub mod primitive;
pub mod registry;

pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    options::register(py, module)?;
    registry::register(py, module)?;
    primitive::install_defaults();
    Ok(())
}
