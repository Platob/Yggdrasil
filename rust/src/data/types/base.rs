/// `DataType` base — the abstract root of the type hierarchy.
///
/// Mirrors `yggdrasil.data.types.base.DataType`.  The Rust side exposes
/// a `#[pyclass(subclass)]` so concrete primitive/nested classes can
/// extend it from Rust (`extends = PyDataType`) and users can also
/// subclass it from Python if they want.
///
/// Only the stable, engine-agnostic surface lives here: `type_id`,
/// equality/hash, a `to_dict()` round-trip, and `autotag()`.  Engine
/// conversions (`to_arrow`, `to_polars`, `to_spark`) are intentionally
/// kept in Python for now — they need pyarrow/polars bindings and
/// belong in the Python package as the canonical source of truth.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::id::{DataTypeId, PyDataTypeId};

/// Rust-side interface implemented by every concrete data type.
///
/// Concrete `#[pyclass]` structs implement this trait so the shared
/// methods on `PyDataType` (e.g. `to_dict`, `autotag`) can delegate
/// without a second Python round-trip.
pub trait DataTypeImpl: Send + Sync {
    fn type_id(&self) -> DataTypeId;

    /// Human-readable name of the concrete class, used in `__repr__`.
    fn type_name(&self) -> &'static str;

    /// Extra fields to merge into `to_dict()` — concrete types override
    /// to expose dtype-specific state (byte_size, precision, unit, ...).
    fn extra_dict_entries(&self, _py: Python<'_>, _dict: &Bound<'_, PyDict>) -> PyResult<()> {
        Ok(())
    }

    /// Extra tag entries merged into `autotag()` alongside `kind`.
    fn extra_tags(&self, _tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {}
}

/// Python-facing abstract base.  Not constructible directly — concrete
/// types do that via `extends = PyDataType` subclasses.
#[pyclass(module = "yggdrasil.rust.data.types", subclass, name = "DataType")]
pub struct PyDataType {}

#[pymethods]
impl PyDataType {
    #[new]
    fn new() -> PyResult<Self> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "DataType is abstract — instantiate a concrete subclass instead",
        ))
    }

    /// `DataType()` with no args returns self; any args is an error.
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        slf: Bound<'py, PyDataType>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDataType>> {
        let has_kwargs = kwargs.map(|k| !k.is_empty()).unwrap_or(false);
        if args.is_empty() && !has_kwargs {
            return Ok(slf);
        }
        Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot call DataType with args or kwargs",
        ))
    }
}

/// Helpers that concrete `#[pymethods]` blocks delegate to so each
/// primitive class doesn't re-derive the shared dict/tag plumbing.
pub fn to_dict<'py, T: DataTypeImpl>(
    py: Python<'py>,
    imp: &T,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let id = imp.type_id();
    dict.set_item("id", id.value())?;
    dict.set_item("name", id.name())?;
    imp.extra_dict_entries(py, &dict)?;
    Ok(dict)
}

pub fn autotag<'py, T: DataTypeImpl>(
    py: Python<'py>,
    imp: &T,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let kind = imp.type_id().name().to_ascii_lowercase().into_bytes();
    dict.set_item(b"kind", kind)?;
    let mut extras: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    imp.extra_tags(&mut extras);
    for (key, value) in extras {
        dict.set_item(key, value)?;
    }
    Ok(dict)
}

pub fn type_id_py<T: DataTypeImpl>(imp: &T) -> PyDataTypeId {
    PyDataTypeId::from(imp.type_id())
}

pub fn repr_str<T: DataTypeImpl>(imp: &T) -> String {
    format!("{}()", imp.type_name())
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDataType>()?;
    Ok(())
}
