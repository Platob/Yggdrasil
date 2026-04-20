/// `ObjectType` — the variant/opaque type.
///
/// Mirrors `yggdrasil.data.types.primitive.ObjectType`.  Used as a
/// catch-all for heterogeneous Python values that don't map cleanly to
/// any primitive/nested shape.  Arrow stands in with `large_binary`;
/// Spark with `BinaryType` — both stay on the Python side.
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::base::{self, DataTypeImpl, PyDataType};
use super::id::{DataTypeId, PyDataTypeId};
use crate::data::engine;

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "ObjectType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyObjectType {}

impl DataTypeImpl for PyObjectType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Object
    }
    fn type_name(&self) -> &'static str {
        "ObjectType"
    }
}

#[pymethods]
impl PyObjectType {
    #[new]
    fn new() -> (Self, PyDataType) {
        (Self {}, PyDataType {})
    }

    #[getter]
    fn type_id(&self) -> PyDataTypeId {
        base::type_id_py(self)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::to_dict(py, self)
    }
    fn autotag<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::autotag(py, self)
    }
    fn to_databricks_ddl(&self) -> &'static str {
        "BINARY"
    }
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::arrow_to_py(py, &engine::object_to_arrow(self))
    }
    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::polars_type_to_py(py, &engine::object_to_polars(self))
    }
    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyObjectType>()?;
    Ok(())
}
