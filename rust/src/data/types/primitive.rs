/// Primitive `DataType` subclasses.
///
/// Mirrors a subset of `yggdrasil.data.types.primitive` — the shapes that
/// are cheap to express without a pyarrow dependency: identity, byte
/// size, precision/scale, unit, timezone.  The expensive engine-side
/// conversions (`to_arrow`, `to_polars`, `to_spark`) are deliberately
/// left on the Python side and not re-implemented here; this module is
/// the Rust-side metadata carrier.
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::base::{self, DataTypeImpl, PyDataType};
use super::id::{DataTypeId, PyDataTypeId};
use crate::data::engine;

// ---------------------------------------------------------------------------
// NullType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "NullType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyNullType {}

impl DataTypeImpl for PyNullType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Null
    }
    fn type_name(&self) -> &'static str {
        "NullType"
    }
}

#[pymethods]
impl PyNullType {
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
        "VOID"
    }

    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::arrow_to_py(py, &engine::null_to_arrow(self))
    }

    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::polars_type_to_py(py, &engine::null_to_polars(self))
    }

    fn __repr__(&self) -> String {
        base::repr_str(self)
    }

    fn __str__(&self) -> &'static str {
        "null"
    }
}

// ---------------------------------------------------------------------------
// BooleanType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "BooleanType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyBooleanType {
    #[pyo3(get)]
    pub byte_size: Option<u32>,
}

impl DataTypeImpl for PyBooleanType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Bool
    }
    fn type_name(&self) -> &'static str {
        "BooleanType"
    }
    fn extra_dict_entries(&self, _py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(b) = self.byte_size {
            dict.set_item("byte_size", b)?;
        }
        Ok(())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        if let Some(b) = self.byte_size {
            tags.push((b"byte_size".to_vec(), b.to_string().into_bytes()));
        }
    }
}

#[pymethods]
impl PyBooleanType {
    #[new]
    #[pyo3(signature = (byte_size = None))]
    fn new(byte_size: Option<u32>) -> (Self, PyDataType) {
        (Self { byte_size }, PyDataType {})
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
        "BOOLEAN"
    }

    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::arrow_to_py(py, &engine::boolean_to_arrow(self))
    }

    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::polars_type_to_py(py, &engine::boolean_to_polars(self))
    }

    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

// ---------------------------------------------------------------------------
// IntegerType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "IntegerType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyIntegerType {
    #[pyo3(get)]
    pub byte_size: Option<u32>,
    #[pyo3(get)]
    pub signed: bool,
}

impl DataTypeImpl for PyIntegerType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Integer
    }
    fn type_name(&self) -> &'static str {
        "IntegerType"
    }
    fn extra_dict_entries(&self, _py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(b) = self.byte_size {
            dict.set_item("byte_size", b)?;
        }
        dict.set_item("signed", self.signed)?;
        Ok(())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        if let Some(b) = self.byte_size {
            tags.push((b"byte_size".to_vec(), b.to_string().into_bytes()));
        }
        tags.push((
            b"signed".to_vec(),
            if self.signed { b"true".to_vec() } else { b"false".to_vec() },
        ));
    }
}

#[pymethods]
impl PyIntegerType {
    #[new]
    #[pyo3(signature = (byte_size = None, signed = true))]
    fn new(byte_size: Option<u32>, signed: bool) -> (Self, PyDataType) {
        (Self { byte_size, signed }, PyDataType {})
    }

    #[getter]
    fn type_id(&self) -> PyDataTypeId {
        base::type_id_py(self)
    }

    /// Width in bits — matches Python `IntegerType.bit_width`.
    #[getter]
    fn bit_width(&self) -> u32 {
        self.byte_size.unwrap_or(8) * 8
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::to_dict(py, self)
    }

    fn autotag<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::autotag(py, self)
    }

    fn to_databricks_ddl(&self) -> &'static str {
        match self.byte_size.unwrap_or(8) {
            1 => "TINYINT",
            2 => "SMALLINT",
            4 => "INT",
            _ => "BIGINT",
        }
    }

    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::arrow_to_py(py, &engine::integer_to_arrow(self))
    }

    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::polars_type_to_py(py, &engine::integer_to_polars(self))
    }

    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

// ---------------------------------------------------------------------------
// FloatingPointType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "FloatingPointType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyFloatingPointType {
    #[pyo3(get)]
    pub byte_size: Option<u32>,
}

impl DataTypeImpl for PyFloatingPointType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Float
    }
    fn type_name(&self) -> &'static str {
        "FloatingPointType"
    }
    fn extra_dict_entries(&self, _py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(b) = self.byte_size {
            dict.set_item("byte_size", b)?;
        }
        Ok(())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        if let Some(b) = self.byte_size {
            tags.push((b"byte_size".to_vec(), b.to_string().into_bytes()));
        }
    }
}

#[pymethods]
impl PyFloatingPointType {
    #[new]
    #[pyo3(signature = (byte_size = None))]
    fn new(byte_size: Option<u32>) -> (Self, PyDataType) {
        (Self { byte_size }, PyDataType {})
    }

    #[getter]
    fn type_id(&self) -> PyDataTypeId {
        base::type_id_py(self)
    }

    #[getter]
    fn bit_width(&self) -> u32 {
        self.byte_size.unwrap_or(8) * 8
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::to_dict(py, self)
    }

    fn autotag<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::autotag(py, self)
    }

    fn to_databricks_ddl(&self) -> &'static str {
        match self.byte_size.unwrap_or(8) {
            2 => "FLOAT",
            4 => "FLOAT",
            _ => "DOUBLE",
        }
    }

    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::arrow_to_py(py, &engine::float_to_arrow(self))
    }

    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::polars_type_to_py(py, &engine::float_to_polars(self))
    }

    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

// ---------------------------------------------------------------------------
// DecimalType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "DecimalType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyDecimalType {
    #[pyo3(get)]
    pub precision: Option<u32>,
    #[pyo3(get)]
    pub scale: Option<i32>,
    #[pyo3(get)]
    pub byte_size: Option<u32>,
}

impl DataTypeImpl for PyDecimalType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Decimal
    }
    fn type_name(&self) -> &'static str {
        "DecimalType"
    }
    fn extra_dict_entries(&self, _py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(p) = self.precision {
            dict.set_item("precision", p)?;
        }
        if let Some(s) = self.scale {
            dict.set_item("scale", s)?;
        }
        if let Some(b) = self.byte_size {
            dict.set_item("byte_size", b)?;
        }
        Ok(())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        if let Some(p) = self.precision {
            tags.push((b"precision".to_vec(), p.to_string().into_bytes()));
        }
        if let Some(s) = self.scale {
            tags.push((b"scale".to_vec(), s.to_string().into_bytes()));
        }
    }
}

#[pymethods]
impl PyDecimalType {
    #[new]
    #[pyo3(signature = (precision = None, scale = None, byte_size = None))]
    fn new(
        precision: Option<u32>,
        scale: Option<i32>,
        byte_size: Option<u32>,
    ) -> (Self, PyDataType) {
        (
            Self {
                precision,
                scale,
                byte_size,
            },
            PyDataType {},
        )
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

    fn to_databricks_ddl(&self) -> String {
        let precision = self.precision.unwrap_or(38);
        let scale = self.scale.unwrap_or(0);
        format!("DECIMAL({precision},{scale})")
    }

    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        let dt = engine::decimal_to_arrow(self)?;
        engine::arrow_to_py(py, &dt)
    }

    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::polars_type_to_py(py, &engine::decimal_to_polars(self))
    }

    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

// ---------------------------------------------------------------------------
// StringType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "StringType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyStringType {
    #[pyo3(get)]
    pub large: bool,
    #[pyo3(get)]
    pub view: bool,
    #[pyo3(get)]
    pub byte_size: Option<u32>,
}

impl DataTypeImpl for PyStringType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::String
    }
    fn type_name(&self) -> &'static str {
        "StringType"
    }
    fn extra_dict_entries(&self, _py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        dict.set_item("large", self.large)?;
        dict.set_item("view", self.view)?;
        if let Some(b) = self.byte_size {
            dict.set_item("byte_size", b)?;
        }
        Ok(())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        if self.large {
            tags.push((b"large".to_vec(), b"true".to_vec()));
        }
        if self.view {
            tags.push((b"view".to_vec(), b"true".to_vec()));
        }
        if let Some(b) = self.byte_size {
            tags.push((b"byte_size".to_vec(), b.to_string().into_bytes()));
        }
    }
}

#[pymethods]
impl PyStringType {
    #[new]
    #[pyo3(signature = (large = false, view = false, byte_size = None))]
    fn new(large: bool, view: bool, byte_size: Option<u32>) -> (Self, PyDataType) {
        (
            Self {
                large,
                view,
                byte_size,
            },
            PyDataType {},
        )
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

    fn to_databricks_ddl(&self) -> String {
        match self.byte_size {
            Some(n) => format!("VARCHAR({n})"),
            None => "STRING".to_string(),
        }
    }

    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::arrow_to_py(py, &engine::string_to_arrow(self))
    }

    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::polars_type_to_py(py, &engine::string_to_polars(self))
    }

    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

// ---------------------------------------------------------------------------
// BinaryType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "BinaryType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyBinaryType {
    #[pyo3(get)]
    pub large: bool,
    #[pyo3(get)]
    pub view: bool,
    #[pyo3(get)]
    pub byte_size: Option<u32>,
}

impl DataTypeImpl for PyBinaryType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Binary
    }
    fn type_name(&self) -> &'static str {
        "BinaryType"
    }
    fn extra_dict_entries(&self, _py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        dict.set_item("large", self.large)?;
        dict.set_item("view", self.view)?;
        if let Some(b) = self.byte_size {
            dict.set_item("byte_size", b)?;
        }
        Ok(())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        if self.large {
            tags.push((b"large".to_vec(), b"true".to_vec()));
        }
        if self.view {
            tags.push((b"view".to_vec(), b"true".to_vec()));
        }
        if let Some(b) = self.byte_size {
            tags.push((b"byte_size".to_vec(), b.to_string().into_bytes()));
        }
    }
}

#[pymethods]
impl PyBinaryType {
    #[new]
    #[pyo3(signature = (large = false, view = false, byte_size = None))]
    fn new(large: bool, view: bool, byte_size: Option<u32>) -> (Self, PyDataType) {
        (
            Self {
                large,
                view,
                byte_size,
            },
            PyDataType {},
        )
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

    fn to_databricks_ddl(&self) -> String {
        match self.byte_size {
            Some(n) => format!("BINARY({n})"),
            None => "BINARY".to_string(),
        }
    }

    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::arrow_to_py(py, &engine::binary_to_arrow(self))
    }

    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        engine::polars_type_to_py(py, &engine::binary_to_polars(self))
    }

    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNullType>()?;
    module.add_class::<PyBooleanType>()?;
    module.add_class::<PyIntegerType>()?;
    module.add_class::<PyFloatingPointType>()?;
    module.add_class::<PyDecimalType>()?;
    module.add_class::<PyStringType>()?;
    module.add_class::<PyBinaryType>()?;
    Ok(())
}
