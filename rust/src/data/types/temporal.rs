/// Temporal primitives — mirrors `yggdrasil.data.types.temporal`.
///
/// Four concrete types live here: `DateType`, `TimeType`, `TimestampType`,
/// `DurationType`.  They share a `unit` + optional `tz` carried as
/// strings; the engine-side semantics (arrow/polars/spark) stay on the
/// Python side.
///
/// `unit` values follow the Python vocabulary: `"s"`, `"ms"`, `"us"`,
/// `"ns"` for sub-second types, `"d"` for `DateType`.  `tz` is a raw
/// IANA zone name (or empty string meaning "no tz set").
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::base::{self, DataTypeImpl, PyDataType};
use super::id::{DataTypeId, PyDataTypeId};

fn validate_unit(unit: &str, allow_day: bool) -> PyResult<()> {
    match unit {
        "s" | "ms" | "us" | "ns" => Ok(()),
        "d" if allow_day => Ok(()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid temporal unit {unit:?} — expected one of \"s\", \"ms\", \"us\", \"ns\"{}",
            if allow_day { " or \"d\"" } else { "" }
        ))),
    }
}

fn put_temporal_extras(
    dict: &Bound<'_, PyDict>,
    byte_size: Option<u32>,
    unit: &str,
    tz: Option<&str>,
) -> PyResult<()> {
    if let Some(b) = byte_size {
        dict.set_item("byte_size", b)?;
    }
    dict.set_item("unit", unit)?;
    if let Some(tz) = tz {
        dict.set_item("tz", tz)?;
    }
    Ok(())
}

fn push_temporal_tags(
    tags: &mut Vec<(Vec<u8>, Vec<u8>)>,
    byte_size: Option<u32>,
    unit: &str,
    tz: Option<&str>,
) {
    if let Some(b) = byte_size {
        tags.push((b"byte_size".to_vec(), b.to_string().into_bytes()));
    }
    tags.push((b"unit".to_vec(), unit.as_bytes().to_vec()));
    if let Some(tz) = tz {
        tags.push((b"tz".to_vec(), tz.as_bytes().to_vec()));
    }
}

// ---------------------------------------------------------------------------
// DateType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "DateType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyDateType {
    #[pyo3(get)]
    pub unit: String,
    #[pyo3(get)]
    pub tz: Option<String>,
    #[pyo3(get)]
    pub byte_size: Option<u32>,
}

impl DataTypeImpl for PyDateType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Date
    }
    fn type_name(&self) -> &'static str {
        "DateType"
    }
    fn extra_dict_entries(&self, _py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        put_temporal_extras(dict, self.byte_size, &self.unit, self.tz.as_deref())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        push_temporal_tags(tags, self.byte_size, &self.unit, self.tz.as_deref());
    }
}

#[pymethods]
impl PyDateType {
    #[new]
    #[pyo3(signature = (unit = "d".to_string(), tz = None, byte_size = None))]
    fn new(unit: String, tz: Option<String>, byte_size: Option<u32>) -> PyResult<(Self, PyDataType)> {
        validate_unit(&unit, true)?;
        Ok((Self { unit, tz, byte_size }, PyDataType {}))
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
        "DATE"
    }
    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

// ---------------------------------------------------------------------------
// TimeType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "TimeType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyTimeType {
    #[pyo3(get)]
    pub unit: String,
    #[pyo3(get)]
    pub tz: Option<String>,
    #[pyo3(get)]
    pub byte_size: Option<u32>,
}

impl DataTypeImpl for PyTimeType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Time
    }
    fn type_name(&self) -> &'static str {
        "TimeType"
    }
    fn extra_dict_entries(&self, _py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        put_temporal_extras(dict, self.byte_size, &self.unit, self.tz.as_deref())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        push_temporal_tags(tags, self.byte_size, &self.unit, self.tz.as_deref());
    }
}

#[pymethods]
impl PyTimeType {
    #[new]
    #[pyo3(signature = (unit = "us".to_string(), tz = None, byte_size = None))]
    fn new(unit: String, tz: Option<String>, byte_size: Option<u32>) -> PyResult<(Self, PyDataType)> {
        validate_unit(&unit, false)?;
        Ok((Self { unit, tz, byte_size }, PyDataType {}))
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
        "TIMESTAMP_NTZ"
    }
    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

// ---------------------------------------------------------------------------
// TimestampType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "TimestampType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyTimestampType {
    #[pyo3(get)]
    pub unit: String,
    #[pyo3(get)]
    pub tz: Option<String>,
    #[pyo3(get)]
    pub byte_size: Option<u32>,
}

impl DataTypeImpl for PyTimestampType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Timestamp
    }
    fn type_name(&self) -> &'static str {
        "TimestampType"
    }
    fn extra_dict_entries(&self, _py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        put_temporal_extras(dict, self.byte_size, &self.unit, self.tz.as_deref())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        push_temporal_tags(tags, self.byte_size, &self.unit, self.tz.as_deref());
    }
}

#[pymethods]
impl PyTimestampType {
    #[new]
    #[pyo3(signature = (unit = "us".to_string(), tz = None, byte_size = None))]
    fn new(unit: String, tz: Option<String>, byte_size: Option<u32>) -> PyResult<(Self, PyDataType)> {
        validate_unit(&unit, false)?;
        Ok((Self { unit, tz, byte_size }, PyDataType {}))
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
        match self.tz.as_deref() {
            Some(_) => "TIMESTAMP",
            None => "TIMESTAMP_NTZ",
        }
    }
    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

// ---------------------------------------------------------------------------
// DurationType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, eq, hash, name = "DurationType")]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PyDurationType {
    #[pyo3(get)]
    pub unit: String,
    #[pyo3(get)]
    pub byte_size: Option<u32>,
}

impl DataTypeImpl for PyDurationType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Duration
    }
    fn type_name(&self) -> &'static str {
        "DurationType"
    }
    fn extra_dict_entries(&self, _py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        put_temporal_extras(dict, self.byte_size, &self.unit, None)
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        push_temporal_tags(tags, self.byte_size, &self.unit, None);
    }
}

#[pymethods]
impl PyDurationType {
    #[new]
    #[pyo3(signature = (unit = "us".to_string(), byte_size = None))]
    fn new(unit: String, byte_size: Option<u32>) -> PyResult<(Self, PyDataType)> {
        validate_unit(&unit, false)?;
        Ok((Self { unit, byte_size }, PyDataType {}))
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
        "INTERVAL"
    }
    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDateType>()?;
    module.add_class::<PyTimeType>()?;
    module.add_class::<PyTimestampType>()?;
    module.add_class::<PyDurationType>()?;
    Ok(())
}
