/// Nested types — `ArrayType`, `MapType`, `StructType`.
///
/// Mirrors `yggdrasil.data.types.nested`.  Nested types hold child
/// `DataField`s; we store them as `Py<PyDataField>` so the Rust layer
/// can reach into them without re-importing the Python object.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::base::{self, DataTypeImpl, PyDataType};
use super::id::{DataTypeId, PyDataTypeId};
use crate::data::engine;
use crate::data::field::PyDataField;

/// Build a default item-field for nested types whose user doesn't care
/// about the element name.  Matches `DEFAULT_FIELD_NAME = ""` and
/// nullable=true.
fn default_item_field(py: Python<'_>, dtype: Bound<'_, PyAny>) -> PyResult<Py<PyDataField>> {
    let field = PyDataField::construct(
        py,
        crate::data::field::default_name().to_string(),
        dtype,
        true,
        None,
        None,
        None,
    )?;
    Py::new(py, field)
}

// ---------------------------------------------------------------------------
// ArrayType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, name = "ArrayType")]
pub struct PyArrayType {
    pub item_field: Py<PyDataField>,
    #[pyo3(get)]
    pub list_size: Option<u32>,
    #[pyo3(get)]
    pub large: bool,
    #[pyo3(get)]
    pub view: bool,
}

impl DataTypeImpl for PyArrayType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Array
    }
    fn type_name(&self) -> &'static str {
        "ArrayType"
    }
    fn extra_dict_entries(&self, py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let f = self.item_field.bind(py);
        let child = f.call_method0("to_dict")?;
        dict.set_item("item_field", child)?;
        if let Some(s) = self.list_size {
            dict.set_item("list_size", s)?;
        }
        dict.set_item("large", self.large)?;
        dict.set_item("view", self.view)?;
        Ok(())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        if let Some(s) = self.list_size {
            tags.push((b"list_size".to_vec(), s.to_string().into_bytes()));
        }
        if self.large {
            tags.push((b"large".to_vec(), b"true".to_vec()));
        }
        if self.view {
            tags.push((b"view".to_vec(), b"true".to_vec()));
        }
    }
}

#[pymethods]
impl PyArrayType {
    #[new]
    #[pyo3(signature = (item_field, list_size = None, large = false, view = false))]
    fn new(
        py: Python<'_>,
        item_field: Bound<'_, PyAny>,
        list_size: Option<u32>,
        large: bool,
        view: bool,
    ) -> PyResult<(Self, PyDataType)> {
        let item = coerce_field(py, item_field)?;
        Ok((
            Self {
                item_field: item,
                list_size,
                large,
                view,
            },
            PyDataType {},
        ))
    }

    #[getter]
    fn item_field<'py>(&self, py: Python<'py>) -> Bound<'py, PyDataField> {
        self.item_field.bind(py).clone()
    }

    #[getter]
    fn type_id(&self) -> PyDataTypeId {
        base::type_id_py(self)
    }

    #[getter]
    fn children_fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        list.append(self.item_field.bind(py))?;
        Ok(list)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::to_dict(py, self)
    }
    fn autotag<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::autotag(py, self)
    }
    fn to_databricks_ddl(&self, py: Python<'_>) -> PyResult<String> {
        let inner = databricks_ddl_of(py, &self.item_field)?;
        Ok(format!("ARRAY<{inner}>"))
    }
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dt = engine::array_to_arrow(py, self)?;
        engine::arrow_to_py(py, &dt)
    }
    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dt = engine::array_to_polars(py, self)?;
        engine::polars_type_to_py(py, &dt)
    }
    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
    fn __eq__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(o) = other.downcast::<PyArrayType>() else {
            return Ok(false);
        };
        let o = o.borrow();
        if self.list_size != o.list_size || self.large != o.large || self.view != o.view {
            return Ok(false);
        }
        self.item_field.bind(py).eq(o.item_field.bind(py))
    }
}

// ---------------------------------------------------------------------------
// MapType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, name = "MapType")]
pub struct PyMapType {
    pub key_field: Py<PyDataField>,
    pub value_field: Py<PyDataField>,
    #[pyo3(get)]
    pub keys_sorted: bool,
}

impl DataTypeImpl for PyMapType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Map
    }
    fn type_name(&self) -> &'static str {
        "MapType"
    }
    fn extra_dict_entries(&self, py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        dict.set_item("key_field", self.key_field.bind(py).call_method0("to_dict")?)?;
        dict.set_item(
            "value_field",
            self.value_field.bind(py).call_method0("to_dict")?,
        )?;
        dict.set_item("keys_sorted", self.keys_sorted)?;
        Ok(())
    }
    fn extra_tags(&self, tags: &mut Vec<(Vec<u8>, Vec<u8>)>) {
        if self.keys_sorted {
            tags.push((b"keys_sorted".to_vec(), b"true".to_vec()));
        }
    }
}

#[pymethods]
impl PyMapType {
    #[new]
    #[pyo3(signature = (key_field, value_field, keys_sorted = false))]
    fn new(
        py: Python<'_>,
        key_field: Bound<'_, PyAny>,
        value_field: Bound<'_, PyAny>,
        keys_sorted: bool,
    ) -> PyResult<(Self, PyDataType)> {
        let key = coerce_field(py, key_field)?;
        let value = coerce_field(py, value_field)?;
        Ok((
            Self {
                key_field: key,
                value_field: value,
                keys_sorted,
            },
            PyDataType {},
        ))
    }

    #[getter]
    fn key_field<'py>(&self, py: Python<'py>) -> Bound<'py, PyDataField> {
        self.key_field.bind(py).clone()
    }

    #[getter]
    fn value_field<'py>(&self, py: Python<'py>) -> Bound<'py, PyDataField> {
        self.value_field.bind(py).clone()
    }

    #[getter]
    fn type_id(&self) -> PyDataTypeId {
        base::type_id_py(self)
    }

    #[getter]
    fn children_fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        list.append(self.key_field.bind(py))?;
        list.append(self.value_field.bind(py))?;
        Ok(list)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::to_dict(py, self)
    }
    fn autotag<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::autotag(py, self)
    }
    fn to_databricks_ddl(&self, py: Python<'_>) -> PyResult<String> {
        let k = databricks_ddl_of(py, &self.key_field)?;
        let v = databricks_ddl_of(py, &self.value_field)?;
        Ok(format!("MAP<{k}, {v}>"))
    }
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dt = engine::map_to_arrow(py, self)?;
        engine::arrow_to_py(py, &dt)
    }
    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dt = engine::map_to_polars(py, self)?;
        engine::polars_type_to_py(py, &dt)
    }
    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
    fn __eq__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(o) = other.downcast::<PyMapType>() else {
            return Ok(false);
        };
        let o = o.borrow();
        if self.keys_sorted != o.keys_sorted {
            return Ok(false);
        }
        if !self.key_field.bind(py).eq(o.key_field.bind(py))? {
            return Ok(false);
        }
        self.value_field.bind(py).eq(o.value_field.bind(py))
    }
}

// ---------------------------------------------------------------------------
// StructType
// ---------------------------------------------------------------------------

#[pyclass(module = "yggdrasil.rust.data.types", extends = PyDataType, frozen, name = "StructType")]
pub struct PyStructType {
    pub fields: Vec<Py<PyDataField>>,
}

impl DataTypeImpl for PyStructType {
    fn type_id(&self) -> DataTypeId {
        DataTypeId::Struct
    }
    fn type_name(&self) -> &'static str {
        "StructType"
    }
    fn extra_dict_entries(&self, py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let items = PyList::empty(py);
        for f in &self.fields {
            items.append(f.bind(py).call_method0("to_dict")?)?;
        }
        dict.set_item("fields", items)?;
        Ok(())
    }
}

#[pymethods]
impl PyStructType {
    #[new]
    #[pyo3(signature = (fields = None))]
    fn new(py: Python<'_>, fields: Option<Bound<'_, PyAny>>) -> PyResult<(Self, PyDataType)> {
        let collected = match fields {
            None => Vec::new(),
            Some(obj) => coerce_field_list(py, obj)?,
        };
        Ok((Self { fields: collected }, PyDataType {}))
    }

    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        let bound: Vec<Bound<'py, PyDataField>> =
            self.fields.iter().map(|f| f.bind(py).clone()).collect();
        PyTuple::new(py, bound).expect("tuple construction")
    }

    #[getter]
    fn type_id(&self) -> PyDataTypeId {
        base::type_id_py(self)
    }

    #[getter]
    fn children_fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for f in &self.fields {
            list.append(f.bind(py))?;
        }
        Ok(list)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::to_dict(py, self)
    }
    fn autotag<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        base::autotag(py, self)
    }
    fn to_databricks_ddl(&self, py: Python<'_>) -> PyResult<String> {
        let mut parts: Vec<String> = Vec::with_capacity(self.fields.len());
        for f in &self.fields {
            let b = f.bind(py);
            let name: String = b.getattr("name")?.extract()?;
            let inner = databricks_ddl_of(py, f)?;
            parts.push(format!("{name}: {inner}"));
        }
        Ok(format!("STRUCT<{}>", parts.join(", ")))
    }
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dt = engine::struct_to_arrow(py, self)?;
        engine::arrow_to_py(py, &dt)
    }
    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dt = engine::struct_to_polars(py, self)?;
        engine::polars_type_to_py(py, &dt)
    }
    fn __repr__(&self) -> String {
        base::repr_str(self)
    }
    fn __eq__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(o) = other.downcast::<PyStructType>() else {
            return Ok(false);
        };
        let o = o.borrow();
        if self.fields.len() != o.fields.len() {
            return Ok(false);
        }
        for (a, b) in self.fields.iter().zip(o.fields.iter()) {
            if !a.bind(py).eq(b.bind(py))? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn coerce_field(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<Py<PyDataField>> {
    if let Ok(existing) = value.extract::<Py<PyDataField>>() {
        return Ok(existing);
    }
    // If it's a DataType-like, wrap into a default-named Field.
    if value.hasattr("type_id")? {
        return default_item_field(py, value);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected DataField or DataType as item/value field",
    ))
}

fn coerce_field_list(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<Vec<Py<PyDataField>>> {
    let mut out = Vec::new();
    let iter = value.try_iter()?;
    for item in iter {
        out.push(coerce_field(py, item?)?);
    }
    Ok(out)
}

fn databricks_ddl_of(py: Python<'_>, field: &Py<PyDataField>) -> PyResult<String> {
    let dtype = field.bind(py).getattr("dtype")?;
    let ddl = dtype.call_method0("to_databricks_ddl")?;
    ddl.extract()
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyArrayType>()?;
    module.add_class::<PyMapType>()?;
    module.add_class::<PyStructType>()?;
    Ok(())
}
