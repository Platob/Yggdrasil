/// `Schema` — ordered dict of `DataField`s + schema-level metadata.
///
/// Mirrors the Rust-portable surface of `yggdrasil.data.schema.Schema`:
/// field ordering, name lookup, append/extend, equality, copy with
/// overrides.  Engine bridges (Arrow/Polars/Spark) stay in Python.
use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use super::field::{normalize_metadata, PyDataField};

#[pyclass(module = "yggdrasil.rust.data", name = "Schema")]
pub struct PySchema {
    pub fields: Vec<Py<PyDataField>>,
    pub metadata: Option<BTreeMap<Vec<u8>, Vec<u8>>>,
}

#[pymethods]
impl PySchema {
    #[new]
    #[pyo3(signature = (fields = None, metadata = None, tags = None))]
    fn new(
        py: Python<'_>,
        fields: Option<Bound<'_, PyAny>>,
        metadata: Option<Bound<'_, PyDict>>,
        tags: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let collected = match fields {
            None => Vec::new(),
            Some(obj) => coerce_field_list(py, obj)?,
        };
        reject_duplicates(py, &collected)?;
        let meta = normalize_metadata(py, metadata.as_ref(), tags.as_ref(), None)?;
        Ok(Self {
            fields: collected,
            metadata: meta,
        })
    }

    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        let bound: Vec<Bound<'py, PyDataField>> =
            self.fields.iter().map(|f| f.bind(py).clone()).collect();
        PyTuple::new(py, bound).expect("tuple construction")
    }

    #[getter]
    fn children_fields<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.fields(py)
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
        self.metadata.as_ref().map(|meta| {
            let dict = PyDict::new(py);
            for (k, v) in meta {
                let _ = dict.set_item(PyBytes::new(py, k), PyBytes::new(py, v));
            }
            dict
        })
    }

    fn names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.fields
            .iter()
            .map(|f| f.bind(py).borrow().name.clone())
            .map(Ok)
            .collect()
    }

    fn field_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.names(py)
    }

    fn field<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyDataField>>> {
        for f in &self.fields {
            if f.bind(py).borrow().name == name {
                return Ok(Some(f.bind(py).clone()));
            }
        }
        Ok(None)
    }

    fn __contains__(&self, py: Python<'_>, name: &str) -> bool {
        self.fields.iter().any(|f| f.bind(py).borrow().name == name)
    }

    fn __len__(&self) -> usize {
        self.fields.len()
    }

    fn __iter__<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let names = slf.names(slf.py())?;
        Ok(PyList::new(slf.py(), names)?.try_iter()?.into_any())
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDataField>> {
        if let Ok(idx) = key.extract::<isize>() {
            let len = self.fields.len() as isize;
            let i = if idx < 0 { idx + len } else { idx };
            if i < 0 || i >= len {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Schema index {idx} out of range for {len}-field schema"
                )));
            }
            return Ok(self.fields[i as usize].bind(py).clone());
        }
        let name: String = key.extract()?;
        match self.field(py, &name)? {
            Some(f) => Ok(f),
            None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Schema has no field named {name:?}"
            ))),
        }
    }

    #[pyo3(signature = (*more))]
    fn append(&mut self, py: Python<'_>, more: &Bound<'_, PyTuple>) -> PyResult<()> {
        for item in more.iter() {
            let field: Py<PyDataField> = item.extract()?;
            if self.__contains__(py, &field.bind(py).borrow().name) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Schema already contains field {:?}",
                    field.bind(py).borrow().name
                )));
            }
            self.fields.push(field);
        }
        Ok(())
    }

    fn extend(&mut self, py: Python<'_>, fields: Bound<'_, PyAny>) -> PyResult<()> {
        let more = coerce_field_list(py, fields)?;
        for f in more {
            if self.__contains__(py, &f.bind(py).borrow().name) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Schema already contains field {:?}",
                    f.bind(py).borrow().name
                )));
            }
            self.fields.push(f);
        }
        Ok(())
    }

    fn equals(
        &self,
        py: Python<'_>,
        other: &Self,
    ) -> PyResult<bool> {
        if self.fields.len() != other.fields.len() {
            return Ok(false);
        }
        for (a, b) in self.fields.iter().zip(other.fields.iter()) {
            if !a.bind(py).eq(b.bind(py))? {
                return Ok(false);
            }
        }
        Ok(self.metadata == other.metadata)
    }

    #[pyo3(signature = (**kwargs))]
    fn copy(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut fields = self.fields.iter().map(|f| f.clone_ref(py)).collect::<Vec<_>>();
        let mut metadata = self.metadata.clone();

        if let Some(kw) = kwargs {
            if let Some(v) = kw.get_item("fields")? {
                fields = coerce_field_list(py, v)?;
                reject_duplicates(py, &fields)?;
            }
            let md = kw.get_item("metadata")?;
            let tg = kw.get_item("tags")?;
            if md.is_some() || tg.is_some() {
                let md_dict = md.as_ref().and_then(|v| v.downcast::<PyDict>().ok()).cloned();
                let tg_dict = tg.as_ref().and_then(|v| v.downcast::<PyDict>().ok()).cloned();
                metadata = normalize_metadata(py, md_dict.as_ref(), tg_dict.as_ref(), None)?;
            }
        }

        Ok(Self { fields, metadata })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let items = PyList::empty(py);
        for f in &self.fields {
            items.append(f.bind(py).call_method0("to_dict")?)?;
        }
        dict.set_item("fields", items)?;
        if let Some(meta) = &self.metadata {
            let m = PyDict::new(py);
            for (k, v) in meta {
                m.set_item(PyBytes::new(py, k), PyBytes::new(py, v))?;
            }
            dict.set_item("metadata", m)?;
        }
        Ok(dict)
    }

    fn __eq__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        match other.downcast::<PySchema>() {
            Ok(o) => self.equals(py, &o.borrow()),
            Err(_) => Ok(false),
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let names: Vec<String> = self
            .fields
            .iter()
            .map(|f| f.bind(py).borrow().name.clone())
            .collect();
        format!("Schema({} fields: {:?})", self.fields.len(), names)
    }
}

fn coerce_field_list(_py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<Vec<Py<PyDataField>>> {
    let mut out = Vec::new();
    for item in value.try_iter()? {
        out.push(item?.extract::<Py<PyDataField>>()?);
    }
    Ok(out)
}

fn reject_duplicates(py: Python<'_>, fields: &[Py<PyDataField>]) -> PyResult<()> {
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::with_capacity(fields.len());
    for f in fields {
        let name = f.bind(py).borrow().name.clone();
        if !seen.insert(name.clone()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Duplicate field name in Schema: {name:?}"
            )));
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (fields = None, *, metadata = None, tags = None))]
pub fn schema(
    py: Python<'_>,
    fields: Option<Bound<'_, PyAny>>,
    metadata: Option<Bound<'_, PyDict>>,
    tags: Option<Bound<'_, PyDict>>,
) -> PyResult<PySchema> {
    PySchema::new(py, fields, metadata, tags)
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySchema>()?;
    module.add_function(wrap_pyfunction!(schema, module)?)?;
    Ok(())
}
