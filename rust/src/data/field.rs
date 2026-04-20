/// `DataField` — name + dtype + nullability + metadata.
///
/// Mirrors the Python `yggdrasil.data.data_field.Field` class, but only
/// the Rust-portable surface: the metadata container and the
/// copy/with_* builders.  Dataclass/pyarrow/polars/spark factories
/// (`from_arrow_field`, `from_dataclass`, `to_arrow_field`, ...) stay in
/// Python.
///
/// `dtype` is stored as `Py<PyAny>` rather than `Py<PyDataType>` because
/// Python users can legitimately pass adapter objects (e.g. an Arrow
/// dtype) that Python-side `Field.__init__` normalizes — the Rust
/// wrapper does not want to replicate that normalization.  Callers from
/// Rust should pass a `PyDataType` instance.
use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};

use super::constants::{DBX_META_PREFIX, DEFAULT_FIELD_NAME, DEFAULT_VALUE_KEY, TAG_PREFIX};

/// Coerce an arbitrary Python object to `bytes`, matching the Python
/// `_to_bytes` helper's permissiveness: str → utf-8, bytes/bytearray →
/// bytes, anything else → `str(x).encode("utf-8")`.
pub fn to_bytes(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = value.extract::<&[u8]>() {
        return Ok(b.to_vec());
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(s.into_bytes());
    }
    // Fallback: str(value).encode("utf-8")
    let str_fn = py.import("builtins")?.getattr("str")?;
    let as_str: String = str_fn.call1((value,))?.extract()?;
    Ok(as_str.into_bytes())
}

/// Normalize a metadata/tags pair into a single `BTreeMap<Vec<u8>,
/// Vec<u8>>`.  Tags are prefixed with `TAG_PREFIX` on their way in to
/// match the Python storage convention.
pub fn normalize_metadata(
    py: Python<'_>,
    metadata: Option<&Bound<'_, PyDict>>,
    tags: Option<&Bound<'_, PyDict>>,
    default: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<BTreeMap<Vec<u8>, Vec<u8>>>> {
    let mut out: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    if let Some(meta) = metadata {
        for (k, v) in meta.iter() {
            out.insert(to_bytes(py, &k)?, to_bytes(py, &v)?);
        }
    }
    if let Some(tags) = tags {
        for (k, v) in tags.iter() {
            let key_bytes = to_bytes(py, &k)?;
            let mut prefixed = TAG_PREFIX.to_vec();
            prefixed.extend_from_slice(&key_bytes);
            out.insert(prefixed, to_bytes(py, &v)?);
        }
    }
    if let Some(default) = default {
        if !default.is_none() {
            let encoded = encode_default(py, default)?;
            out.insert(DEFAULT_VALUE_KEY.to_vec(), encoded);
        }
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

fn encode_default(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    // Mirrors the Python side's opportunistic JSON encoding for plain
    // scalars; otherwise falls back to str(value).
    let json = py.import("json")?;
    match json.call_method1("dumps", (value,)) {
        Ok(s) => Ok(s.extract::<String>()?.into_bytes()),
        Err(_) => to_bytes(py, value),
    }
}

#[pyclass(module = "yggdrasil.rust.data", name = "DataField")]
pub struct PyDataField {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub nullable: bool,
    pub dtype: Py<PyAny>,
    pub metadata: Option<BTreeMap<Vec<u8>, Vec<u8>>>,
}

impl PyDataField {
    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            name: self.name.clone(),
            nullable: self.nullable,
            dtype: self.dtype.clone_ref(py),
            metadata: self.metadata.clone(),
        }
    }

    /// Public constructor used by sibling modules (nested types).
    pub fn construct(
        py: Python<'_>,
        name: String,
        dtype: Bound<'_, PyAny>,
        nullable: bool,
        metadata: Option<Bound<'_, PyDict>>,
        tags: Option<Bound<'_, PyDict>>,
        default: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Self::new(py, name, dtype, nullable, metadata, tags, default)
    }
}

impl PyDataField {
    /// Shallow clone with optional overrides — the Rust-side helper
    /// behind the Python `copy()` builder.
    pub fn with_overrides(
        &self,
        py: Python<'_>,
        name: Option<String>,
        dtype: Option<Py<PyAny>>,
        nullable: Option<bool>,
        metadata: Option<BTreeMap<Vec<u8>, Vec<u8>>>,
    ) -> Self {
        Self {
            name: name.unwrap_or_else(|| self.name.clone()),
            nullable: nullable.unwrap_or(self.nullable),
            dtype: dtype.unwrap_or_else(|| self.dtype.clone_ref(py)),
            metadata: metadata.or_else(|| self.metadata.clone()),
        }
    }
}

#[pymethods]
impl PyDataField {
    #[new]
    #[pyo3(signature = (name, dtype, nullable = true, metadata = None, tags = None, default = None))]
    fn new(
        py: Python<'_>,
        name: String,
        dtype: Bound<'_, PyAny>,
        nullable: bool,
        metadata: Option<Bound<'_, PyDict>>,
        tags: Option<Bound<'_, PyDict>>,
        default: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let meta = normalize_metadata(py, metadata.as_ref(), tags.as_ref(), default.as_ref())?;
        Ok(Self {
            name,
            nullable,
            dtype: dtype.unbind(),
            metadata: meta,
        })
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.dtype.bind(py).clone()
    }

    #[setter]
    fn set_dtype(&mut self, value: Bound<'_, PyAny>) {
        self.dtype = value.unbind();
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
        self.metadata.as_ref().map(|meta| metadata_to_dict(py, meta))
    }

    #[setter]
    fn set_metadata(
        &mut self,
        py: Python<'_>,
        value: Option<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.metadata = normalize_metadata(py, value.as_ref(), None, None)?;
        Ok(())
    }

    #[getter]
    fn tags<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
        self.metadata.as_ref().map(|meta| {
            let dict = PyDict::new(py);
            for (k, v) in meta {
                if let Some(tail) = k.strip_prefix(TAG_PREFIX) {
                    let _ = dict.set_item(PyBytes::new(py, tail), PyBytes::new(py, v));
                }
            }
            dict
        })
    }

    #[getter]
    fn databricks_metadata<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        if let Some(meta) = &self.metadata {
            for (k, v) in meta {
                if let Some(tail) = k.strip_prefix(DBX_META_PREFIX) {
                    let _ = dict.set_item(PyBytes::new(py, tail), PyBytes::new(py, v));
                }
            }
        }
        dict
    }

    /// Boolean tag read — true iff the tag exists and is a truthy value
    /// (`true`/`1`/`yes` variants), matching the Python `_tag_flag`.
    fn tag_flag(&self, key: &[u8]) -> bool {
        let full = [TAG_PREFIX, key].concat();
        match self.metadata.as_ref().and_then(|m| m.get(&full)) {
            Some(v) => matches!(
                v.as_slice(),
                b"true" | b"True" | b"TRUE" | b"1" | b"yes" | b"YES"
            ),
            None => false,
        }
    }

    #[getter]
    fn partition_by(&self) -> bool {
        self.tag_flag(b"partition_by")
    }

    #[getter]
    fn cluster_by(&self) -> bool {
        self.tag_flag(b"cluster_by")
    }

    #[getter]
    fn primary_key(&self) -> bool {
        self.tag_flag(b"primary_key")
    }

    #[getter]
    fn children_fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let fields = self
            .dtype
            .bind(py)
            .getattr("children_fields")
            .ok()
            .and_then(|v| v.extract::<Vec<Py<PyAny>>>().ok())
            .unwrap_or_default();
        let list = PyList::empty(py);
        for f in fields {
            list.append(f)?;
        }
        Ok(list)
    }

    #[pyo3(signature = (other, check_names = true, check_dtypes = true, check_metadata = true))]
    fn equals(
        &self,
        py: Python<'_>,
        other: &Self,
        check_names: bool,
        check_dtypes: bool,
        check_metadata: bool,
    ) -> PyResult<bool> {
        if check_names && self.name != other.name {
            return Ok(false);
        }
        if self.nullable != other.nullable {
            return Ok(false);
        }
        if check_dtypes {
            let eq = self.dtype.bind(py).eq(other.dtype.bind(py))?;
            if !eq {
                return Ok(false);
            }
        }
        if check_metadata && self.metadata != other.metadata {
            return Ok(false);
        }
        Ok(true)
    }

    #[pyo3(signature = (**kwargs))]
    pub fn copy(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut name = None;
        let mut dtype = None;
        let mut nullable = None;
        let mut metadata = self.metadata.clone();

        if let Some(kw) = kwargs {
            if let Some(v) = kw.get_item("name")? {
                name = Some(v.extract::<String>()?);
            }
            if let Some(v) = kw.get_item("dtype")? {
                dtype = Some(v.unbind());
            }
            if let Some(v) = kw.get_item("nullable")? {
                nullable = Some(v.extract::<bool>()?);
            }
            let mut md = kw.get_item("metadata")?;
            let tags = kw.get_item("tags")?;
            let default = kw.get_item("default")?;
            if md.is_some() || tags.is_some() || default.is_some() {
                let md_dict = md
                    .as_ref()
                    .and_then(|v| v.downcast::<PyDict>().ok())
                    .cloned();
                let tg_dict = tags.as_ref().and_then(|v| v.downcast::<PyDict>().ok()).cloned();
                metadata = normalize_metadata(
                    py,
                    md_dict.as_ref(),
                    tg_dict.as_ref(),
                    default.as_ref(),
                )?;
                md.take();
            }
        }

        Ok(self.with_overrides(py, name, dtype, nullable, metadata))
    }

    fn with_name(&self, py: Python<'_>, name: String) -> Self {
        self.with_overrides(py, Some(name), None, None, self.metadata.clone())
    }

    fn with_dtype(&self, py: Python<'_>, dtype: Bound<'_, PyAny>) -> Self {
        self.with_overrides(py, None, Some(dtype.unbind()), None, self.metadata.clone())
    }

    fn with_nullable(&self, py: Python<'_>, nullable: bool) -> Self {
        self.with_overrides(py, None, None, Some(nullable), self.metadata.clone())
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("name", &self.name)?;
        let dtype_dict = self.dtype.bind(py).call_method0("to_dict").ok();
        if let Some(d) = dtype_dict {
            dict.set_item("dtype", d)?;
        }
        dict.set_item("nullable", self.nullable)?;
        if let Some(meta) = self.metadata.as_ref() {
            dict.set_item("metadata", metadata_to_dict(py, meta))?;
        }
        Ok(dict)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let dtype_repr = self
            .dtype
            .bind(py)
            .repr()
            .ok()
            .and_then(|b| b.extract::<String>().ok())
            .unwrap_or_else(|| "?".to_string());
        format!(
            "DataField(name={:?}, dtype={}, nullable={})",
            self.name, dtype_repr, self.nullable
        )
    }

    fn __eq__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        match other.downcast::<PyDataField>() {
            Ok(o) => self.equals(py, &o.borrow(), true, true, true),
            Err(_) => Ok(false),
        }
    }
}

fn metadata_to_dict<'py>(py: Python<'py>, meta: &BTreeMap<Vec<u8>, Vec<u8>>) -> Bound<'py, PyDict> {
    let dict = PyDict::new(py);
    for (k, v) in meta {
        let _ = dict.set_item(PyBytes::new(py, k), PyBytes::new(py, v));
    }
    dict
}

/// `field(...)` convenience factory — mirrors the Python free function.
#[pyfunction]
#[pyo3(signature = (name, dtype, *, nullable = true, metadata = None, tags = None, default = None))]
pub fn field(
    py: Python<'_>,
    name: String,
    dtype: Bound<'_, PyAny>,
    nullable: bool,
    metadata: Option<Bound<'_, PyDict>>,
    tags: Option<Bound<'_, PyDict>>,
    default: Option<Bound<'_, PyAny>>,
) -> PyResult<PyDataField> {
    PyDataField::new(py, name, dtype, nullable, metadata, tags, default)
}

pub fn default_name() -> &'static str {
    DEFAULT_FIELD_NAME
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDataField>()?;
    module.add_function(wrap_pyfunction!(field, module)?)?;
    Ok(())
}
