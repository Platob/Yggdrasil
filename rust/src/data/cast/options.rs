/// `CastOptions` — the shared options carrier for cast dispatch.
///
/// Mirrors the Rust-portable surface of
/// `yggdrasil.data.cast.options.CastOptions`: source/target fields,
/// safety/strictness flags, and datetime parsing formats.  Engine
/// pools (pyarrow memory pool, polars strategy objects) stay on the
/// Python side — Rust just carries the flags.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::data::field::PyDataField;

const DEFAULT_DATETIME_FORMATS: &[&str] = &[
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
];

#[pyclass(module = "yggdrasil.rust.data.cast", name = "CastOptions")]
pub struct PyCastOptions {
    pub source_field: Option<Py<PyDataField>>,
    pub target_field: Option<Py<PyDataField>>,
    #[pyo3(get, set)]
    pub safe: bool,
    #[pyo3(get, set)]
    pub strict_match_names: bool,
    #[pyo3(get, set)]
    pub add_missing_fields: bool,
    #[pyo3(get, set)]
    pub add_missing_columns: bool,
    #[pyo3(get, set)]
    pub allow_add_columns: bool,
    pub datetime_formats: Vec<String>,
}

impl Default for PyCastOptions {
    fn default() -> Self {
        Self {
            source_field: None,
            target_field: None,
            safe: false,
            strict_match_names: false,
            add_missing_fields: true,
            add_missing_columns: true,
            allow_add_columns: false,
            datetime_formats: DEFAULT_DATETIME_FORMATS.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl PyCastOptions {
    /// Explicit GIL-aware clone — `Py<T>` is not `Clone`, so callers
    /// must provide a token.
    pub fn clone_with(&self, py: Python<'_>) -> Self {
        Self {
            source_field: self.source_field.as_ref().map(|f| f.clone_ref(py)),
            target_field: self.target_field.as_ref().map(|f| f.clone_ref(py)),
            safe: self.safe,
            strict_match_names: self.strict_match_names,
            add_missing_fields: self.add_missing_fields,
            add_missing_columns: self.add_missing_columns,
            allow_add_columns: self.allow_add_columns,
            datetime_formats: self.datetime_formats.clone(),
        }
    }
}

#[pymethods]
impl PyCastOptions {
    #[new]
    #[pyo3(signature = (
        source_field = None,
        target_field = None,
        safe = false,
        strict_match_names = false,
        add_missing_fields = true,
        add_missing_columns = true,
        allow_add_columns = false,
        datetime_formats = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        source_field: Option<Py<PyDataField>>,
        target_field: Option<Py<PyDataField>>,
        safe: bool,
        strict_match_names: bool,
        add_missing_fields: bool,
        add_missing_columns: bool,
        allow_add_columns: bool,
        datetime_formats: Option<Vec<String>>,
    ) -> Self {
        Self {
            source_field,
            target_field,
            safe,
            strict_match_names,
            add_missing_fields,
            add_missing_columns,
            allow_add_columns,
            datetime_formats: datetime_formats
                .unwrap_or_else(|| DEFAULT_DATETIME_FORMATS.iter().map(|s| s.to_string()).collect()),
        }
    }

    #[getter]
    fn source_field<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDataField>> {
        self.source_field.as_ref().map(|f| f.bind(py).clone())
    }

    #[setter]
    fn set_source_field(&mut self, value: Option<Py<PyDataField>>) {
        self.source_field = value;
    }

    #[getter]
    fn target_field<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDataField>> {
        self.target_field.as_ref().map(|f| f.bind(py).clone())
    }

    #[setter]
    fn set_target_field(&mut self, value: Option<Py<PyDataField>>) {
        self.target_field = value;
    }

    #[getter]
    fn datetime_formats<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::new(py, &self.datetime_formats).expect("tuple construction")
    }

    #[setter]
    fn set_datetime_formats(&mut self, value: Vec<String>) {
        self.datetime_formats = value;
    }

    /// Shallow clone with field/flag overrides.  Matches the Python
    /// `CastOptions.copy(**kwargs)` builder.
    #[pyo3(signature = (**kwargs))]
    fn copy(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut out = self.clone_with(py);
        let Some(kw) = kwargs else {
            return Ok(out);
        };
        if let Some(v) = kw.get_item("source_field")? {
            out.source_field = if v.is_none() { None } else { Some(v.extract()?) };
        }
        if let Some(v) = kw.get_item("target_field")? {
            out.target_field = if v.is_none() { None } else { Some(v.extract()?) };
        }
        if let Some(v) = kw.get_item("safe")? {
            out.safe = v.extract()?;
        }
        if let Some(v) = kw.get_item("strict_match_names")? {
            out.strict_match_names = v.extract()?;
        }
        if let Some(v) = kw.get_item("add_missing_fields")? {
            out.add_missing_fields = v.extract()?;
        }
        if let Some(v) = kw.get_item("add_missing_columns")? {
            out.add_missing_columns = v.extract()?;
        }
        if let Some(v) = kw.get_item("allow_add_columns")? {
            out.allow_add_columns = v.extract()?;
        }
        if let Some(v) = kw.get_item("datetime_formats")? {
            out.datetime_formats = v.extract()?;
        }
        Ok(out)
    }

    fn with_source(&self, py: Python<'_>, field: Py<PyDataField>) -> Self {
        let mut c = self.clone_with(py);
        c.source_field = Some(field);
        c
    }

    fn with_target(&self, py: Python<'_>, field: Py<PyDataField>) -> Self {
        let mut c = self.clone_with(py);
        c.target_field = Some(field);
        c
    }
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCastOptions>()?;
    Ok(())
}
