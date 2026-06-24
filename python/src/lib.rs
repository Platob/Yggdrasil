//! PyO3 bindings for ygg — a thin layer over `ygg-core`.
//!
//! These classes only marshal types across the boundary and expose a
//! Pythonic surface (`snake_case` accessors, `__str__`/`__repr__`). All
//! parsing logic lives in `ygg-core`; parse errors are mapped to
//! `ValueError`.

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ygg_core::{Uri as CoreUri, Url as CoreUrl};

/// A URI split into its RFC 3986 components.
#[pyclass(name = "Uri", module = "ygg", frozen)]
#[derive(Clone)]
struct Uri {
    inner: CoreUri,
}

#[pymethods]
impl Uri {
    /// Parse a string into its URI components.
    #[staticmethod]
    fn parse(value: &str) -> PyResult<Self> {
        CoreUri::parse(value)
            .map(|inner| Uri { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn scheme(&self) -> Option<String> {
        self.inner.scheme.clone()
    }

    #[getter]
    fn authority(&self) -> Option<String> {
        self.inner.authority.clone()
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path.clone()
    }

    #[getter]
    fn query(&self) -> Option<String> {
        self.inner.query.clone()
    }

    #[getter]
    fn fragment(&self) -> Option<String> {
        self.inner.fragment.clone()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Uri.parse({:?})", self.inner.to_string())
    }

    fn __richcmp__(&self, other: &Uri, op: CompareOp, py: Python<'_>) -> PyObject {
        match op {
            CompareOp::Eq => (self.inner == other.inner).into_py(py),
            CompareOp::Ne => (self.inner != other.inner).into_py(py),
            _ => py.NotImplemented(),
        }
    }
}

/// A parsed URL: a located URI with its authority decomposed.
#[pyclass(name = "Url", module = "ygg", frozen)]
#[derive(Clone)]
struct Url {
    inner: CoreUrl,
}

#[pymethods]
impl Url {
    /// Parse a string into a URL (requires a scheme and a host).
    #[staticmethod]
    fn parse(value: &str) -> PyResult<Self> {
        CoreUrl::parse(value)
            .map(|inner| Url { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn scheme(&self) -> String {
        self.inner.scheme.clone()
    }

    #[getter]
    fn username(&self) -> Option<String> {
        self.inner.username.clone()
    }

    #[getter]
    fn password(&self) -> Option<String> {
        self.inner.password.clone()
    }

    #[getter]
    fn host(&self) -> String {
        self.inner.host.clone()
    }

    #[getter]
    fn port(&self) -> Option<u16> {
        self.inner.port
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path.clone()
    }

    #[getter]
    fn query(&self) -> Option<String> {
        self.inner.query.clone()
    }

    #[getter]
    fn fragment(&self) -> Option<String> {
        self.inner.fragment.clone()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Url.parse({:?})", self.inner.to_string())
    }

    fn __richcmp__(&self, other: &Url, op: CompareOp, py: Python<'_>) -> PyObject {
        match op {
            CompareOp::Eq => (self.inner == other.inner).into_py(py),
            CompareOp::Ne => (self.inner != other.inner).into_py(py),
            _ => py.NotImplemented(),
        }
    }
}

/// The `ygg` extension module.
#[pymodule]
fn ygg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", ygg_core::VERSION)?;
    m.add_class::<Uri>()?;
    m.add_class::<Url>()?;
    Ok(())
}
