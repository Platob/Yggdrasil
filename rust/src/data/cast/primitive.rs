/// Primitive-to-primitive Rust converters.
///
/// Covers the cross-cast matrix between `BOOL`, `INTEGER`, `FLOAT`,
/// `DECIMAL` (as strings), `STRING`, `BINARY` — the shapes that benefit
/// from running in Rust instead of Python bytecode.  Temporal, nested,
/// and engine-typed casts stay in Python.
///
/// Semantics match the Python `yggdrasil.data.types.primitive`
/// `_convert_pyobj` methods where practical:
///   - `safe=True` propagates to parse/coerce failures (raise instead
///     of silently truncating).  `safe=False` (default) matches arrow's
///     `cast(..., safe=False)` laxity — truncation allowed.
///   - `None` is preserved end-to-end.
use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyFloat, PyInt};

use super::options::PyCastOptions;
use super::registry::register_converter as register;
use crate::data::types::id::DataTypeId;

static INSTALLED: OnceLock<()> = OnceLock::new();

pub fn install_defaults() {
    INSTALLED.get_or_init(install);
}

fn install() {
    // Null source → any target: always None.
    for tgt in [
        DataTypeId::Bool,
        DataTypeId::Integer,
        DataTypeId::Float,
        DataTypeId::Decimal,
        DataTypeId::String,
        DataTypeId::Binary,
    ] {
        register(Some(DataTypeId::Null), tgt, to_null);
    }

    // Bool → *
    register(Some(DataTypeId::Bool), DataTypeId::Integer, bool_to_int);
    register(Some(DataTypeId::Bool), DataTypeId::Float, bool_to_float);
    register(Some(DataTypeId::Bool), DataTypeId::String, bool_to_str);

    // Integer → *
    register(Some(DataTypeId::Integer), DataTypeId::Bool, int_to_bool);
    register(Some(DataTypeId::Integer), DataTypeId::Float, int_to_float);
    register(Some(DataTypeId::Integer), DataTypeId::String, int_to_str);
    register(Some(DataTypeId::Integer), DataTypeId::Binary, int_to_binary);

    // Float → *
    register(Some(DataTypeId::Float), DataTypeId::Bool, float_to_bool);
    register(Some(DataTypeId::Float), DataTypeId::Integer, float_to_int);
    register(Some(DataTypeId::Float), DataTypeId::String, float_to_str);

    // String → *
    register(Some(DataTypeId::String), DataTypeId::Bool, str_to_bool);
    register(Some(DataTypeId::String), DataTypeId::Integer, str_to_int);
    register(Some(DataTypeId::String), DataTypeId::Float, str_to_float);
    register(Some(DataTypeId::String), DataTypeId::Binary, str_to_binary);

    // Binary → String
    register(Some(DataTypeId::Binary), DataTypeId::String, binary_to_str);
}

fn is_none(value: &Bound<'_, PyAny>) -> bool {
    value.is_none()
}

fn py_none(py: Python<'_>) -> Py<PyAny> {
    py.None()
}

// ── Null source ────────────────────────────────────────────────────────────

fn to_null(py: Python<'_>, _value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    Ok(py_none(py))
}

// ── Bool source ────────────────────────────────────────────────────────────

fn bool_to_int(py: Python<'_>, value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let b: bool = value.extract()?;
    Ok((b as i64).into_pyobject(py)?.into_any().unbind())
}

fn bool_to_float(py: Python<'_>, value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let b: bool = value.extract()?;
    Ok(PyFloat::new(py, if b { 1.0 } else { 0.0 }).into_any().unbind())
}

fn bool_to_str(py: Python<'_>, value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let b: bool = value.extract()?;
    let s = if b { "true" } else { "false" };
    Ok(s.into_pyobject(py)?.into_any().unbind())
}

// ── Integer source ─────────────────────────────────────────────────────────

fn int_to_bool(py: Python<'_>, value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let n: i64 = value.extract()?;
    Ok((n != 0).into_pyobject(py)?.to_owned().into_any().unbind())
}

fn int_to_float(py: Python<'_>, value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let n: i64 = value.extract()?;
    Ok(PyFloat::new(py, n as f64).into_any().unbind())
}

fn int_to_str(py: Python<'_>, value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let n: i64 = value.extract()?;
    Ok(n.to_string().into_pyobject(py)?.into_any().unbind())
}

fn int_to_binary(py: Python<'_>, value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let n: i64 = value.extract()?;
    Ok(PyBytes::new(py, n.to_string().as_bytes()).into_any().unbind())
}

// ── Float source ───────────────────────────────────────────────────────────

fn float_to_bool(py: Python<'_>, value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let f: f64 = value.extract()?;
    Ok((f != 0.0 && !f.is_nan()).into_pyobject(py)?.to_owned().into_any().unbind())
}

fn float_to_int(py: Python<'_>, value: Bound<'_, PyAny>, options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let f: f64 = value.extract()?;
    if f.is_nan() || f.is_infinite() {
        if options.safe {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot cast NaN/Inf to integer with safe=True",
            ));
        }
        return Ok(py_none(py));
    }
    if options.safe && f.fract() != 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cannot cast non-integer float {f} to integer with safe=True"
        )));
    }
    Ok((f.trunc() as i64).into_pyobject(py)?.into_any().unbind())
}

fn float_to_str(py: Python<'_>, value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let f: f64 = value.extract()?;
    Ok(format!("{f}").into_pyobject(py)?.into_any().unbind())
}

// ── String source ──────────────────────────────────────────────────────────

const BOOL_TRUE: &[&str] = &["true", "t", "1", "yes", "y", "on"];
const BOOL_FALSE: &[&str] = &["false", "f", "0", "no", "n", "off", ""];

fn str_to_bool(py: Python<'_>, value: Bound<'_, PyAny>, options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let s: String = value.extract()?;
    let lower = s.trim().to_ascii_lowercase();
    if BOOL_TRUE.contains(&lower.as_str()) {
        return Ok(true.into_pyobject(py)?.to_owned().into_any().unbind());
    }
    if BOOL_FALSE.contains(&lower.as_str()) {
        return Ok(false.into_pyobject(py)?.to_owned().into_any().unbind());
    }
    if options.safe {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cannot parse {s:?} as bool — expected one of {:?}",
            BOOL_TRUE.iter().chain(BOOL_FALSE.iter()).collect::<Vec<_>>()
        )));
    }
    Ok(py_none(py))
}

fn str_to_int(py: Python<'_>, value: Bound<'_, PyAny>, options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let s: String = value.extract()?;
    match s.trim().parse::<i64>() {
        Ok(n) => Ok(n.into_pyobject(py)?.into_any().unbind()),
        Err(_) => {
            // Fall through to Python int() so "1_000", "0x10", and
            // arbitrary-precision literals all still work.
            let builtins = py.import("builtins")?;
            let int_fn = builtins.getattr("int")?;
            match int_fn.call1((s.clone(),)) {
                Ok(v) => Ok(v.unbind()),
                Err(e) => {
                    if options.safe {
                        Err(e)
                    } else {
                        Ok(py_none(py))
                    }
                }
            }
        }
    }
}

fn str_to_float(py: Python<'_>, value: Bound<'_, PyAny>, options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let s: String = value.extract()?;
    match s.trim().parse::<f64>() {
        Ok(f) => Ok(PyFloat::new(py, f).into_any().unbind()),
        Err(_) => {
            let builtins = py.import("builtins")?;
            let float_fn = builtins.getattr("float")?;
            match float_fn.call1((s.clone(),)) {
                Ok(v) => Ok(v.unbind()),
                Err(e) => {
                    if options.safe {
                        Err(e)
                    } else {
                        Ok(py_none(py))
                    }
                }
            }
        }
    }
}

fn str_to_binary(py: Python<'_>, value: Bound<'_, PyAny>, _options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let s: String = value.extract()?;
    Ok(PyBytes::new(py, s.as_bytes()).into_any().unbind())
}

// ── Binary source ──────────────────────────────────────────────────────────

fn binary_to_str(py: Python<'_>, value: Bound<'_, PyAny>, options: &PyCastOptions) -> PyResult<Py<PyAny>> {
    if is_none(&value) { return Ok(py_none(py)); }
    let bytes: Vec<u8> = value.extract()?;
    match String::from_utf8(bytes) {
        Ok(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        Err(e) => {
            if options.safe {
                Err(pyo3::exceptions::PyUnicodeDecodeError::new_err(format!(
                    "{e}"
                )))
            } else {
                let bytes = e.into_bytes();
                Ok(String::from_utf8_lossy(&bytes)
                    .into_owned()
                    .into_pyobject(py)?
                    .into_any()
                    .unbind())
            }
        }
    }
}

// Make the PyInt import used even if only as a marker; the import is
// kept for future integer-specific casts.
#[allow(dead_code)]
fn _pyint_marker(_x: &PyInt) {}
