/// Cast registry — `(DataTypeId, DataTypeId) -> Converter`.
///
/// The Rust registry is a pragmatic subset of the Python one.  The
/// Python side keys converters on arbitrary type hints (Python types,
/// `DataType` subclasses, `Any` wildcards) with MRO-based fallback and
/// one-hop composition; the Rust side keys on the `DataTypeId`
/// discriminant pair, which is plenty for primitive-to-primitive
/// casts — the only shape Rust tries to accelerate.
///
/// Dispatch order:
/// 1. Exact `(src, tgt)` match.
/// 2. Identity (`src == tgt`).
/// 3. `(Any, tgt)` wildcard fallback.
/// 4. One-hop composition `src -> mid -> tgt`.
/// 5. Python delegation: if none of the above fire and the target
///    exposes a `_cast_rust_any` method, call it — this is how we
///    hand off to the Python cast layer for shapes Rust doesn't cover.
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use pyo3::prelude::*;
use pyo3::types::PyAny;

use super::options::PyCastOptions;
use crate::data::types::id::{from_pyany as type_id_from_pyany, DataTypeId, PyDataTypeId};

/// A rust converter takes the input value, the CastOptions, and produces
/// a Python value.  Converters live behind `Arc` so the registry can be
/// read without cloning the function.
pub type Converter = fn(
    py: Python<'_>,
    value: Bound<'_, PyAny>,
    options: &PyCastOptions,
) -> PyResult<Py<PyAny>>;

/// Key form used by both exact and wildcard entries.  `source = None`
/// represents the `Any` wildcard — the entry fires whenever the exact
/// `(source, target)` lookup misses.
type Key = (Option<DataTypeId>, DataTypeId);

static REGISTRY: OnceLock<RwLock<HashMap<Key, Converter>>> = OnceLock::new();

fn registry() -> &'static RwLock<HashMap<Key, Converter>> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register a converter.  `source = None` installs an `Any -> target`
/// wildcard.  Overwrites any prior converter at the same key.
pub fn register_converter(source: Option<DataTypeId>, target: DataTypeId, f: Converter) {
    registry().write().expect("registry poisoned").insert((source, target), f);
}

/// Look up a converter for `(source, target)`.
pub fn lookup(source: DataTypeId, target: DataTypeId) -> Option<Converter> {
    let map = registry().read().expect("registry poisoned");
    if let Some(f) = map.get(&(Some(source), target)) {
        return Some(*f);
    }
    if let Some(f) = map.get(&(None, target)) {
        return Some(*f);
    }
    None
}

fn find_one_hop(source: DataTypeId, target: DataTypeId) -> Option<(DataTypeId, Converter, Converter)> {
    let map = registry().read().expect("registry poisoned");
    for mid in DataTypeId::ALL {
        if mid == source || mid == target {
            continue;
        }
        let Some(a) = map.get(&(Some(source), mid)).or_else(|| map.get(&(None, mid))) else {
            continue;
        };
        let Some(b) = map.get(&(Some(mid), target)).or_else(|| map.get(&(None, target))) else {
            continue;
        };
        return Some((mid, *a, *b));
    }
    None
}

fn identity_convert(
    _py: Python<'_>,
    value: Bound<'_, PyAny>,
    _options: &PyCastOptions,
) -> PyResult<Py<PyAny>> {
    Ok(value.unbind())
}

/// `convert(value, source, target, options=None)` — Python entry
/// point.  Accepts `DataTypeId` enums or ints for `source`/`target`.
#[pyfunction]
#[pyo3(signature = (value, source, target, options = None))]
pub fn convert(
    py: Python<'_>,
    value: Bound<'_, PyAny>,
    source: Bound<'_, PyAny>,
    target: Bound<'_, PyAny>,
    options: Option<Py<PyCastOptions>>,
) -> PyResult<Py<PyAny>> {
    let src_id = type_id_from_pyany(&source)?.inner;
    let tgt_id = type_id_from_pyany(&target)?.inner;

    // Snapshot the options once up-front.  `clone_with` is cheap: it
    // refcount-bumps the `Py<PyDataField>` handles and copies a handful
    // of small scalars.  This lets the converters take a plain
    // `&PyCastOptions` without fighting borrow-guard lifetimes.
    let opts_local = match options.as_ref() {
        Some(o) => o.bind(py).borrow().clone_with(py),
        None => PyCastOptions::default(),
    };
    let opts_ref = &opts_local;

    if src_id == tgt_id {
        return identity_convert(py, value, opts_ref);
    }

    if let Some(f) = lookup(src_id, tgt_id) {
        return f(py, value, opts_ref);
    }

    if let Some((_mid, first, second)) = find_one_hop(src_id, tgt_id) {
        let intermediate = first(py, value, opts_ref)?;
        return second(py, intermediate.into_bound(py), opts_ref);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "No converter registered for {} -> {}",
        src_id.name(),
        tgt_id.name()
    )))
}

/// `has_converter(source, target)` — introspection helper.
#[pyfunction]
pub fn has_converter(source: Bound<'_, PyAny>, target: Bound<'_, PyAny>) -> PyResult<bool> {
    let src = type_id_from_pyany(&source)?.inner;
    let tgt = type_id_from_pyany(&target)?.inner;
    if src == tgt {
        return Ok(true);
    }
    if lookup(src, tgt).is_some() {
        return Ok(true);
    }
    Ok(find_one_hop(src, tgt).is_some())
}

/// List all registered `(source, target)` pairs — for debugging.
#[pyfunction]
pub fn registered_pairs() -> Vec<(Option<PyDataTypeId>, PyDataTypeId)> {
    registry()
        .read()
        .expect("registry poisoned")
        .keys()
        .map(|(s, t)| (s.map(PyDataTypeId::from), PyDataTypeId::from(*t)))
        .collect()
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(convert, module)?)?;
    module.add_function(wrap_pyfunction!(has_converter, module)?)?;
    module.add_function(wrap_pyfunction!(registered_pairs, module)?)?;
    Ok(())
}
