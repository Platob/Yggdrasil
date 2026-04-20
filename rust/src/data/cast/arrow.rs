/// Rust-side Arrow array casting via `arrow::compute::kernels::cast`.
///
/// Exposed as `yggdrasil.rust.data.cast.cast_arrow_array(array, target,
/// safe=False)`.  Accepts any PyArrow `Array` (or `ChunkedArray`) and a
/// target — either a `yggdrasil.rust.data.types.*` instance, a
/// `pyarrow.DataType`, or a string the PyArrow parser understands.
/// Returns a new PyArrow `Array` / `ChunkedArray`.
///
/// The cast kernel runs in Rust using the arrow compute crate; the
/// wheel links against a single shared arrow implementation so casts
/// dispatch without round-tripping through Python.
use arrow::array::{make_array, ArrayData};
use arrow::compute::kernels::cast::{cast_with_options, CastOptions as ArrowCastOptions};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::ArrayRef;
use arrow_schema::DataType as ArrowDataType;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::data::engine;

fn resolve_target(py: Python<'_>, target: &Bound<'_, PyAny>) -> PyResult<ArrowDataType> {
    // 1) PyArrow DataType: already what we want.
    if let Ok(dt) = ArrowDataType::from_pyarrow_bound(target) {
        return Ok(dt);
    }
    // 2) Our PyDataType subclass: call `.to_arrow()` then unwrap.
    if target.hasattr("to_arrow").unwrap_or(false) {
        if let Ok(arrow_any) = target.call_method0("to_arrow") {
            return ArrowDataType::from_pyarrow_bound(&arrow_any);
        }
    }
    // 3) Polars DataType (class or instance): route through our
    //    engine::from_polars which round-trips via pyarrow.
    if is_polars_dtype(py, target) {
        // Polars accepts either the class (pl.Int32) or an instance
        // (pl.Int32()).  Normalize to instance: class objects have
        // `__class__` == `type`, instances don't.
        let pl = py.import("polars")?;
        let series_fn = pl.getattr("Series")?;
        let empty = pyo3::types::PyList::empty(py);
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("name", "")?;
        kwargs.set_item("values", empty)?;
        kwargs.set_item("dtype", target.clone())?;
        let series = series_fn.call((), Some(&kwargs))?;
        let arrow_series = series.call_method0("to_arrow")?;
        let arrow_type = arrow_series.getattr("type")?;
        return ArrowDataType::from_pyarrow_bound(&arrow_type);
    }
    // 4) String: leave parsing to PyArrow (`pa.type_for_alias`).
    if let Ok(name) = target.extract::<String>() {
        let pa = py.import("pyarrow")?;
        let dt = pa.call_method1("type_for_alias", (name,))?;
        return ArrowDataType::from_pyarrow_bound(&dt);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "cast target must be a pyarrow DataType, a polars DataType, a yggdrasil DataType, or a type alias string",
    ))
}

fn is_polars_dtype(py: Python<'_>, target: &Bound<'_, PyAny>) -> bool {
    let Ok(pl) = py.import("polars") else {
        return false;
    };
    let Ok(base) = pl.getattr("DataType") else {
        return false;
    };
    // Accept both `pl.Int32` (class) and `pl.Int32()` (instance) — for
    // classes, `issubclass` works; for instances, `isinstance`.
    let builtins = match py.import("builtins") {
        Ok(b) => b,
        Err(_) => return false,
    };
    let isinstance = builtins.getattr("isinstance").ok();
    let issubclass = builtins.getattr("issubclass").ok();
    if let Some(f) = isinstance.as_ref() {
        if let Ok(r) = f.call1((target.clone(), base.clone())) {
            if r.is_truthy().unwrap_or(false) {
                return true;
            }
        }
    }
    if let Some(f) = issubclass.as_ref() {
        if target.is_instance_of::<pyo3::types::PyType>() {
            if let Ok(r) = f.call1((target.clone(), base)) {
                if r.is_truthy().unwrap_or(false) {
                    return true;
                }
            }
        }
    }
    false
}

fn cast_one(input: &ArrayRef, target: &ArrowDataType, safe: bool) -> PyResult<ArrayRef> {
    // Python / pyarrow convention: `safe=True` means "raise on loss".
    // arrow-rs convention: `safe=true` means "replace failures with
    // NULL" (i.e. lossy is OK).  Invert the flag on the way in.
    let options = ArrowCastOptions {
        safe: !safe,
        ..Default::default()
    };
    cast_with_options(input, target, &options)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Arrow cast error: {e}")))
}

#[pyfunction]
#[pyo3(signature = (array, target, safe = false))]
pub fn cast_arrow_array(
    py: Python<'_>,
    array: Bound<'_, PyAny>,
    target: Bound<'_, PyAny>,
    safe: bool,
) -> PyResult<Py<PyAny>> {
    let target_dt = resolve_target(py, &target)?;

    // ChunkedArray path: iterate chunks via PyArrow.
    let is_chunked = array
        .getattr("__class__")
        .and_then(|c| c.getattr("__name__"))
        .and_then(|n| n.extract::<String>())
        .map(|name| name == "ChunkedArray")
        .unwrap_or(false);

    if is_chunked {
        let pa = py.import("pyarrow")?;
        let chunks = array.getattr("chunks")?;
        let chunk_iter = chunks.try_iter()?;
        let mut out_chunks: Vec<Py<PyAny>> = Vec::new();
        for chunk in chunk_iter {
            let chunk = chunk?;
            let data = ArrayData::from_pyarrow_bound(&chunk)?;
            let arr = make_array(data);
            let casted = cast_one(&arr, &target_dt, safe)?;
            out_chunks.push(casted.to_data().to_pyarrow(py)?);
        }
        let out_list = PyList::new(py, out_chunks)?;
        let target_py = target_dt.to_pyarrow(py)?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("type", target_py)?;
        let chunked = pa
            .getattr("chunked_array")?
            .call((out_list,), Some(&kwargs))?;
        return Ok(chunked.unbind());
    }

    let data = ArrayData::from_pyarrow_bound(&array)?;
    let arr: ArrayRef = make_array(data);
    let casted = cast_one(&arr, &target_dt, safe)?;
    casted.to_data().to_pyarrow(py)
}

/// `cast_polars_series(series, target, safe=False)` — round-trip the
/// polars Series through Arrow for the cast, then back.  Relies on
/// polars' native `to_arrow` / `from_arrow` bridges so we don't have to
/// duplicate the dispatch matrix.
#[pyfunction]
#[pyo3(signature = (series, target, safe = false))]
pub fn cast_polars_series(
    py: Python<'_>,
    series: Bound<'_, PyAny>,
    target: Bound<'_, PyAny>,
    safe: bool,
) -> PyResult<Py<PyAny>> {
    let pl = py.import("polars")?;
    let arrow_chunked = series.call_method0("to_arrow")?;
    // polars' to_arrow returns either a pa.Array or pa.ChunkedArray.
    // cast_arrow_array handles both.
    let casted = cast_arrow_array(py, arrow_chunked, target, safe)?;
    // `pl.from_arrow` accepts the Array/ChunkedArray; apply the series
    // name via a follow-up rename (no `name` kwarg on from_arrow).
    let series_out = pl
        .getattr("from_arrow")?
        .call1((casted,))?;
    let name = series.getattr("name")?;
    let renamed = series_out.call_method1("rename", (name,))?;
    Ok(renamed.unbind())
}

/// Resolve a target through the same matrix cast_arrow_array uses, and
/// return the pyarrow DataType — handy for tests.
#[pyfunction]
pub fn resolve_arrow_target(py: Python<'_>, target: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dt = resolve_target(py, &target)?;
    engine::arrow_to_py(py, &dt)
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(cast_arrow_array, module)?)?;
    module.add_function(wrap_pyfunction!(cast_polars_series, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_arrow_target, module)?)?;
    Ok(())
}
