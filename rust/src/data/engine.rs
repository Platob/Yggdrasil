/// Engine bridges — Arrow and Polars.
///
/// Each of our `PyDataType` subclasses exposes `to_arrow()` and
/// `to_polars()` methods; this module holds the shared translation
/// logic so the per-type `#[pymethods]` blocks stay thin.
///
/// # Arrow
/// We construct `arrow_schema::DataType` in Rust, then use the
/// `arrow::pyarrow::ToPyArrow` trait to cross the FFI boundary into a
/// `pyarrow.DataType` Python object via the C Data Interface.
///
/// # Polars
/// We construct `polars::prelude::DataType` in Rust (internal
/// representation for future Series-level casts), but the Python
/// `polars.*` classes are instantiated via pyo3 getattr calls — the
/// polars Python wrapper is not on crates.io, so there is no Rust-side
/// `to_python()`.
use std::sync::Arc;

use arrow::pyarrow::ToPyArrow;
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, TimeUnit};
use polars::prelude::{DataType as PolarsDataType, Field as PolarsField, TimeUnit as PolarsTimeUnit};
use pyo3::prelude::*;

use crate::data::field::PyDataField;
use crate::data::types::nested::{PyArrayType, PyMapType, PyStructType};
use crate::data::types::object::PyObjectType;
use crate::data::types::primitive::{
    PyBinaryType, PyBooleanType, PyDecimalType, PyFloatingPointType, PyIntegerType, PyNullType,
    PyStringType,
};
use crate::data::types::temporal::{PyDateType, PyDurationType, PyTimestampType, PyTimeType};

// ---------------------------------------------------------------------------
// unit parsing
// ---------------------------------------------------------------------------

fn arrow_time_unit(unit: &str) -> PyResult<TimeUnit> {
    match unit {
        "s" => Ok(TimeUnit::Second),
        "ms" => Ok(TimeUnit::Millisecond),
        "us" => Ok(TimeUnit::Microsecond),
        "ns" => Ok(TimeUnit::Nanosecond),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cannot map unit {other:?} to an Arrow TimeUnit"
        ))),
    }
}

fn polars_time_unit(unit: &str) -> PyResult<PolarsTimeUnit> {
    match unit {
        "ms" => Ok(PolarsTimeUnit::Milliseconds),
        "us" => Ok(PolarsTimeUnit::Microseconds),
        "ns" => Ok(PolarsTimeUnit::Nanoseconds),
        // Polars has no second-precision datetime type — round up.
        "s" => Ok(PolarsTimeUnit::Milliseconds),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cannot map unit {other:?} to a Polars TimeUnit"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Null
// ---------------------------------------------------------------------------

pub fn null_to_arrow(_t: &PyNullType) -> ArrowDataType {
    ArrowDataType::Null
}

pub fn null_to_polars(_t: &PyNullType) -> PolarsDataType {
    PolarsDataType::Null
}

// ---------------------------------------------------------------------------
// Object (variant catch-all) — Arrow has no variant; stand-in is
// `large_binary`, matching the Python side.
// ---------------------------------------------------------------------------

pub fn object_to_arrow(_t: &PyObjectType) -> ArrowDataType {
    ArrowDataType::LargeBinary
}

pub fn object_to_polars(_t: &PyObjectType) -> PolarsDataType {
    // Polars-rs `Object` requires the `object` feature which we don't
    // pull in.  The Python side stands in with binary anyway; match it.
    PolarsDataType::Binary
}

// ---------------------------------------------------------------------------
// Boolean
// ---------------------------------------------------------------------------

pub fn boolean_to_arrow(_t: &PyBooleanType) -> ArrowDataType {
    ArrowDataType::Boolean
}

pub fn boolean_to_polars(_t: &PyBooleanType) -> PolarsDataType {
    PolarsDataType::Boolean
}

// ---------------------------------------------------------------------------
// Integer
// ---------------------------------------------------------------------------

pub fn integer_to_arrow(t: &PyIntegerType) -> ArrowDataType {
    let bytes = t.byte_size.unwrap_or(8);
    match (bytes, t.signed) {
        (1, true) => ArrowDataType::Int8,
        (2, true) => ArrowDataType::Int16,
        (4, true) => ArrowDataType::Int32,
        (8, true) => ArrowDataType::Int64,
        (1, false) => ArrowDataType::UInt8,
        (2, false) => ArrowDataType::UInt16,
        (4, false) => ArrowDataType::UInt32,
        (8, false) => ArrowDataType::UInt64,
        // Unknown width → default to 64-bit of the requested sign.
        (_, true) => ArrowDataType::Int64,
        (_, false) => ArrowDataType::UInt64,
    }
}

pub fn integer_to_polars(t: &PyIntegerType) -> PolarsDataType {
    let bytes = t.byte_size.unwrap_or(8);
    match (bytes, t.signed) {
        (1, true) => PolarsDataType::Int8,
        (2, true) => PolarsDataType::Int16,
        (4, true) => PolarsDataType::Int32,
        (8, true) => PolarsDataType::Int64,
        (1, false) => PolarsDataType::UInt8,
        (2, false) => PolarsDataType::UInt16,
        (4, false) => PolarsDataType::UInt32,
        (8, false) => PolarsDataType::UInt64,
        (_, true) => PolarsDataType::Int64,
        (_, false) => PolarsDataType::UInt64,
    }
}

// ---------------------------------------------------------------------------
// FloatingPoint
// ---------------------------------------------------------------------------

pub fn float_to_arrow(t: &PyFloatingPointType) -> ArrowDataType {
    match t.byte_size.unwrap_or(8) {
        2 => ArrowDataType::Float16,
        4 => ArrowDataType::Float32,
        _ => ArrowDataType::Float64,
    }
}

pub fn float_to_polars(t: &PyFloatingPointType) -> PolarsDataType {
    match t.byte_size.unwrap_or(8) {
        // Polars has no f16 — widen to f32.
        2 | 4 => PolarsDataType::Float32,
        _ => PolarsDataType::Float64,
    }
}

// ---------------------------------------------------------------------------
// Decimal
// ---------------------------------------------------------------------------

pub fn decimal_to_arrow(t: &PyDecimalType) -> PyResult<ArrowDataType> {
    let precision = t.precision.unwrap_or(38) as u8;
    let scale = t.scale.unwrap_or(0) as i8;
    // Arrow decimal128 for precision <= 38, decimal256 above.
    if precision <= 38 {
        Ok(ArrowDataType::Decimal128(precision, scale))
    } else {
        Ok(ArrowDataType::Decimal256(precision, scale))
    }
}

pub fn decimal_to_polars(t: &PyDecimalType) -> PolarsDataType {
    let precision = t.precision.map(|p| p as usize);
    let scale = t.scale.unwrap_or(0) as usize;
    PolarsDataType::Decimal(precision, Some(scale))
}

// ---------------------------------------------------------------------------
// String / Binary
// ---------------------------------------------------------------------------

pub fn string_to_arrow(t: &PyStringType) -> ArrowDataType {
    if t.view {
        ArrowDataType::Utf8View
    } else if t.large {
        ArrowDataType::LargeUtf8
    } else {
        ArrowDataType::Utf8
    }
}

pub fn string_to_polars(_t: &PyStringType) -> PolarsDataType {
    PolarsDataType::String
}

pub fn binary_to_arrow(t: &PyBinaryType) -> ArrowDataType {
    if let Some(n) = t.byte_size {
        return ArrowDataType::FixedSizeBinary(n as i32);
    }
    if t.view {
        ArrowDataType::BinaryView
    } else if t.large {
        ArrowDataType::LargeBinary
    } else {
        ArrowDataType::Binary
    }
}

pub fn binary_to_polars(_t: &PyBinaryType) -> PolarsDataType {
    PolarsDataType::Binary
}

// ---------------------------------------------------------------------------
// Date / Time / Timestamp / Duration
// ---------------------------------------------------------------------------

pub fn date_to_arrow(t: &PyDateType) -> ArrowDataType {
    match t.unit.as_str() {
        "d" => ArrowDataType::Date32,
        _ => ArrowDataType::Date64,
    }
}

pub fn date_to_polars(_t: &PyDateType) -> PolarsDataType {
    PolarsDataType::Date
}

pub fn time_to_arrow(t: &PyTimeType) -> PyResult<ArrowDataType> {
    let unit = arrow_time_unit(&t.unit)?;
    Ok(match unit {
        TimeUnit::Second | TimeUnit::Millisecond => ArrowDataType::Time32(unit),
        TimeUnit::Microsecond | TimeUnit::Nanosecond => ArrowDataType::Time64(unit),
    })
}

pub fn time_to_polars(t: &PyTimeType) -> PyResult<PolarsDataType> {
    let _unit = polars_time_unit(&t.unit)?;
    Ok(PolarsDataType::Time)
}

pub fn timestamp_to_arrow(t: &PyTimestampType) -> PyResult<ArrowDataType> {
    let unit = arrow_time_unit(&t.unit)?;
    let tz = t.tz.as_deref().map(Arc::from);
    Ok(ArrowDataType::Timestamp(unit, tz))
}

pub fn timestamp_to_polars(t: &PyTimestampType) -> PyResult<PolarsDataType> {
    let unit = polars_time_unit(&t.unit)?;
    Ok(PolarsDataType::Datetime(unit, t.tz.as_ref().map(|s| s.clone().into())))
}

pub fn duration_to_arrow(t: &PyDurationType) -> PyResult<ArrowDataType> {
    let unit = arrow_time_unit(&t.unit)?;
    Ok(ArrowDataType::Duration(unit))
}

pub fn duration_to_polars(t: &PyDurationType) -> PyResult<PolarsDataType> {
    let unit = polars_time_unit(&t.unit)?;
    Ok(PolarsDataType::Duration(unit))
}

// ---------------------------------------------------------------------------
// Nested — Array, Map, Struct
// ---------------------------------------------------------------------------

pub fn field_to_arrow(py: Python<'_>, field: &PyDataField) -> PyResult<ArrowField> {
    let dtype = field.dtype.bind(py);
    let arrow_any = dtype.call_method0("to_arrow")?;
    // Unwrap PyArrow DataType -> Rust arrow DataType via FFI.
    let arrow_dt = arrow::pyarrow::FromPyArrow::from_pyarrow_bound(&arrow_any)?;
    Ok(ArrowField::new(&field.name, arrow_dt, field.nullable))
}

pub fn field_to_polars(py: Python<'_>, field: &PyDataField) -> PyResult<PolarsField> {
    let dtype = field.dtype.bind(py);
    let pl_dt = polars_from_type(py, dtype)?;
    Ok(PolarsField::new(field.name.as_str().into(), pl_dt))
}

pub fn array_to_arrow(py: Python<'_>, t: &PyArrayType) -> PyResult<ArrowDataType> {
    let item = t.item_field.bind(py).borrow();
    let inner = field_to_arrow(py, &item)?;
    if let Some(n) = t.list_size {
        Ok(ArrowDataType::FixedSizeList(Arc::new(inner), n as i32))
    } else if t.view {
        // Arrow has ListView/LargeListView in 55+
        if t.large {
            Ok(ArrowDataType::LargeListView(Arc::new(inner)))
        } else {
            Ok(ArrowDataType::ListView(Arc::new(inner)))
        }
    } else if t.large {
        Ok(ArrowDataType::LargeList(Arc::new(inner)))
    } else {
        Ok(ArrowDataType::List(Arc::new(inner)))
    }
}

pub fn array_to_polars(py: Python<'_>, t: &PyArrayType) -> PyResult<PolarsDataType> {
    let item = t.item_field.bind(py).borrow();
    let inner = polars_from_type(py, item.dtype.bind(py))?;
    if let Some(n) = t.list_size {
        Ok(PolarsDataType::Array(Box::new(inner), n as usize))
    } else {
        Ok(PolarsDataType::List(Box::new(inner)))
    }
}

pub fn map_to_arrow(py: Python<'_>, t: &PyMapType) -> PyResult<ArrowDataType> {
    let key = t.key_field.bind(py).borrow();
    let value = t.value_field.bind(py).borrow();
    let key_arrow = field_to_arrow(py, &key)?.with_nullable(false);
    let value_arrow = field_to_arrow(py, &value)?;
    let struct_field = ArrowField::new(
        "entries",
        ArrowDataType::Struct(vec![key_arrow, value_arrow].into()),
        false,
    );
    Ok(ArrowDataType::Map(Arc::new(struct_field), t.keys_sorted))
}

pub fn map_to_polars(py: Python<'_>, t: &PyMapType) -> PyResult<PolarsDataType> {
    // Polars has no native Map type — model it as a List<Struct<key, value>>.
    let key = t.key_field.bind(py).borrow();
    let value = t.value_field.bind(py).borrow();
    let key_pl = polars_from_type(py, key.dtype.bind(py))?;
    let value_pl = polars_from_type(py, value.dtype.bind(py))?;
    let inner = PolarsDataType::Struct(vec![
        PolarsField::new("key".into(), key_pl),
        PolarsField::new("value".into(), value_pl),
    ]);
    Ok(PolarsDataType::List(Box::new(inner)))
}

pub fn struct_to_arrow(py: Python<'_>, t: &PyStructType) -> PyResult<ArrowDataType> {
    let mut fields = Vec::with_capacity(t.fields.len());
    for f in &t.fields {
        let borrowed = f.bind(py).borrow();
        fields.push(field_to_arrow(py, &borrowed)?);
    }
    Ok(ArrowDataType::Struct(fields.into()))
}

pub fn struct_to_polars(py: Python<'_>, t: &PyStructType) -> PyResult<PolarsDataType> {
    let mut fields = Vec::with_capacity(t.fields.len());
    for f in &t.fields {
        let borrowed = f.bind(py).borrow();
        fields.push(field_to_polars(py, &borrowed)?);
    }
    Ok(PolarsDataType::Struct(fields))
}

// ---------------------------------------------------------------------------
// Python-side entry points
// ---------------------------------------------------------------------------

/// Convert an arrow `DataType` into a PyArrow Python object.
pub fn arrow_to_py(py: Python<'_>, dt: &ArrowDataType) -> PyResult<Py<PyAny>> {
    dt.to_pyarrow(py)
}

/// Resolve *any* of our Rust PyDataType subclasses into its `polars`
/// DataType counterpart.  Used by nested/map/struct conversions that
/// need to recurse into child fields.
pub fn polars_from_type(
    py: Python<'_>,
    dtype: &pyo3::Bound<'_, pyo3::PyAny>,
) -> PyResult<PolarsDataType> {
    if let Ok(t) = dtype.downcast::<PyNullType>() {
        return Ok(null_to_polars(&t.borrow()));
    }
    if let Ok(t) = dtype.downcast::<PyObjectType>() {
        return Ok(object_to_polars(&t.borrow()));
    }
    if let Ok(t) = dtype.downcast::<PyBooleanType>() {
        return Ok(boolean_to_polars(&t.borrow()));
    }
    if let Ok(t) = dtype.downcast::<PyIntegerType>() {
        return Ok(integer_to_polars(&t.borrow()));
    }
    if let Ok(t) = dtype.downcast::<PyFloatingPointType>() {
        return Ok(float_to_polars(&t.borrow()));
    }
    if let Ok(t) = dtype.downcast::<PyDecimalType>() {
        return Ok(decimal_to_polars(&t.borrow()));
    }
    if let Ok(t) = dtype.downcast::<PyStringType>() {
        return Ok(string_to_polars(&t.borrow()));
    }
    if let Ok(t) = dtype.downcast::<PyBinaryType>() {
        return Ok(binary_to_polars(&t.borrow()));
    }
    if let Ok(t) = dtype.downcast::<PyDateType>() {
        return Ok(date_to_polars(&t.borrow()));
    }
    if let Ok(t) = dtype.downcast::<PyTimeType>() {
        return time_to_polars(&t.borrow());
    }
    if let Ok(t) = dtype.downcast::<PyTimestampType>() {
        return timestamp_to_polars(&t.borrow());
    }
    if let Ok(t) = dtype.downcast::<PyDurationType>() {
        return duration_to_polars(&t.borrow());
    }
    if let Ok(t) = dtype.downcast::<PyArrayType>() {
        return array_to_polars(py, &t.borrow());
    }
    if let Ok(t) = dtype.downcast::<PyMapType>() {
        return map_to_polars(py, &t.borrow());
    }
    if let Ok(t) = dtype.downcast::<PyStructType>() {
        return struct_to_polars(py, &t.borrow());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Unsupported DataType for polars conversion: {}",
        dtype
            .get_type()
            .name()
            .map(|s| s.to_string())
            .unwrap_or_default()
    )))
}

/// Reverse bridge: take a PyArrow `DataType` and produce the matching
/// `PyDataType` subclass as a `Py<PyAny>`.  Covers the primitives and
/// temporal shapes.  Nested reconstruction recurses through child
/// `Field` definitions, producing `PyArrayType` / `PyMapType` /
/// `PyStructType` instances.
pub fn from_pyarrow(py: Python<'_>, dtype: &Bound<'_, pyo3::PyAny>) -> PyResult<Py<pyo3::PyAny>> {
    let arrow_dt = arrow::pyarrow::FromPyArrow::from_pyarrow_bound(dtype)?;
    from_arrow_datatype(py, &arrow_dt)
}

fn from_arrow_datatype(py: Python<'_>, dt: &ArrowDataType) -> PyResult<Py<pyo3::PyAny>> {
    use crate::data::field::PyDataField;
    use crate::data::types::nested::{PyArrayType, PyMapType, PyStructType};
    use crate::data::types::object::PyObjectType;
    use crate::data::types::primitive::{
        PyBinaryType, PyBooleanType, PyDecimalType, PyFloatingPointType, PyIntegerType,
        PyNullType, PyStringType,
    };
    use crate::data::types::temporal::{PyDateType, PyDurationType, PyTimeType, PyTimestampType};
    use crate::data::types::base::PyDataType;

    fn build<T: pyo3::PyClass<BaseType = PyDataType>>(
        py: Python<'_>,
        concrete: T,
    ) -> PyResult<Py<pyo3::PyAny>> {
        let init = pyo3::PyClassInitializer::from(PyDataType {}).add_subclass(concrete);
        Ok(Py::new(py, init)?.into_any())
    }

    Ok(match dt {
        ArrowDataType::Null => build(py, PyNullType {})?,
        ArrowDataType::Boolean => build(py, PyBooleanType { byte_size: None })?,
        ArrowDataType::Int8 => build(py, PyIntegerType { byte_size: Some(1), signed: true })?,
        ArrowDataType::Int16 => build(py, PyIntegerType { byte_size: Some(2), signed: true })?,
        ArrowDataType::Int32 => build(py, PyIntegerType { byte_size: Some(4), signed: true })?,
        ArrowDataType::Int64 => build(py, PyIntegerType { byte_size: Some(8), signed: true })?,
        ArrowDataType::UInt8 => build(py, PyIntegerType { byte_size: Some(1), signed: false })?,
        ArrowDataType::UInt16 => build(py, PyIntegerType { byte_size: Some(2), signed: false })?,
        ArrowDataType::UInt32 => build(py, PyIntegerType { byte_size: Some(4), signed: false })?,
        ArrowDataType::UInt64 => build(py, PyIntegerType { byte_size: Some(8), signed: false })?,
        ArrowDataType::Float16 => build(py, PyFloatingPointType { byte_size: Some(2) })?,
        ArrowDataType::Float32 => build(py, PyFloatingPointType { byte_size: Some(4) })?,
        ArrowDataType::Float64 => build(py, PyFloatingPointType { byte_size: Some(8) })?,
        ArrowDataType::Utf8 => build(py, PyStringType { large: false, view: false, byte_size: None })?,
        ArrowDataType::LargeUtf8 => build(py, PyStringType { large: true, view: false, byte_size: None })?,
        ArrowDataType::Utf8View => build(py, PyStringType { large: false, view: true, byte_size: None })?,
        ArrowDataType::Binary => build(py, PyBinaryType { large: false, view: false, byte_size: None })?,
        ArrowDataType::LargeBinary => {
            // Distinguish ObjectType (large_binary stand-in) from plain
            // LargeBinary: we can't — assume plain LargeBinary.  Users
            // who want ObjectType construct it explicitly.
            build(py, PyBinaryType { large: true, view: false, byte_size: None })?
        }
        ArrowDataType::BinaryView => build(py, PyBinaryType { large: false, view: true, byte_size: None })?,
        ArrowDataType::FixedSizeBinary(n) => build(py, PyBinaryType {
            large: false,
            view: false,
            byte_size: Some(*n as u32),
        })?,
        ArrowDataType::Decimal128(p, s) => build(py, PyDecimalType {
            precision: Some(*p as u32),
            scale: Some(*s as i32),
            byte_size: None,
        })?,
        ArrowDataType::Decimal256(p, s) => build(py, PyDecimalType {
            precision: Some(*p as u32),
            scale: Some(*s as i32),
            byte_size: None,
        })?,
        ArrowDataType::Date32 => build(py, PyDateType {
            unit: "d".to_string(),
            tz: None,
            byte_size: Some(4),
        })?,
        ArrowDataType::Date64 => build(py, PyDateType {
            unit: "ms".to_string(),
            tz: None,
            byte_size: Some(8),
        })?,
        ArrowDataType::Time32(unit) | ArrowDataType::Time64(unit) => build(py, PyTimeType {
            unit: time_unit_str(unit).to_string(),
            tz: None,
            byte_size: None,
        })?,
        ArrowDataType::Timestamp(unit, tz) => build(py, PyTimestampType {
            unit: time_unit_str(unit).to_string(),
            tz: tz.as_ref().map(|t| t.to_string()),
            byte_size: None,
        })?,
        ArrowDataType::Duration(unit) => build(py, PyDurationType {
            unit: time_unit_str(unit).to_string(),
            byte_size: None,
        })?,
        ArrowDataType::List(field) | ArrowDataType::LargeList(field)
        | ArrowDataType::ListView(field) | ArrowDataType::LargeListView(field) => {
            let child = arrow_child_field_to_pyfield(py, field)?;
            let large = matches!(dt, ArrowDataType::LargeList(_) | ArrowDataType::LargeListView(_));
            let view = matches!(dt, ArrowDataType::ListView(_) | ArrowDataType::LargeListView(_));
            build(py, PyArrayType {
                item_field: child,
                list_size: None,
                large,
                view,
            })?
        }
        ArrowDataType::FixedSizeList(field, size) => {
            let child = arrow_child_field_to_pyfield(py, field)?;
            build(py, PyArrayType {
                item_field: child,
                list_size: Some(*size as u32),
                large: false,
                view: false,
            })?
        }
        ArrowDataType::Map(field, keys_sorted) => {
            let struct_dt = field.data_type();
            let ArrowDataType::Struct(inner_fields) = struct_dt else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Malformed Arrow Map: entries field is not a struct",
                ));
            };
            if inner_fields.len() != 2 {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Malformed Arrow Map: expected struct<key, value>",
                ));
            }
            let key = arrow_child_field_to_pyfield(py, &inner_fields[0])?;
            let value = arrow_child_field_to_pyfield(py, &inner_fields[1])?;
            build(py, PyMapType {
                key_field: key,
                value_field: value,
                keys_sorted: *keys_sorted,
            })?
        }
        ArrowDataType::Struct(fields) => {
            let mut children: Vec<Py<PyDataField>> = Vec::with_capacity(fields.len());
            for f in fields {
                children.push(arrow_child_field_to_pyfield(py, f)?);
            }
            build(py, PyStructType { fields: children })?
        }
        other => {
            let _ = PyObjectType {};
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported Arrow DataType: {other:?}"
            )));
        }
    })
}

fn arrow_child_field_to_pyfield(
    py: Python<'_>,
    field: &ArrowField,
) -> PyResult<Py<crate::data::field::PyDataField>> {
    let dtype_py = from_arrow_datatype(py, field.data_type())?;
    let pf = crate::data::field::PyDataField::construct(
        py,
        field.name().clone(),
        dtype_py.bind(py).clone(),
        field.is_nullable(),
        None,
        None,
        None,
    )?;
    Py::new(py, pf)
}

fn time_unit_str(unit: &TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    }
}

/// Reverse bridge: accept a `polars.DataType` Python object and produce
/// the matching `PyDataType` subclass.  We route through an empty
/// `pl.Series(dtype=...).to_arrow()` to borrow polars' own mapping,
/// then delegate to `from_pyarrow`.
pub fn from_polars(py: Python<'_>, dtype: &Bound<'_, pyo3::PyAny>) -> PyResult<Py<pyo3::PyAny>> {
    let pl = py.import("polars")?;
    let series_fn = pl.getattr("Series")?;
    let empty_list = pyo3::types::PyList::empty(py);
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("name", "")?;
    kwargs.set_item("values", empty_list)?;
    kwargs.set_item("dtype", dtype.clone())?;
    let series = series_fn.call((), Some(&kwargs))?;
    let arrow_series = series.call_method0("to_arrow")?;
    let arrow_type = arrow_series.getattr("type")?;
    from_pyarrow(py, &arrow_type)
}

/// Convert our internal `polars::DataType` representation into the
/// Python `polars.*` class instance by doing a `getattr` lookup on the
/// `polars` module.  This is the only place pyo3 reaches across to the
/// Python polars package — keeps the call site centralized.
pub fn polars_type_to_py(py: Python<'_>, dtype: &PolarsDataType) -> PyResult<Py<PyAny>> {
    let pl = py.import("polars")?;
    let out = match dtype {
        PolarsDataType::Null => pl.getattr("Null")?.call0()?,
        PolarsDataType::Boolean => pl.getattr("Boolean")?.call0()?,
        PolarsDataType::Int8 => pl.getattr("Int8")?.call0()?,
        PolarsDataType::Int16 => pl.getattr("Int16")?.call0()?,
        PolarsDataType::Int32 => pl.getattr("Int32")?.call0()?,
        PolarsDataType::Int64 => pl.getattr("Int64")?.call0()?,
        PolarsDataType::UInt8 => pl.getattr("UInt8")?.call0()?,
        PolarsDataType::UInt16 => pl.getattr("UInt16")?.call0()?,
        PolarsDataType::UInt32 => pl.getattr("UInt32")?.call0()?,
        PolarsDataType::UInt64 => pl.getattr("UInt64")?.call0()?,
        PolarsDataType::Float32 => pl.getattr("Float32")?.call0()?,
        PolarsDataType::Float64 => pl.getattr("Float64")?.call0()?,
        PolarsDataType::String => pl.getattr("String")?.call0()?,
        PolarsDataType::Binary => pl.getattr("Binary")?.call0()?,
        PolarsDataType::Date => pl.getattr("Date")?.call0()?,
        PolarsDataType::Time => pl.getattr("Time")?.call0()?,
        PolarsDataType::Datetime(unit, tz) => {
            let unit_str = match unit {
                PolarsTimeUnit::Milliseconds => "ms",
                PolarsTimeUnit::Microseconds => "us",
                PolarsTimeUnit::Nanoseconds => "ns",
            };
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("time_unit", unit_str)?;
            if let Some(tz) = tz {
                kwargs.set_item("time_zone", tz.as_str())?;
            }
            pl.getattr("Datetime")?.call((), Some(&kwargs))?
        }
        PolarsDataType::Duration(unit) => {
            let unit_str = match unit {
                PolarsTimeUnit::Milliseconds => "ms",
                PolarsTimeUnit::Microseconds => "us",
                PolarsTimeUnit::Nanoseconds => "ns",
            };
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("time_unit", unit_str)?;
            pl.getattr("Duration")?.call((), Some(&kwargs))?
        }
        PolarsDataType::Decimal(precision, scale) => {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("precision", *precision)?;
            kwargs.set_item("scale", *scale)?;
            pl.getattr("Decimal")?.call((), Some(&kwargs))?
        }
        PolarsDataType::List(inner) => {
            let inner_py = polars_type_to_py(py, inner)?;
            pl.getattr("List")?.call1((inner_py,))?
        }
        PolarsDataType::Array(inner, size) => {
            let inner_py = polars_type_to_py(py, inner)?;
            pl.getattr("Array")?.call1((inner_py, *size))?
        }
        PolarsDataType::Struct(fields) => {
            let dict = pyo3::types::PyDict::new(py);
            for f in fields {
                dict.set_item(f.name().as_str(), polars_type_to_py(py, f.dtype())?)?;
            }
            pl.getattr("Struct")?.call1((dict,))?
        }
        other => {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported polars DataType for Python bridge: {other:?}"
            )));
        }
    };
    Ok(out.unbind())
}
