/// `DataTypeId` — stable integer identity for each `DataType` kind.
///
/// Mirrors `yggdrasil.data.types.id.DataTypeId` on the Python side (an
/// `IntEnum`).  The integer values are part of the user-facing contract
/// (used in `to_dict()` payloads, merge ordering, and dispatch) and must
/// match the Python enum exactly.
use pyo3::prelude::*;

/// The raw integer discriminants.  Kept in a plain Rust enum so call
/// sites inside the crate can pattern-match without going through Python.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum DataTypeId {
    Object = 0,
    Null = 1,
    Bool = 2,
    Integer = 3,
    Float = 4,
    Decimal = 5,
    Date = 6,
    Time = 7,
    Timestamp = 8,
    Duration = 9,
    Binary = 10,
    String = 11,

    Array = 32,
    Map = 33,
    Struct = 34,
    Union = 35,

    Dictionary = 65,
    Json = 66,
    Enum = 67,
}

impl DataTypeId {
    /// Uppercase name matching the Python `IntEnum` member name.
    pub const fn name(self) -> &'static str {
        match self {
            DataTypeId::Object => "OBJECT",
            DataTypeId::Null => "NULL",
            DataTypeId::Bool => "BOOL",
            DataTypeId::Integer => "INTEGER",
            DataTypeId::Float => "FLOAT",
            DataTypeId::Decimal => "DECIMAL",
            DataTypeId::Date => "DATE",
            DataTypeId::Time => "TIME",
            DataTypeId::Timestamp => "TIMESTAMP",
            DataTypeId::Duration => "DURATION",
            DataTypeId::Binary => "BINARY",
            DataTypeId::String => "STRING",
            DataTypeId::Array => "ARRAY",
            DataTypeId::Map => "MAP",
            DataTypeId::Struct => "STRUCT",
            DataTypeId::Union => "UNION",
            DataTypeId::Dictionary => "DICTIONARY",
            DataTypeId::Json => "JSON",
            DataTypeId::Enum => "ENUM",
        }
    }

    pub const fn value(self) -> u8 {
        self as u8
    }

    pub fn from_value(value: i64) -> Option<Self> {
        match value {
            0 => Some(DataTypeId::Object),
            1 => Some(DataTypeId::Null),
            2 => Some(DataTypeId::Bool),
            3 => Some(DataTypeId::Integer),
            4 => Some(DataTypeId::Float),
            5 => Some(DataTypeId::Decimal),
            6 => Some(DataTypeId::Date),
            7 => Some(DataTypeId::Time),
            8 => Some(DataTypeId::Timestamp),
            9 => Some(DataTypeId::Duration),
            10 => Some(DataTypeId::Binary),
            11 => Some(DataTypeId::String),
            32 => Some(DataTypeId::Array),
            33 => Some(DataTypeId::Map),
            34 => Some(DataTypeId::Struct),
            35 => Some(DataTypeId::Union),
            65 => Some(DataTypeId::Dictionary),
            66 => Some(DataTypeId::Json),
            67 => Some(DataTypeId::Enum),
            _ => None,
        }
    }

    pub const ALL: [DataTypeId; 19] = [
        DataTypeId::Object,
        DataTypeId::Null,
        DataTypeId::Bool,
        DataTypeId::Integer,
        DataTypeId::Float,
        DataTypeId::Decimal,
        DataTypeId::Date,
        DataTypeId::Time,
        DataTypeId::Timestamp,
        DataTypeId::Duration,
        DataTypeId::Binary,
        DataTypeId::String,
        DataTypeId::Array,
        DataTypeId::Map,
        DataTypeId::Struct,
        DataTypeId::Union,
        DataTypeId::Dictionary,
        DataTypeId::Json,
        DataTypeId::Enum,
    ];
}

/// Python-facing wrapper.  Exposes the enum as `DataTypeId` on the
/// `yggdrasil.rust.data.types` module and supports `int(x)` coercion,
/// equality with plain ints, and a `.name` / `.value` pair that matches
/// `enum.IntEnum`.
///
/// We do not subclass `IntEnum` from Rust (PyO3 cannot do that cleanly);
/// instead, the canonical Python `DataTypeId` stays the `IntEnum` in
/// `yggdrasil.data.types.id` and this class mirrors its shape for users
/// who only have `yggrs` available.
#[pyclass(module = "yggdrasil.rust.data.types", frozen, eq, ord, hash, name = "DataTypeId")]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PyDataTypeId {
    pub inner: DataTypeId,
}

#[pymethods]
impl PyDataTypeId {
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    fn value(&self) -> u8 {
        self.inner.value()
    }

    fn __int__(&self) -> u8 {
        self.inner.value()
    }

    fn __index__(&self) -> u8 {
        self.inner.value()
    }

    fn __repr__(&self) -> String {
        format!("DataTypeId.{}", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Coerce an int, str, or existing member — mirrors `IntEnum(value)`.
    #[staticmethod]
    fn coerce(value: Bound<'_, PyAny>) -> PyResult<Self> {
        from_pyany(&value)
    }
}

impl From<DataTypeId> for PyDataTypeId {
    fn from(inner: DataTypeId) -> Self {
        Self { inner }
    }
}

/// Coerce an arbitrary Python object into a `PyDataTypeId`.  Accepts the
/// existing Python `IntEnum`, a plain int matching a known discriminant,
/// or the uppercase member name as a string.
pub fn from_pyany(value: &Bound<'_, PyAny>) -> PyResult<PyDataTypeId> {
    if let Ok(existing) = value.extract::<PyDataTypeId>() {
        return Ok(existing);
    }
    if let Ok(int_value) = value.extract::<i64>() {
        if let Some(id) = DataTypeId::from_value(int_value) {
            return Ok(id.into());
        }
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown DataTypeId value: {int_value}"
        )));
    }
    if let Ok(name) = value.extract::<String>() {
        let upper = name.to_uppercase();
        for id in DataTypeId::ALL {
            if id.name() == upper {
                return Ok(id.into());
            }
        }
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown DataTypeId name: {name:?}"
        )));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "DataTypeId expects an int, str, or existing DataTypeId",
    ))
}

/// Attach `DataTypeId` and each member as a module attribute so callers
/// can write `from yggdrasil.rust.data.types import DataTypeId` *and*
/// `DataTypeId.INTEGER`-style access works without a metaclass.
pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDataTypeId>()?;

    let cls = module.getattr("DataTypeId")?;
    for id in DataTypeId::ALL {
        let member = PyDataTypeId::from(id);
        cls.setattr(id.name(), Py::new(module.py(), member)?)?;
    }
    Ok(())
}
