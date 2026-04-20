/// Shared byte-string constants matching `yggdrasil.data.constants`.
///
/// These are part of the user contract: tag keys are persisted to Arrow
/// metadata and Databricks properties, so the byte values must match the
/// Python side exactly.
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub const TAG_PREFIX: &[u8] = b"t:";
pub const DBX_META_PREFIX: &[u8] = b"databricks:";
pub const DEFAULT_VALUE_KEY: &[u8] = b"default";
pub const DEFAULT_FIELD_NAME: &str = "";

pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("TAG_PREFIX", PyBytes::new(py, TAG_PREFIX))?;
    module.add("DBX_META_PREFIX", PyBytes::new(py, DBX_META_PREFIX))?;
    module.add("DEFAULT_VALUE_KEY", PyBytes::new(py, DEFAULT_VALUE_KEY))?;
    module.add("DEFAULT_FIELD_NAME", DEFAULT_FIELD_NAME)?;
    Ok(())
}
