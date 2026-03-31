use pyo3::prelude::*;

#[pyfunction]
fn utf8_len(values: Vec<Option<String>>) -> Vec<Option<usize>> {
    values
        .into_iter()
        .map(|value| value.map(|text| text.chars().count()))
        .collect()
}

#[pymodule]
fn yggdrasil_rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(utf8_len, module)?)?;
    Ok(())
}
