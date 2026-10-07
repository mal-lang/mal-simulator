//! PyO3 bindings from `malsim-core` to the Python package in `python/malsim`,
//! built as `malsim._native`.

use pyo3::prelude::*;

#[pymodule]
fn _native(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
