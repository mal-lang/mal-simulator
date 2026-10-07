//! PyO3 bindings from `malsim-core` to the Python package in `python/malsim`,
//! built as `malsim._native`.
//!
//! `node_count` below is a Phase A1 smoke-test function only (see
//! `PORTING_NOTES.md` §5/A1, §10): it proves that a `maltoolbox.AttackGraph`
//! Python object handed to this, a *separately compiled* extension module,
//! can still reach mal-toolbox's shared `Rc<RefCell<AttackGraph>>` - not a
//! direct pyclass downcast (that doesn't work across independently-built
//! `cdylib`s, see §10), but via the `PyCapsule` mal-toolbox's
//! `PyAttackGraph::__inner_capsule__` hands out for exactly this purpose.
//! Not meant as a real public API.

use std::cell::RefCell;
use std::rc::Rc;

use maltoolbox_attackgraph::AttackGraph;
use pyo3::types::PyCapsule;
use pyo3::prelude::*;

/// Must match mal-toolbox's `py-bindings/maltoolbox-attackgraph-py/src/
/// graph.rs`'s `INNER_CAPSULE_NAME` exactly - this string is the only
/// runtime check standing in for compile-time type safety across the
/// module boundary.
const INNER_CAPSULE_NAME: &std::ffi::CStr = c"maltoolbox._native.AttackGraph.inner";

/// Extracts the shared `Rc<RefCell<AttackGraph>>` handle from a Python
/// `maltoolbox.AttackGraph` object via its `__inner_capsule__()` method.
fn extract_shared_graph(graph: &Bound<'_, PyAny>) -> PyResult<Rc<RefCell<AttackGraph>>> {
    let capsule_obj = graph.call_method0("__inner_capsule__")?;
    let capsule = capsule_obj.cast::<PyCapsule>()?;
    let ptr = capsule.pointer_checked(Some(INNER_CAPSULE_NAME))?;
    let typed_ptr = ptr.as_ptr() as *const RefCell<AttackGraph>;
    // SAFETY: the capsule itself still owns the one strong reference
    // `__inner_capsule__` parked via `Rc::into_raw` - its destructor will
    // reclaim and drop *that* reference when the capsule is GC'd. We must
    // not call `Rc::from_raw` directly on this pointer, since that would
    // reclaim the capsule's own reference out from under it (a double
    // free once the capsule's destructor also runs - confirmed by this
    // crashing with `malloc(): unaligned tcache chunk detected` when
    // tried). Instead, bump the strong count first to mint a brand new,
    // independently-owned reference, then reconstruct an `Rc` from that.
    unsafe {
        Rc::increment_strong_count(typed_ptr);
        Ok(Rc::from_raw(typed_ptr))
    }
}

/// Reads the node count of a `maltoolbox.AttackGraph` through the shared
/// `Rc<RefCell<AttackGraph>>` handle extracted above, without going through
/// any further Python-level attribute access.
#[pyfunction]
fn node_count(graph: &Bound<'_, PyAny>) -> PyResult<usize> {
    let shared = extract_shared_graph(graph)?;
    let count = shared.borrow().nodes.len();
    Ok(count)
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(node_count, m)?)?;
    Ok(())
}
