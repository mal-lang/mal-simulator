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
//! `model_asset_count` is the Phase B3 equivalent for `maltoolbox.Model` /
//! `PyModel::__inner_capsule__` (see `PORTING_NOTES.md` §6/B3). Neither is
//! meant as a real public API.

use std::cell::RefCell;
use std::rc::Rc;

use maltoolbox_attackgraph::AttackGraph;
use maltoolbox_model::Model;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

mod dyna_simulator;
mod simulator;

/// Must match mal-toolbox's `py-bindings/maltoolbox-attackgraph-py/src/
/// graph.rs`'s `INNER_CAPSULE_NAME` exactly - this string is the only
/// runtime check standing in for compile-time type safety across the
/// module boundary.
const INNER_CAPSULE_NAME: &std::ffi::CStr = c"maltoolbox._native.AttackGraph.inner";

/// Must match mal-toolbox's `py-bindings/maltoolbox-model-py/src/model.rs`'s
/// `INNER_CAPSULE_NAME` exactly - same role as `INNER_CAPSULE_NAME` above,
/// for `maltoolbox.Model` instead of `maltoolbox.AttackGraph` (Phase B3).
const MODEL_INNER_CAPSULE_NAME: &std::ffi::CStr = c"maltoolbox._native.Model.inner";

/// Extracts the shared `Rc<RefCell<AttackGraph>>` handle from a Python
/// `maltoolbox.AttackGraph` object via its `__inner_capsule__()` method.
/// `pub(crate)` so `simulator.rs`'s `Simulator` pyclass (Phase A8) can
/// reuse it too, rather than re-implementing the same capsule handoff.
pub(crate) fn extract_shared_graph(graph: &Bound<'_, PyAny>) -> PyResult<Rc<RefCell<AttackGraph>>> {
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

/// Extracts the shared `Rc<RefCell<Model>>` handle from a Python
/// `maltoolbox.Model` object via its `__inner_capsule__()` method - Model
/// counterpart of `extract_shared_graph` above (Phase B3, PORTING_NOTES.md
/// §6/B3). `pub(crate)` for the same reason: future dyna native entry
/// points (B4) will need this handle too, not just this phase's smoke test.
pub(crate) fn extract_shared_model(model: &Bound<'_, PyAny>) -> PyResult<Rc<RefCell<Model>>> {
    let capsule_obj = model.call_method0("__inner_capsule__")?;
    let capsule = capsule_obj.cast::<PyCapsule>()?;
    let ptr = capsule.pointer_checked(Some(MODEL_INNER_CAPSULE_NAME))?;
    let typed_ptr = ptr.as_ptr() as *const RefCell<Model>;
    // SAFETY: same reasoning as `extract_shared_graph` above - the capsule
    // still owns the one strong reference `__inner_capsule__` parked, so we
    // bump the strong count and mint an independently-owned `Rc` rather
    // than reclaiming the capsule's own reference directly.
    unsafe {
        Rc::increment_strong_count(typed_ptr);
        Ok(Rc::from_raw(typed_ptr))
    }
}

/// Reads the asset count of a `maltoolbox.Model` through the shared
/// `Rc<RefCell<Model>>` handle extracted above, without going through any
/// further Python-level attribute access. Phase B3 smoke-test function,
/// same tier as `node_count` above - not a real public API.
#[pyfunction]
fn model_asset_count(model: &Bound<'_, PyAny>) -> PyResult<usize> {
    let shared = extract_shared_model(model)?;
    let count = shared.borrow().assets.len();
    Ok(count)
}

/// Test-support-only utility (not a real public API, same tier as
/// `set_detector_rates` below): adds an asset directly through the shared
/// `Rc<RefCell<Model>>` handle, bypassing `maltoolbox`'s Python-level
/// `Model.add_asset()` entirely - proves the *other* direction of Phase
/// B3's double-visibility requirement (a mutation made through the native
/// handle must be visible back on the Python side), which
/// `model_asset_count` alone (read-only) can't exercise.
#[pyfunction]
fn model_add_asset_native(model: &Bound<'_, PyAny>, asset_type: &str) -> PyResult<i64> {
    let shared = extract_shared_model(model)?;
    let mut model = shared.borrow_mut();
    model
        .add_asset(asset_type, None, None, None, None, true)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Test-support-only utility (not a real public API, same tier as
/// `node_count` above): directly mutates a detector's `tprate`/`fprate`
/// through the shared `Rc<RefCell<AttackGraph>>`, bypassing
/// `maltoolbox`'s Python-level `node.detectors` binding entirely.
///
/// Found necessary during Phase A9 (PORTING_NOTES.md §5/§10): `maltoolbox`'s
/// `AttackGraphNode.detectors` Python property is a *Python-side* cached
/// `dict` seeded once from the core's generation-time detector data -
/// `node.detectors['x'] = Detector(...)` mutates only that cache, never the
/// real `AttackGraphNode.detectors` field malsim-core's Rust hot path
/// reads, so tests that used this pattern to force deterministic
/// tprate/fprate for a run were silently exercising the *old* rates once
/// `collect_logs`/`collect_false_positives` moved to Rust. This function
/// mutates the real field directly instead.
#[pyfunction]
fn set_detector_rates(
    graph: &Bound<'_, PyAny>,
    node_id: i64,
    label: &str,
    tprate: Option<f64>,
    fprate: Option<f64>,
) -> PyResult<()> {
    let shared = extract_shared_graph(graph)?;
    let mut graph = shared.borrow_mut();
    let key = *graph.id_to_node.get(&node_id).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "node id {node_id} is not part of this graph"
        ))
    })?;
    let detector = graph
        .nodes
        .get_mut(key)
        .expect("id_to_node only maps to live nodes")
        .detectors
        .get_mut(label)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "node {node_id} has no detector labeled \"{label}\""
            ))
        })?;
    detector.tprate = tprate;
    detector.fprate = fprate;
    Ok(())
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(node_count, m)?)?;
    m.add_function(wrap_pyfunction!(model_asset_count, m)?)?;
    m.add_function(wrap_pyfunction!(model_add_asset_native, m)?)?;
    m.add_function(wrap_pyfunction!(set_detector_rates, m)?)?;
    m.add_class::<simulator::Simulator>()?;
    m.add_class::<dyna_simulator::DynaSimulator>()?;
    Ok(())
}
