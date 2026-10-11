//! `malsim._native.DynaSimulator` - the pyclass backing
//! `DynaMalSimulator.reset()`/`.step()`: a thin wrapper around
//! `malsim_core::dyna_simulator::DynaSimulator`, reusing `simulator.rs`'s
//! dict conversions (`PORTING_NOTES.md` §6 Phase B4/B5, §11).

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use malsim_core::dyna_simulator::DynaSimulator as CoreDynaSimulator;

use crate::simulator::{
    build_reset_output, build_step_output, extract_actions, not_reset_err, parse_agents,
    parse_settings, sim_err_to_py,
};
use crate::{extract_shared_graph, extract_shared_model};

/// Python handle on a `malsim_core::dyna_simulator::DynaSimulator`
/// (dynamic instance model/attack graph). `model` must be the
/// `maltoolbox.Model` `graph` was built from; its state at construction is
/// the snapshot every reset restores (mirroring `DynaMalSimulator.__init__`).
#[pyclass(name = "DynaSimulator", module = "malsim._native", unsendable)]
pub struct DynaSimulator {
    inner: CoreDynaSimulator,
}

const CLASS_NAME: &str = "DynaSimulator";

#[pymethods]
impl DynaSimulator {
    #[new]
    fn new(graph: &Bound<'_, PyAny>, model: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(DynaSimulator {
            inner: CoreDynaSimulator::new(
                extract_shared_graph(graph)?,
                extract_shared_model(model)?,
            ),
        })
    }

    /// Restores the shared `Model`/`AttackGraph` to the construction-time
    /// snapshot - see `CoreDynaSimulator::restore_model`. Python's
    /// `dyna_reset` calls this before flattening agent settings, so nodes
    /// removed by the previous episode's model effects are back (with
    /// their regenerated ids) when the rules are resolved.
    fn restore_model_native(&mut self) -> PyResult<()> {
        self.inner
            .restore_model()
            .map_err(|e| sim_err_to_py(e, CLASS_NAME))
    }

    /// Resets the simulator - see `CoreDynaSimulator::reset`. `agents`
    /// must already be resolved against the restored graph
    /// (`restore_model_native`). Same output shape as
    /// `Simulator.reset_native`.
    fn reset_native(
        &mut self,
        py: Python<'_>,
        settings: &Bound<'_, PyDict>,
        agents: &Bound<'_, PyDict>,
        seed: u64,
    ) -> PyResult<Py<PyAny>> {
        let graph_rc = self.inner.graph().clone();
        let settings = parse_settings(settings)?;
        let agents = parse_agents(&graph_rc.borrow(), agents)?;
        let state = self
            .inner
            .reset(&settings, agents, seed)
            .map_err(|e| sim_err_to_py(e, CLASS_NAME))?;
        let graph = graph_rc.borrow();
        build_reset_output(py, &graph, state)
    }

    /// Steps the simulation - see `CoreDynaSimulator::step`. Same output
    /// shape as `Simulator.step_native`, plus `sim_state`'s
    /// `step_modification_record` (and the full graph-state maps on a step
    /// whose record is non-empty - see `build_step_output`).
    fn step_native(&mut self, py: Python<'_>, actions: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let action_nodes = {
            let state = self
                .inner
                .state()
                .ok_or_else(|| not_reset_err(CLASS_NAME))?;
            extract_actions(&self.inner.graph().borrow(), state, actions)?
        };
        let outcome = self
            .inner
            .step(&action_nodes)
            .map_err(|e| sim_err_to_py(e, CLASS_NAME))?;
        let state = self
            .inner
            .state()
            .ok_or_else(|| not_reset_err(CLASS_NAME))?;
        build_step_output(
            py,
            &self.inner.graph().borrow(),
            state,
            &outcome.step,
            Some(&outcome.step_modification_record),
        )
    }
}
