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
use crate::state::{
    current_sim_state_to_py, no_state_err, recording_to_py, snapshot_to_py, terminated_to_py,
    Episode, EpisodeSlot,
};
use crate::{extract_shared_graph, extract_shared_model};
use malsim_core::simulator::SimState;

/// Python handle on a `malsim_core::dyna_simulator::DynaSimulator`
/// (dynamic instance model/attack graph). `model` must be the
/// `maltoolbox.Model` `graph` was built from; its state at construction is
/// the snapshot every reset restores (mirroring `DynaMalSimulator.__init__`).
#[pyclass(name = "DynaSimulator", module = "malsim._native", unsendable)]
pub struct DynaSimulator {
    inner: CoreDynaSimulator,
    /// The current episode's archive slot - see `state::EpisodeSlot`.
    episode: EpisodeSlot,
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
            episode: EpisodeSlot::new(),
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
        self.episode.end_episode(self.inner.state());
        let state = self
            .inner
            .reset(&settings, agents, seed)
            .map_err(|e| sim_err_to_py(e, CLASS_NAME))?;
        let graph = graph_rc.borrow();
        build_reset_output(py, &graph, state)
    }

    /// The full current state (agents and `sim_state`), translated on
    /// every call - see `state::snapshot_to_py`.
    fn agent_states_native(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = self.live_state()?;
        let graph = self.inner.graph().borrow();
        let iteration = state.history().iteration();
        snapshot_to_py(
            py,
            &graph,
            state.snapshot_at(iteration),
            state.history(),
            iteration,
            None,
        )
    }

    /// The current `sim_state` part only - see `state::current_sim_state_to_py`.
    fn sim_state_native(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = self.live_state()?;
        current_sim_state_to_py(py, &self.inner.graph().borrow(), state)
    }

    /// The episode's recording so far - see `state::recording_to_py`.
    fn recording_native(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = self.live_state()?;
        recording_to_py(py, &self.inner.graph().borrow(), state)
    }

    /// Per agent name, whether that agent is terminated.
    fn terminated_native(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        terminated_to_py(py, self.live_state()?)
    }

    /// A handle on the current episode, to read its earlier iterations
    /// (also after the next reset) - see `state::Episode`.
    fn episode_native(slf: &Bound<'_, Self>) -> PyResult<Episode> {
        let this = slf.borrow();
        this.live_state()?;
        Ok(this
            .episode
            .handle(this.inner.graph().clone(), slf.clone().into_any().unbind()))
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

impl DynaSimulator {
    /// The live core state - an error before the first successful reset.
    pub(crate) fn live_state(&self) -> PyResult<&SimState> {
        self.inner.state().ok_or_else(|| no_state_err(CLASS_NAME))
    }
}
