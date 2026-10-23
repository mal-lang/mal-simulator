//! Translation of the simulator state into plain Python data, read on
//! demand by `MalSimulator`/`DynaMalSimulator`'s `agent_states`/
//! `sim_state`/`recording` attributes (`PORTING_NOTES.md` §14): the Python
//! simulators keep no state of their own, so every access lands here.
//!
//! Also `Episode`, the handle a lazily-read `previous_state` resolves
//! through: it reads the live simulator while its episode is current, and
//! an archived copy of the episode once `reset` has started a new one.

use std::cell::RefCell;
use std::rc::Rc;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use malsim_core::history::{
    AttackerSnapshot, DefenderSnapshot, EpisodeHistory, EpisodeRecord, StateSnapshot,
};
use malsim_core::model_effects::ModEffectOp;
use malsim_core::simulator::SimState;
use malsim_core::types::NodeSet;

use crate::dyna_simulator::DynaSimulator;
use crate::simulator::{
    id_value_map, insert_graph_state_fields, log_entry_to_py, mod_effect_op_to_py, stable_ids,
    Simulator,
};

/// Like `stable_ids`, but keeps the given order (and drops dead ids).
fn ordered_stable_ids(
    graph: &AttackGraph,
    ids: impl IntoIterator<Item = AttackGraphNodeId>,
) -> Vec<i64> {
    ids.into_iter()
        .filter_map(|id| graph.nodes.get(id).map(|n| n.id))
        .collect()
}

fn performed_nodes_order_to_py<'py>(
    py: Python<'py>,
    graph: &AttackGraph,
    order: &std::collections::BTreeMap<u64, NodeSet>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (iteration, nodes) in order {
        d.set_item(iteration, stable_ids(graph, nodes.iter().copied()))?;
    }
    Ok(d)
}

fn modification_record_to_py(py: Python<'_>, record: &[ModEffectOp]) -> PyResult<Vec<Py<PyAny>>> {
    record
        .iter()
        .map(|op| mod_effect_op_to_py(py, op))
        .collect()
}

/// The `sim_state` part of the wire format: the full graph state,
/// `enabled_defenses` and the modification record so far (always empty on
/// the static path).
fn sim_state_to_py<'py>(
    py: Python<'py>,
    graph: &AttackGraph,
    graph_state: &malsim_core::graph_state::GraphState,
    enabled_defenses: &NodeSet,
    modification_record: &[ModEffectOp],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "enabled_defenses",
        stable_ids(graph, enabled_defenses.iter().copied()),
    )?;
    insert_graph_state_fields(&d, graph, graph_state)?;
    d.set_item(
        "modification_record",
        modification_record_to_py(py, modification_record)?,
    )?;
    Ok(d)
}

fn attacker_to_py<'py>(
    py: Python<'py>,
    graph: &AttackGraph,
    a: &AttackerSnapshot,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("type", "attacker")?;
    d.set_item(
        "entry_points",
        stable_ids(graph, a.entry_points.iter().copied()),
    )?;
    d.set_item(
        "performed_nodes",
        stable_ids(graph, a.performed_nodes.iter().copied()),
    )?;
    d.set_item(
        "attempted_nodes",
        stable_ids(graph, a.attempted_nodes.iter().copied()),
    )?;
    d.set_item(
        "action_surface",
        stable_ids(graph, a.action_surface.iter().copied()),
    )?;
    d.set_item("num_attempts", id_value_map(graph, &a.num_attempts))?;
    d.set_item("ttc_values", id_value_map(graph, &a.ttc_values))?;
    d.set_item(
        "impossible_steps",
        stable_ids(graph, a.impossible_steps.iter().copied()),
    )?;
    d.set_item(
        "performed_nodes_order",
        performed_nodes_order_to_py(py, graph, &a.performed_nodes_order)?,
    )?;
    d.set_item("iteration", a.iteration)?;
    Ok(d)
}

fn defender_to_py<'py>(
    py: Python<'py>,
    graph: &AttackGraph,
    de: &DefenderSnapshot,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("type", "defender")?;
    d.set_item(
        "performed_nodes",
        stable_ids(graph, de.performed_nodes.iter().copied()),
    )?;
    d.set_item(
        "compromised_nodes",
        stable_ids(graph, de.compromised_nodes.iter().copied()),
    )?;
    d.set_item(
        "observed_nodes",
        stable_ids(graph, de.observed_nodes.iter().copied()),
    )?;
    d.set_item(
        "action_surface",
        stable_ids(graph, de.action_surface.iter().copied()),
    )?;
    let logs: Vec<Py<PyAny>> = de
        .logs
        .iter()
        .map(|log| log_entry_to_py(py, graph, log))
        .collect::<PyResult<_>>()?;
    d.set_item("logs", logs)?;
    d.set_item(
        "performed_nodes_order",
        performed_nodes_order_to_py(py, graph, &de.performed_nodes_order)?,
    )?;
    d.set_item("iteration", de.iteration)?;
    Ok(d)
}

/// The full state at `iteration` as plain data: `{"iteration", "sim_state",
/// "agents": {name: {...}}}`, all nodes as stable ids. With `agent`, only
/// that agent is translated.
pub(crate) fn snapshot_to_py(
    py: Python<'_>,
    graph: &AttackGraph,
    snapshot: Option<StateSnapshot>,
    history: &EpisodeHistory,
    iteration: u64,
    agent: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let snapshot: StateSnapshot = snapshot.ok_or_else(|| {
        PyValueError::new_err(format!(
            "iteration {iteration} is after the episode's current iteration {}",
            history.iteration()
        ))
    })?;
    let agents = PyDict::new(py);
    for (name, a) in &snapshot.attackers {
        if agent.is_none_or(|n| n == name) {
            agents.set_item(name, attacker_to_py(py, graph, a)?)?;
        }
    }
    for (name, de) in &snapshot.defenders {
        if agent.is_none_or(|n| n == name) {
            agents.set_item(name, defender_to_py(py, graph, de)?)?;
        }
    }
    let out = PyDict::new(py);
    out.set_item("iteration", snapshot.iteration)?;
    out.set_item(
        "sim_state",
        sim_state_to_py(
            py,
            graph,
            &snapshot.graph_state,
            &snapshot.enabled_defenses,
            &snapshot.modification_record,
        )?,
    )?;
    out.set_item("agents", agents)?;
    Ok(out.into())
}

/// The current `sim_state` only (no agents) - see `sim_state_to_py`.
pub(crate) fn current_sim_state_to_py(
    py: Python<'_>,
    graph: &AttackGraph,
    state: &SimState,
) -> PyResult<Py<PyAny>> {
    Ok(sim_state_to_py(
        py,
        graph,
        &state.graph_state,
        &state.enabled_defenses,
        state.history().modification_record(),
    )?
    .into())
}

/// The whole episode's recording: `{iteration: {agent: [stable id, ...]}}`
/// for every step so far (1-based iterations, as `MalSimulator.recording`
/// keys them), each agent's nodes requested-actions-first.
pub(crate) fn recording_to_py(
    py: Python<'_>,
    graph: &AttackGraph,
    state: &SimState,
) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    let history = state.history();
    for iteration in 1..=history.iteration() {
        let Some(entry) = history.recording(iteration) else {
            continue;
        };
        let per_agent = PyDict::new(py);
        for (name, nodes) in entry {
            per_agent.set_item(name, ordered_stable_ids(graph, nodes.iter().copied()))?;
        }
        out.set_item(iteration, per_agent)?;
    }
    Ok(out.into())
}

/// The error a state read raises before the first successful reset.
pub(crate) fn no_state_err(class_name: &str) -> PyErr {
    PyValueError::new_err(format!("{class_name} state read before reset_native()"))
}

/// Per agent, whether it is terminated (`attacker_is_terminated`/
/// `defender_is_terminated`).
pub(crate) fn terminated_to_py(py: Python<'_>, state: &SimState) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    for name in state.attackers.keys() {
        out.set_item(name, state.attacker_is_terminated(name) == Some(true))?;
    }
    let defenders_terminated = state.defender_is_terminated();
    for name in state.defenders.keys() {
        out.set_item(name, defenders_terminated)?;
    }
    Ok(out.into())
}

/// The archive slot of the simulator's current episode, owned by the
/// `Simulator`/`DynaSimulator` pyclass and shared with every `Episode`
/// handle given out for that episode.
pub(crate) struct EpisodeSlot {
    archive: Rc<RefCell<Option<EpisodeRecord>>>,
}

impl EpisodeSlot {
    pub(crate) fn new() -> Self {
        EpisodeSlot {
            archive: Rc::new(RefCell::new(None)),
        }
    }

    /// Called right before a reset replaces `current`: if any `Episode`
    /// handle of the ending episode is still alive, archive a copy of it
    /// for them; then start a fresh slot for the next episode.
    pub(crate) fn end_episode(&mut self, current: Option<&SimState>) {
        if Rc::strong_count(&self.archive) > 1 {
            if let Some(state) = current {
                *self.archive.borrow_mut() = Some(state.episode_record());
            }
        }
        self.archive = Rc::new(RefCell::new(None));
    }

    pub(crate) fn handle(&self, graph: Rc<RefCell<AttackGraph>>, sim: Py<PyAny>) -> Episode {
        Episode {
            archive: self.archive.clone(),
            graph,
            sim,
        }
    }
}

/// A handle on one episode of a simulator, to read its state at an
/// iteration of that episode (`snapshot_native`). Python's lazy
/// `previous_state` keeps one.
#[pyclass(name = "Episode", module = "malsim._native", unsendable)]
pub struct Episode {
    archive: Rc<RefCell<Option<EpisodeRecord>>>,
    graph: Rc<RefCell<AttackGraph>>,
    sim: Py<PyAny>,
}

#[pymethods]
impl Episode {
    /// The state at `iteration` of this episode (0 is right after reset),
    /// same shape as `Simulator.agent_states_native`. With `agent`, only
    /// that agent is included.
    #[pyo3(signature = (iteration, agent=None))]
    fn snapshot_native(
        &self,
        py: Python<'_>,
        iteration: u64,
        agent: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow();
        if let Some(record) = self.archive.borrow().as_ref() {
            let snapshot = record.snapshot_at(iteration);
            return snapshot_to_py(py, &graph, snapshot, record.history(), iteration, agent);
        }
        let sim = self.sim.bind(py);
        if let Ok(sim) = sim.cast::<Simulator>() {
            let sim = sim.borrow();
            let state = sim.live_state()?;
            let snapshot = state.snapshot_at(iteration);
            return snapshot_to_py(py, &graph, snapshot, state.history(), iteration, agent);
        }
        let sim = sim.cast::<DynaSimulator>()?.borrow();
        let state = sim.live_state()?;
        let snapshot = state.snapshot_at(iteration);
        snapshot_to_py(py, &graph, snapshot, state.history(), iteration, agent)
    }
}
