//! `malsim._native.Simulator` - the Phase A8 native simulator pyclass,
//! extended at Phase A9 to back `MalSimulator.reset()`/`.step()` (see
//! `PORTING_NOTES.md` §5 Phase A8/A9/§10).
//!
//! A thin wrapper around `malsim_core::simulator::Simulator`, which owns
//! the reset/step orchestration: this module parses the flat Python dicts
//! into `MalSimulatorSettings`/`FlatAgentSettings`, calls the core
//! `Simulator`, and builds the plain-`dict` output from its `StepOutcome`
//! and `SimState`.
//!
//! Scope still deliberately smaller than the full Python
//! `MalSimulatorSettings`/`AttackerSettings`/`DefenderSettings` surface:
//! - No "multiple entry point sets, sampled at reset" support
//!   (`AttackerSettings.entry_points` as a `tuple[Set, ...]`) - only a
//!   single flat set of entry points per attacker. A9 resolves that
//!   sampling in Python (reusing `attacker_state_factories.py::
//!   get_entry_points` unchanged) before calling `reset_native`, since the
//!   sampling itself has no RNG-reproducibility stakes worth porting - see
//!   PORTING_NOTES.md §10.
//! - No rewards (unchanged, pure Python, per §2.4 - not this module's job
//!   regardless of phase).
//!
//! Per-agent TTC distribution overrides (`AttackerSettings.ttc_dists`) *are*
//! supported as of A9 - see `parse_ttc_dist_from_py`/
//! `extract_optional_ttc_dist_overrides` below.
//! Every other hot-loop-relevant field (`ttc_mode`, both
//! `AttackSurfaceSettings` fields, both bernoulli toggles,
//! `compromise_entrypoints_at_start`, entry points/goals/actionable steps/
//! observable steps/false positive and negative rates) is present, using
//! the same already-flattened id-keyed shapes (§2.4) A5-A7 established.
//!
//! All node references crossing the FFI boundary (in either direction) are
//! the stable, user-facing `AttackGraphNode.id: i64` - *not* the internal
//! `AttackGraphNodeId` slotmap key malsim-core's hot-path functions use
//! internally. `AttackGraph::id_to_node`/`graph.nodes[id].id` are the two
//! directions of that translation (see `to_node_id`/`stable_ids` below).

use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use malsim_core::event_logger::LogEntry;
use malsim_core::graph_state::{GraphState, TtcMode};
use malsim_core::model_effects::{AssetOp, AssetRef, AssocOp, ModEffectOp};
use malsim_core::settings::{
    FlatAgentSettings, FlatAttackerSettings, FlatDefenderSettings, MalSimulatorSettings,
};
use malsim_core::simulator::{Simulator as CoreSimulator, SimulatorError, StepOutcome};
use malsim_core::ttc::{named_ttc_dist, DistFunction, Operation, TtcDist};

use crate::{extract_shared_graph, extract_shared_model};

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Maps a core `SimulatorError` to the Python exception this module has
/// always raised for that case: every error is a `ValueError` carrying the
/// wrapped error's message, except `NotReset`, which keeps this module's
/// own `step_native`-worded message.
fn sim_err_to_py(e: SimulatorError) -> PyErr {
    match e {
        SimulatorError::NotReset => not_reset_err(),
        other => to_py_err(other),
    }
}

fn not_reset_err() -> PyErr {
    PyValueError::new_err("Simulator.step_native() called before reset_native()")
}

fn not_attached_err() -> PyErr {
    PyValueError::new_err(
        "Simulator.dyna_step_native() called before dyna_reset_native() - no Model is attached",
    )
}

// --- Node id translation (stable i64 <-> AttackGraphNodeId) ---

fn to_node_id(graph: &AttackGraph, stable_id: i64) -> PyResult<AttackGraphNodeId> {
    graph.id_to_node.get(&stable_id).copied().ok_or_else(|| {
        PyValueError::new_err(format!(
            "node id {stable_id} is not part of this simulator's attack graph"
        ))
    })
}

/// Sorted so every id-set crossing the FFI boundary (`action_surface`,
/// `performed_nodes`, ...) has a deterministic order - `HashSet`'s default
/// hasher is seeded randomly per process, so an unsorted `Vec` collected
/// from one would give a *different* order on every run even for the
/// exact same seed and graph (found during Phase A9 - see
/// PORTING_NOTES.md §10: this silently broke run-to-run reproducibility,
/// not just parity with the old Python frozenset ordering).
/// An id in `ids` can be stale: the dyna path lets a step's own model
/// effect remove the very asset a node belongs to (including a node this
/// same step just compromised/enabled - `PORTING_NOTES.md` §0 B5), so by
/// the time output is serialized, `performed_nodes`/`action_surface`/etc.
/// can hold an id with no live node left to translate. Such an id is
/// silently dropped: the thing it refers to no longer exists in the
/// graph, so it has no stable id left to report - mirrors `collect_logs`'s
/// same-shaped fix in `malsim-core::event_logger`.
fn stable_ids(graph: &AttackGraph, ids: impl IntoIterator<Item = AttackGraphNodeId>) -> Vec<i64> {
    let mut ids: Vec<i64> = ids
        .into_iter()
        .filter_map(|id| graph.nodes.get(id).map(|n| n.id))
        .collect();
    ids.sort_unstable();
    ids
}

fn id_value_map<V: Copy>(
    graph: &AttackGraph,
    map: &HashMap<AttackGraphNodeId, V>,
) -> HashMap<i64, V> {
    map.iter()
        .filter_map(|(&id, &v)| graph.nodes.get(id).map(|n| (n.id, v)))
        .collect()
}

// --- Input extraction helpers ---

fn get_dict_value<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(match dict.get_item(key)? {
        Some(v) if !v.is_none() => Some(v),
        _ => None,
    })
}

fn require_dict_value<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    get_dict_value(dict, key)?.ok_or_else(|| {
        PyValueError::new_err(format!("agent config missing required \"{key}\" field"))
    })
}

fn extract_id_list(
    graph: &AttackGraph,
    obj: &Bound<'_, PyAny>,
) -> PyResult<Vec<AttackGraphNodeId>> {
    let ids: Vec<i64> = obj.extract()?;
    ids.into_iter().map(|id| to_node_id(graph, id)).collect()
}

fn extract_id_set(
    graph: &AttackGraph,
    obj: &Bound<'_, PyAny>,
) -> PyResult<HashSet<AttackGraphNodeId>> {
    Ok(extract_id_list(graph, obj)?.into_iter().collect())
}

fn extract_optional_id_set(
    graph: &AttackGraph,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<HashSet<AttackGraphNodeId>>> {
    get_dict_value(dict, key)?
        .map(|v| extract_id_set(graph, &v))
        .transpose()
}

fn extract_optional_rate_map(
    graph: &AttackGraph,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<HashMap<AttackGraphNodeId, f64>>> {
    let Some(v) = get_dict_value(dict, key)? else {
        return Ok(None);
    };
    let raw: HashMap<i64, f64> = v.extract()?;
    let mut out = HashMap::with_capacity(raw.len());
    for (id, rate) in raw {
        out.insert(to_node_id(graph, id)?, rate);
    }
    Ok(Some(out))
}

fn get_bool(dict: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    match get_dict_value(dict, key)? {
        Some(v) => v.extract(),
        None => Ok(default),
    }
}

/// Port of `TTCDist.from_dict`'s Python-dict parsing (§2.4: `ttc_dists`
/// overrides cross the FFI boundary as the same `to_dict()`/`from_dict()`
/// wire shape Python's `TTCDist` already uses - `{"name", "arguments"}`
/// for a leaf distribution, a named-distribution shortcut, or
/// `{"lhs", "rhs", "type"}` for a combined one) - built directly against
/// `ttc::TtcDist::new`/`with_combine`/`named_ttc_dist` rather than via
/// `serde_json::Value`, so this module doesn't need a direct `serde_json`
/// dependency (it's only transitively present via `malsim-core`).
fn parse_ttc_dist_from_py(dict: &Bound<'_, PyDict>) -> PyResult<TtcDist> {
    if let Some(name_obj) = get_dict_value(dict, "name")? {
        let name: String = name_obj.extract()?;
        if let Some(dist) = named_ttc_dist(&name) {
            return Ok(dist);
        }
        let function = DistFunction::from_str(&name).map_err(|_| {
            PyValueError::new_err(format!("unknown distribution function name \"{name}\""))
        })?;
        let args: Vec<f64> = require_dict_value(dict, "arguments")?.extract()?;
        return TtcDist::new(function, args).map_err(to_py_err);
    }

    let lhs_dict = require_dict_value(dict, "lhs")?.cast::<PyDict>()?.clone();
    let rhs_dict = require_dict_value(dict, "rhs")?.cast::<PyDict>()?.clone();
    let op_str: String = require_dict_value(dict, "type")?.extract()?;
    let op = Operation::from_str(&op_str).map_err(|_| {
        PyValueError::new_err(format!("unknown ttc combine operation \"{op_str}\""))
    })?;
    let lhs = parse_ttc_dist_from_py(&lhs_dict)?;
    let rhs = parse_ttc_dist_from_py(&rhs_dict)?;
    Ok(lhs.with_combine(rhs, op))
}

/// Parses the optional per-attacker `ttc_dists` override map
/// (`dict[node_id, ttc_dict]`, already flattened from
/// `AttackerSettings.ttc_dists: NodePropertyRule[TTCDist]` on the Python
/// side via `.per_node()` + `.to_dict()` - §2.4) into
/// `HashMap<AttackGraphNodeId, TtcDist>`.
fn extract_optional_ttc_dist_overrides(
    graph: &AttackGraph,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<HashMap<AttackGraphNodeId, TtcDist>>> {
    let Some(v) = get_dict_value(dict, key)? else {
        return Ok(None);
    };
    let raw = v.cast::<PyDict>()?;
    let mut out = HashMap::with_capacity(raw.len());
    for (k, val) in raw.iter() {
        let node_id: i64 = k.extract()?;
        let dist_dict = val.cast::<PyDict>()?.clone();
        out.insert(
            to_node_id(graph, node_id)?,
            parse_ttc_dist_from_py(&dist_dict)?,
        );
    }
    Ok(Some(out))
}

fn parse_ttc_mode(s: &str) -> PyResult<TtcMode> {
    match s {
        "DISABLED" => Ok(TtcMode::Disabled),
        "EFFORT_BASED_PER_STEP_SAMPLE" => Ok(TtcMode::EffortBasedPerStepSample),
        "PER_STEP_SAMPLE" => Ok(TtcMode::PerStepSample),
        "PRE_SAMPLE" => Ok(TtcMode::PreSample),
        "EXPECTED_VALUE" => Ok(TtcMode::ExpectedValue),
        other => Err(PyValueError::new_err(format!(
            "unknown ttc_mode \"{other}\""
        ))),
    }
}

/// Parses the flat settings dict (`native_settings.py::flatten_sim_settings`'s
/// output - no nested `attack_surface` key, `skip_compromised`/
/// `skip_unnecessary` at the top level) into a `MalSimulatorSettings`.
/// Fields the dict doesn't carry (`seed`, `uncompromise_untraversable_steps`)
/// keep their `Default`.
fn parse_settings(dict: &Bound<'_, PyDict>) -> PyResult<MalSimulatorSettings> {
    let ttc_mode_str = match get_dict_value(dict, "ttc_mode")? {
        Some(v) => v.extract::<String>()?,
        None => "DISABLED".to_string(),
    };
    let mut settings = MalSimulatorSettings {
        ttc_mode: parse_ttc_mode(&ttc_mode_str)?,
        ..MalSimulatorSettings::default()
    };
    settings.run_defense_step_bernoullis = get_bool(dict, "run_defense_step_bernoullis", true)?;
    settings.run_attack_step_bernoullis = get_bool(dict, "run_attack_step_bernoullis", true)?;
    settings.attack_surface.skip_compromised = get_bool(dict, "skip_compromised", true)?;
    settings.attack_surface.skip_unnecessary = get_bool(dict, "skip_unnecessary", false)?;
    settings.compromise_entrypoints_at_start =
        get_bool(dict, "compromise_entrypoints_at_start", true)?;
    Ok(settings)
}

/// Parses one attacker's flat dict (`native_settings.py::
/// flatten_attacker_settings`'s output) into `FlatAttackerSettings`.
fn parse_attacker(graph: &AttackGraph, cfg: &Bound<'_, PyDict>) -> PyResult<FlatAttackerSettings> {
    let entry_points = extract_id_set(graph, &require_dict_value(cfg, "entry_points")?)?;
    let goals = extract_optional_id_set(graph, cfg, "goals")?.unwrap_or_default();
    let actionable_steps = extract_optional_id_set(graph, cfg, "actionable_steps")?;
    let ttc_dists = extract_optional_ttc_dist_overrides(graph, cfg, "ttc_dists")?;
    Ok(FlatAttackerSettings {
        entry_points,
        goals,
        actionable_steps,
        ttc_dists,
    })
}

/// Parses one defender's flat dict (`native_settings.py::
/// flatten_defender_settings`'s output) into `FlatDefenderSettings`.
fn parse_defender(graph: &AttackGraph, cfg: &Bound<'_, PyDict>) -> PyResult<FlatDefenderSettings> {
    let actionable_steps = extract_optional_id_set(graph, cfg, "actionable_steps")?;
    let observable_steps = extract_optional_id_set(graph, cfg, "observable_steps")?;
    let false_positive_rates = extract_optional_rate_map(graph, cfg, "false_positive_rates")?;
    let false_negative_rates = extract_optional_rate_map(graph, cfg, "false_negative_rates")?;
    Ok(FlatDefenderSettings {
        actionable_steps,
        observable_steps,
        false_positive_rates,
        false_negative_rates,
    })
}

/// Parses the `agents` dict (`name -> flat per-agent dict`) into the
/// `(name, FlatAgentSettings)` list `CoreSimulator::reset` takes. Every
/// agent's `type` is checked first (in dict order), then attackers are
/// parsed, then defenders - the same order these errors surfaced in before
/// the orchestration moved to malsim-core.
fn parse_agents(
    graph: &AttackGraph,
    agents: &Bound<'_, PyDict>,
) -> PyResult<Vec<(String, FlatAgentSettings)>> {
    let mut attacker_cfgs = Vec::new();
    let mut defender_cfgs = Vec::new();
    for (name_obj, cfg_obj) in agents.iter() {
        let name: String = name_obj.extract()?;
        let cfg = cfg_obj.cast::<PyDict>()?.clone();
        let kind: String = require_dict_value(&cfg, "type")?.extract()?;
        match kind.as_str() {
            "attacker" => attacker_cfgs.push((name, cfg)),
            "defender" => defender_cfgs.push((name, cfg)),
            other => {
                return Err(PyValueError::new_err(format!(
                    "agent \"{name}\" has unknown type \"{other}\" (expected \"attacker\" or \"defender\")"
                )))
            }
        }
    }

    let mut out = Vec::with_capacity(attacker_cfgs.len() + defender_cfgs.len());
    for (name, cfg) in attacker_cfgs {
        let parsed = parse_attacker(graph, &cfg)?;
        out.push((name, FlatAgentSettings::Attacker(parsed)));
    }
    for (name, cfg) in defender_cfgs {
        let parsed = parse_defender(graph, &cfg)?;
        out.push((name, FlatAgentSettings::Defender(parsed)));
    }
    Ok(out)
}

#[pyclass(name = "Simulator", module = "malsim._native", unsendable)]
pub struct Simulator {
    /// A plain `CoreSimulator::new` until `dyna_reset_native` is first
    /// called, which replaces it with `CoreSimulator::new_dyna` attached to
    /// that call's `model`; later `dyna_reset_native` calls reuse it (a
    /// different `model` passed later is ignored) - §6 Phase B4.
    inner: CoreSimulator,
}

#[pymethods]
impl Simulator {
    #[new]
    fn new(graph: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Simulator {
            inner: CoreSimulator::new(extract_shared_graph(graph)?),
        })
    }

    /// Resets the simulator - see `CoreSimulator::reset`.
    fn reset_native(
        &mut self,
        py: Python<'_>,
        settings: &Bound<'_, PyDict>,
        agents: &Bound<'_, PyDict>,
        seed: u64,
    ) -> PyResult<Py<PyAny>> {
        self.do_reset(py, settings, agents, seed)
    }

    /// Restores the shared `Model`/`AttackGraph` to the pristine snapshot -
    /// see `CoreSimulator::restore_model`. Attaches `model` first if this
    /// is the first dyna call (same as `dyna_reset_native`). Python's
    /// `dyna_reset` calls this before flattening agent settings, so nodes
    /// removed by the previous episode's model effects are back (with
    /// their regenerated ids) when the rules are resolved.
    fn dyna_restore_model_native(&mut self, model: &Bound<'_, PyAny>) -> PyResult<()> {
        self.attach_model(model)?;
        self.inner.restore_model().map_err(sim_err_to_py)
    }

    /// Dyna-aware reset (Phase B4, A8/B3-equivalent entry point): attaches
    /// `model` the first time this is called (via `extract_shared_model` +
    /// `CoreSimulator::new_dyna`, which captures the pristine snapshot),
    /// then runs `CoreSimulator::reset`, which restores the shared
    /// `Model`/`AttackGraph` to that snapshot before the shared reset -
    /// mirroring `dyna_reset`'s `reset_model_effects` call followed by
    /// `compute_initial_graph_state`/`reset_agents`. `agents` must already be
    /// resolved against the restored graph (`dyna_restore_model_native`).
    fn dyna_reset_native(
        &mut self,
        py: Python<'_>,
        settings: &Bound<'_, PyDict>,
        agents: &Bound<'_, PyDict>,
        model: &Bound<'_, PyAny>,
        seed: u64,
    ) -> PyResult<Py<PyAny>> {
        self.attach_model(model)?;
        self.do_reset(py, settings, agents, seed)
    }

    /// Steps the simulation - see `CoreSimulator::step`.
    fn step_native(&mut self, py: Python<'_>, actions: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let action_nodes = self.extract_actions(actions)?;
        let outcome = self.inner.step(&action_nodes).map_err(sim_err_to_py)?;
        self.build_step_output(py, &outcome)
    }

    /// Dyna-aware step (Phase B4, A9-equivalent entry point) - see
    /// `CoreSimulator::step`. Requires `dyna_reset_native` to have been
    /// called at least once (for a `Model` to be attached).
    fn dyna_step_native(
        &mut self,
        py: Python<'_>,
        actions: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        if self.inner.model().is_none() {
            return Err(not_attached_err());
        }
        let action_nodes = self.extract_actions(actions)?;
        let outcome = self.inner.step(&action_nodes).map_err(sim_err_to_py)?;
        self.build_step_output(py, &outcome)
    }
}

impl Simulator {
    /// Shared body of `reset_native`/`dyna_reset_native` - see each
    /// pymethod's doc comment for what differs before this is called.
    /// Switches the inner simulator to a dyna one over `model` on the first
    /// dyna call; later calls keep the already-attached model (a different
    /// `model` is ignored, as before).
    fn attach_model(&mut self, model: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.inner.model().is_none() {
            let model_rc = extract_shared_model(model)?;
            self.inner = CoreSimulator::new_dyna(self.inner.graph().clone(), model_rc);
        }
        Ok(())
    }

    fn do_reset(
        &mut self,
        py: Python<'_>,
        settings: &Bound<'_, PyDict>,
        agents: &Bound<'_, PyDict>,
        seed: u64,
    ) -> PyResult<Py<PyAny>> {
        let settings = parse_settings(settings)?;
        let agents = {
            let graph = self.inner.graph().borrow();
            parse_agents(&graph, agents)?
        };
        self.inner
            .reset(&settings, agents, seed)
            .map_err(sim_err_to_py)?;
        self.build_reset_output(py)
    }

    /// Translates `step_native`'s `actions` dict (`name -> [stable id]`)
    /// into `CoreSimulator::step`'s input. Mirrors `_pre_step_check`'s
    /// `KeyError` on an `actions` key that names no registered agent
    /// (checked in key order, before any node id is parsed); an agent
    /// missing from `actions` (or mapped to `None`) takes no action.
    fn extract_actions(
        &self,
        actions: &Bound<'_, PyDict>,
    ) -> PyResult<HashMap<String, Vec<AttackGraphNodeId>>> {
        let state = self.inner.state().ok_or_else(not_reset_err)?;

        for key_obj in actions.keys() {
            let name: String = key_obj.extract()?;
            if !state.attackers.contains_key(&name) && !state.defenders.contains_key(&name) {
                return Err(sim_err_to_py(SimulatorError::UnknownAgent(name)));
            }
        }

        let mut action_nodes: HashMap<String, Vec<AttackGraphNodeId>> = HashMap::new();
        let graph = self.inner.graph().borrow();
        for name in state.attackers.keys().chain(state.defenders.keys()) {
            let nodes = match actions.get_item(name)? {
                Some(v) if !v.is_none() => extract_id_list(&graph, &v)?,
                _ => Vec::new(),
            };
            action_nodes.insert(name.clone(), nodes);
        }
        Ok(action_nodes)
    }
}

/// Writes `ttc_values`/`impossible_attack_steps`/`necessity_per_node`/
/// `pre_enabled_defenses` into `sim_state` from `graph_state` - the full
/// episode-accumulated `GraphState`, not a delta. Shared by
/// `build_reset_output` (always) and `build_step_output` (only on a dyna
/// step that actually ran a model effect - see that function's doc
/// comment for why these can't be a delta against the previous step).
fn insert_graph_state_fields(
    sim_state: &Bound<'_, PyDict>,
    graph: &AttackGraph,
    graph_state: &GraphState,
) -> PyResult<()> {
    sim_state.set_item("ttc_values", id_value_map(graph, &graph_state.ttc_values))?;
    sim_state.set_item(
        "impossible_attack_steps",
        stable_ids(graph, graph_state.impossible_attack_steps.iter().copied()),
    )?;
    sim_state.set_item(
        "necessity_per_node",
        id_value_map(graph, &graph_state.necessity_per_node),
    )?;
    sim_state.set_item(
        "pre_enabled_defenses",
        stable_ids(graph, graph_state.pre_enabled_defenses.iter().copied()),
    )?;
    Ok(())
}

impl Simulator {
    /// Builds `reset_native`'s plain-`dict` return value - full
    /// episode-initial state for every field, since there's no previous
    /// step to delta against at reset. See module docs for the exact
    /// shape. Deliberately no custom pyclasses in the return value (per
    /// A8's own plan text: "keep it boring and inspectable").
    fn build_reset_output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = self.inner.state().ok_or_else(not_reset_err)?;
        let graph = self.inner.graph().borrow();

        let sim_state = PyDict::new(py);
        sim_state.set_item(
            "enabled_defenses",
            stable_ids(&graph, state.enabled_defenses.iter().copied()),
        )?;
        insert_graph_state_fields(&sim_state, &graph, &state.graph_state)?;

        let defenders_terminated = state.defender_is_terminated();

        let agents = PyDict::new(py);
        for (name, a) in &state.attackers {
            let d = PyDict::new(py);
            d.set_item("type", "attacker")?;
            d.set_item(
                "performed_nodes",
                stable_ids(&graph, a.performed_nodes.iter().copied()),
            )?;
            d.set_item(
                "attempted_nodes",
                stable_ids(&graph, a.attempted_nodes.iter().copied()),
            )?;
            d.set_item(
                "action_surface",
                stable_ids(&graph, a.action_surface.iter().copied()),
            )?;
            d.set_item("num_attempts", id_value_map(&graph, &a.num_attempts))?;
            d.set_item("ttc_values", id_value_map(&graph, &a.ttc_values))?;
            d.set_item(
                "impossible_steps",
                stable_ids(&graph, a.impossible_steps.iter().copied()),
            )?;
            d.set_item("iteration", a.iteration)?;
            d.set_item("terminated", state.attacker_is_terminated(name))?;
            agents.set_item(name, d)?;
        }

        for (name, de) in &state.defenders {
            let d = PyDict::new(py);
            d.set_item("type", "defender")?;
            d.set_item(
                "performed_nodes",
                stable_ids(&graph, de.performed_nodes.iter().copied()),
            )?;
            d.set_item(
                "compromised_nodes",
                stable_ids(&graph, de.compromised_nodes.iter().copied()),
            )?;
            d.set_item(
                "observed_nodes",
                stable_ids(&graph, de.observed_nodes.iter().copied()),
            )?;
            d.set_item(
                "action_surface",
                stable_ids(&graph, de.action_surface.iter().copied()),
            )?;
            d.set_item("iteration", de.iteration)?;
            d.set_item("terminated", defenders_terminated)?;

            let logs: Vec<Py<PyAny>> = de
                .logs
                .iter()
                .map(|log| log_entry_to_py(py, &graph, log))
                .collect::<PyResult<_>>()?;
            d.set_item("logs", logs)?;

            agents.set_item(name, d)?;
        }

        let out = PyDict::new(py);
        out.set_item("sim_state", sim_state)?;
        out.set_item("agents", agents)?;
        Ok(out.into())
    }

    /// Builds `step_native`'s plain-`dict` return value - unlike
    /// `build_reset_output`, every monotonically-growing field is this
    /// step's *delta* only (`step_*` keys - see module docs' wire-format
    /// table), not the full episode-accumulated value: the accumulated
    /// values still live in the core `SimState` for Rust's own internal use
    /// (`get_attack_surface`, `*_is_terminated`, ...), they just no longer
    /// cross the FFI boundary redundantly on every call. Episode-static
    /// fields (`ttc_values`, `necessity_per_node`,
    /// `impossible_attack_steps`, `pre_enabled_defenses`) are dropped
    /// entirely for a plain `step_native` call (`step_modification_record`
    /// is always empty there) - Python caches them from `reset_native`'s
    /// output, same as before.
    ///
    /// **Not episode-static for `dyna_step_native`, though** (PORTING_NOTES.md
    /// §6 Phase B5's TTC-gap fix): `dyna_attacker_step`/`dyna_defender_step`
    /// grow `state.graph_state` in place via `fold_new_nodes_into_graph_state`
    /// whenever a model effect creates nodes mid-episode, and
    /// `necessity_per_node` is a full-graph recompute each time (not just new
    /// keys) - so a true per-field delta isn't provably correct. Whenever
    /// `step_modification_record` is non-empty (a model effect genuinely ran
    /// this step), this resends the full current maps via
    /// `insert_graph_state_fields`, same shape as `build_reset_output`;
    /// Python rebuilds its `GraphState` wholesale from them that step only.
    /// `action_surface`, `iteration` and `terminated` are unchanged (full
    /// current value every step), matching `build_reset_output`.
    fn build_step_output(&self, py: Python<'_>, outcome: &StepOutcome) -> PyResult<Py<PyAny>> {
        let state = self.inner.state().ok_or_else(not_reset_err)?;
        let graph = self.inner.graph().borrow();

        let step_enabled_defenses = &outcome.step_enabled_defenses;
        let step_modification_record = &outcome.step_modification_record;
        // Every node any attacker compromised this step - what each
        // defender's `compromised_nodes` grew by.
        let step_compromised_nodes: HashSet<AttackGraphNodeId> = outcome
            .attackers
            .values()
            .flat_map(|a| a.step_performed_nodes.iter().copied())
            .collect();

        let sim_state = PyDict::new(py);
        sim_state.set_item(
            "step_enabled_defenses",
            stable_ids(&graph, step_enabled_defenses.iter().copied()),
        )?;
        // Always present (empty for plain `step_native` - §6 Phase B4):
        // this step's model-effect modification record, as plain dicts
        // (`{"kind": "asset"/"assoc", "type": "ADDITIVE"/"SUBTRACTIVE",
        // ...}`) - Python resolves these back into `AssetOp`/`AssocOp`
        // objects and appends them onto `DynaMalSimulatorState.
        // modification_record` (delta-only, same wire-format discipline
        // A10 established for `logs`/etc. - see module docs).
        let modification_record: Vec<Py<PyAny>> = step_modification_record
            .iter()
            .map(|op| mod_effect_op_to_py(py, op))
            .collect::<PyResult<_>>()?;
        sim_state.set_item("step_modification_record", modification_record)?;
        if !step_modification_record.is_empty() {
            insert_graph_state_fields(&sim_state, &graph, &state.graph_state)?;
        }

        let defenders_terminated = state.defender_is_terminated();

        let agents = PyDict::new(py);
        for (name, a) in &state.attackers {
            let delta = &outcome.attackers[name];
            let (step_performed_nodes, step_attempted_nodes) =
                (&delta.step_performed_nodes, &delta.step_attempted_nodes);
            let d = PyDict::new(py);
            d.set_item("type", "attacker")?;
            d.set_item(
                "step_performed_nodes",
                stable_ids(&graph, step_performed_nodes.iter().copied()),
            )?;
            d.set_item(
                "step_attempted_nodes",
                stable_ids(&graph, step_attempted_nodes.iter().copied()),
            )?;
            d.set_item(
                "action_surface",
                stable_ids(&graph, a.action_surface.iter().copied()),
            )?;
            d.set_item("iteration", a.iteration)?;
            d.set_item("terminated", state.attacker_is_terminated(name))?;
            agents.set_item(name, d)?;
        }

        for (name, de) in &state.defenders {
            let delta = &outcome.defenders[name];
            let (step_observed_nodes, step_logs) = (&delta.step_observed_nodes, &delta.step_logs);
            let d = PyDict::new(py);
            d.set_item("type", "defender")?;
            d.set_item(
                "step_performed_nodes",
                stable_ids(&graph, step_enabled_defenses.iter().copied()),
            )?;
            d.set_item(
                "step_compromised_nodes",
                stable_ids(&graph, step_compromised_nodes.iter().copied()),
            )?;
            d.set_item(
                "step_observed_nodes",
                stable_ids(&graph, step_observed_nodes.iter().copied()),
            )?;
            d.set_item(
                "action_surface",
                stable_ids(&graph, de.action_surface.iter().copied()),
            )?;
            d.set_item("iteration", de.iteration)?;
            d.set_item("terminated", defenders_terminated)?;

            let logs: Vec<Py<PyAny>> = step_logs
                .iter()
                .map(|log| log_entry_to_py(py, &graph, log))
                .collect::<PyResult<_>>()?;
            d.set_item("step_logs", logs)?;

            agents.set_item(name, d)?;
        }

        let out = PyDict::new(py);
        out.set_item("sim_state", sim_state)?;
        out.set_item("agents", agents)?;
        Ok(out.into())
    }
}

/// Writes one `AssetRef`'s `id`/`asset_type`/`name` into `d` under
/// `prefix`-qualified keys (e.g. `prefix = "asset"` -> `asset_id`/
/// `asset_type`/`asset_name`) - shared by every `mod_effect_op_to_py`
/// branch below.
fn set_asset_ref(d: &Bound<'_, PyDict>, prefix: &str, asset: &AssetRef) -> PyResult<()> {
    d.set_item(format!("{prefix}_id"), asset.id)?;
    d.set_item(format!("{prefix}_type"), &asset.asset_type)?;
    d.set_item(format!("{prefix}_name"), &asset.name)?;
    Ok(())
}

/// Converts one `ModEffectOp` (Phase B1/B5) into the plain dict shape
/// that `dyna_step_native`'s `step_modification_record` crosses the FFI
/// boundary with. Every asset reference carries its `AssetRef` snapshot
/// (id + type + name) rather than a bare id - per `AssetRef`'s own doc
/// comment, the asset an op describes may already be gone from the live
/// `Model` by the time Python reads this record (e.g. a later op in the
/// same record removed it, or it's itself a removal), so Python must
/// never need to resolve these ids back through `model.assets` - see
/// `PORTING_NOTES.md` §6 Phase B5's differences-log entry.
fn mod_effect_op_to_py(py: Python<'_>, op: &ModEffectOp) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    match op {
        ModEffectOp::Asset(AssetOp::Added { asset }) => {
            d.set_item("kind", "asset")?;
            d.set_item("type", "ADDITIVE")?;
            set_asset_ref(&d, "asset", asset)?;
        }
        ModEffectOp::Asset(AssetOp::Removed { asset, .. }) => {
            d.set_item("kind", "asset")?;
            d.set_item("type", "SUBTRACTIVE")?;
            set_asset_ref(&d, "asset", asset)?;
        }
        ModEffectOp::Assoc(AssocOp::Added {
            left,
            field_name,
            right,
        }) => {
            d.set_item("kind", "assoc")?;
            d.set_item("type", "ADDITIVE")?;
            set_asset_ref(&d, "left_asset", left)?;
            d.set_item("field_name", field_name)?;
            set_asset_ref(&d, "right_asset", right)?;
        }
        ModEffectOp::Assoc(AssocOp::Removed {
            left,
            field_name,
            right,
        }) => {
            d.set_item("kind", "assoc")?;
            d.set_item("type", "SUBTRACTIVE")?;
            set_asset_ref(&d, "left_asset", left)?;
            d.set_item("field_name", field_name)?;
            set_asset_ref(&d, "right_asset", right)?;
        }
    }
    Ok(d.into())
}

fn stable_id_of(graph: &AttackGraph, id: AttackGraphNodeId) -> PyResult<i64> {
    graph
        .nodes
        .get(id)
        .map(|n| n.id)
        .ok_or_else(|| PyValueError::new_err(format!("node {id:?} is no longer in the graph")))
}

fn log_entry_to_py(py: Python<'_>, graph: &AttackGraph, log: &LogEntry) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("timestep", log.timestep)?;
    let (detector_node_id, detector_label) = &log.detector_id;
    // `detector_node_id`/`trigger` are expected to always be live here:
    // `collect_logs` already only ever constructs a `LogEntry` from a node
    // it just confirmed live, and nothing mutates the graph between there
    // and here - these `stable_id_of` calls should never actually hit
    // their error arm. Guarded anyway (rather than a direct `graph.
    // nodes[...]` index) so a future change that violates that invariant
    // surfaces as a catchable Python exception, not an uncatchable Rust
    // panic across the FFI boundary - the exact failure mode
    // `PORTING_NOTES.md` §0 B5 is about.
    d.set_item("detector_node_id", stable_id_of(graph, *detector_node_id)?)?;
    d.set_item("detector_label", detector_label)?;
    d.set_item("trigger", stable_id_of(graph, log.trigger)?)?;
    // Unlike `detector_node_id`/`trigger`, a context node id *can*
    // legitimately go stale: it's picked from `previous_compromised_nodes`
    // (history accumulated across the whole episode), so an id here may
    // name a node some earlier, unrelated step's model effect already
    // removed. Drop just that label rather than failing the whole log
    // entry - mirrors `collect_logs`'s "nothing left to report" choice.
    let context: HashMap<String, i64> = log
        .context
        .iter()
        .filter_map(|(label, &id)| graph.nodes.get(id).map(|n| (label.clone(), n.id)))
        .collect();
    d.set_item("context", context)?;
    d.set_item("false_positive", log.false_positive)?;
    Ok(d.into())
}
