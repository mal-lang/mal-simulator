//! `malsim._native.Simulator` - the Phase A8 native simulator pyclass,
//! extended at Phase A9 to back `MalSimulator.reset()`/`.step()` (see
//! `PORTING_NOTES.md` §5 Phase A8/A9/§10).
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
//! `extract_optional_ttc_dist_overrides`/`attacker_ttc_overrides` below.
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

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::str::FromStr;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rand::rngs::StdRng;
use rand::SeedableRng;

use malsim_core::attack_surface::{get_attack_surface, get_effects_of_attack_step};
use malsim_core::attacker_step::{attacker_is_terminated, attacker_step};
use malsim_core::defender_step::{defender_is_terminated, defender_step};
use malsim_core::defense_surface::get_defense_surface;
use malsim_core::event_logger::{collect_false_positives, collect_logs, LogEntry};
use malsim_core::graph_state::{
    attack_step_ttc_value, compute_initial_graph_state, is_impossible_attack_step, GraphState,
    TtcMode,
};
use malsim_core::observability::observed_nodes;
use malsim_core::ttc::{named_ttc_dist, DistFunction, Operation, TtcDist};

use crate::extract_shared_graph;

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn not_reset_err() -> PyErr {
    PyValueError::new_err("Simulator.step_native() called before reset_native()")
}

/// Per-defender intermediate results computed in `step_native`'s first
/// defender-update pass (defense surface, newly observed nodes, new logs,
/// previous `performed_nodes`) before the second pass applies them - kept
/// as a named alias purely to satisfy clippy's `type_complexity` lint.
type DefenderStepUpdate = (
    HashSet<AttackGraphNodeId>,
    HashSet<AttackGraphNodeId>,
    Vec<LogEntry>,
    HashSet<AttackGraphNodeId>,
);

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
fn stable_ids(graph: &AttackGraph, ids: impl IntoIterator<Item = AttackGraphNodeId>) -> Vec<i64> {
    let mut ids: Vec<i64> = ids.into_iter().map(|id| graph.nodes[id].id).collect();
    ids.sort_unstable();
    ids
}

fn id_value_map<V: Copy>(
    graph: &AttackGraph,
    map: &HashMap<AttackGraphNodeId, V>,
) -> HashMap<i64, V> {
    map.iter()
        .map(|(&id, &v)| (graph.nodes[id].id, v))
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

/// Computes an attacker's override-only `ttc_values`/`impossible_steps`
/// (§2.4/§10: when `ttc_dists` is configured, these fields hold *only*
/// the override-affected nodes - mirroring
/// `attacker_state_factories.py::attacker_overriding_ttc_settings`
/// exactly, not a merge with the graph-wide values).
fn attacker_ttc_overrides(
    graph: &AttackGraph,
    rng: &mut StdRng,
    ttc_mode: TtcMode,
    ttc_dist_overrides: &HashMap<AttackGraphNodeId, TtcDist>,
) -> PyResult<(HashMap<AttackGraphNodeId, f64>, HashSet<AttackGraphNodeId>)> {
    let mut ttc_values = HashMap::new();
    let mut impossible_steps = HashSet::new();
    for (&node_id, dist) in ttc_dist_overrides {
        let node = &graph.nodes[node_id];
        if let Some(v) =
            attack_step_ttc_value(node, Some(dist), ttc_mode, rng).map_err(to_py_err)?
        {
            ttc_values.insert(node_id, v);
        }
        if is_impossible_attack_step(node, Some(dist), rng).map_err(to_py_err)? {
            impossible_steps.insert(node_id);
        }
    }
    Ok((ttc_values, impossible_steps))
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

/// Port of the subset of `MalSimulatorSettings`/`AttackSurfaceSettings`
/// this module's hot loop reads - see module docs for what's deliberately
/// not here yet. Flattened to one dict (no nested `attack_surface` key)
/// since there's no settings struct on this side to mirror the nesting of.
struct NativeSettings {
    ttc_mode: TtcMode,
    run_defense_step_bernoullis: bool,
    run_attack_step_bernoullis: bool,
    skip_compromised: bool,
    skip_unnecessary: bool,
    compromise_entrypoints_at_start: bool,
}

fn parse_settings(dict: &Bound<'_, PyDict>) -> PyResult<NativeSettings> {
    let ttc_mode_str = match get_dict_value(dict, "ttc_mode")? {
        Some(v) => v.extract::<String>()?,
        None => "DISABLED".to_string(),
    };
    Ok(NativeSettings {
        ttc_mode: parse_ttc_mode(&ttc_mode_str)?,
        run_defense_step_bernoullis: get_bool(dict, "run_defense_step_bernoullis", true)?,
        run_attack_step_bernoullis: get_bool(dict, "run_attack_step_bernoullis", true)?,
        skip_compromised: get_bool(dict, "skip_compromised", true)?,
        skip_unnecessary: get_bool(dict, "skip_unnecessary", false)?,
        compromise_entrypoints_at_start: get_bool(dict, "compromise_entrypoints_at_start", true)?,
    })
}

// --- Per-agent runtime state ---

/// The Rust-side equivalent of the mutable parts of `AttackerState` this
/// phase needs (§3) - no `AttackerState` exists on the Rust side itself
/// yet, so this is a private, module-local struct, not a port of a
/// specific Python type.
struct AttackerRuntime {
    entry_points: HashSet<AttackGraphNodeId>,
    goals: HashSet<AttackGraphNodeId>,
    actionable_steps: Option<HashSet<AttackGraphNodeId>>,
    performed_nodes: HashSet<AttackGraphNodeId>,
    attempted_nodes: HashSet<AttackGraphNodeId>,
    action_surface: HashSet<AttackGraphNodeId>,
    num_attempts: HashMap<AttackGraphNodeId, u64>,
    iteration: u64,
    /// Per-node TTC distribution overrides (`AttackerSettings.ttc_dists`,
    /// already flattened - §2.4) - `None` when this attacker has no
    /// override configured, matching `attempt_attacker_step`'s existing
    /// `ttc_dist_overrides` parameter shape.
    ttc_dist_overrides: Option<HashMap<AttackGraphNodeId, TtcDist>>,
    /// The `AttackerState.ttc_values`/`.impossible_steps` fields this
    /// attacker exposes - override-only when `ttc_dist_overrides` is
    /// `Some`, otherwise a clone of the graph-wide values (see
    /// `attacker_ttc_overrides`'s doc comment).
    ttc_values: HashMap<AttackGraphNodeId, f64>,
    impossible_steps: HashSet<AttackGraphNodeId>,
}

/// The Rust-side equivalent of the mutable parts of `DefenderState` - see
/// `AttackerRuntime`'s doc comment.
struct DefenderRuntime {
    actionable_steps: Option<HashSet<AttackGraphNodeId>>,
    observable_steps: Option<HashSet<AttackGraphNodeId>>,
    false_positive_rates: Option<HashMap<AttackGraphNodeId, f64>>,
    false_negative_rates: Option<HashMap<AttackGraphNodeId, f64>>,
    performed_nodes: HashSet<AttackGraphNodeId>,
    compromised_nodes: HashSet<AttackGraphNodeId>,
    observed_nodes: HashSet<AttackGraphNodeId>,
    action_surface: HashSet<AttackGraphNodeId>,
    iteration: u64,
    logs: Vec<LogEntry>,
}

/// Everything computed/mutated by `reset_native`/`step_native`, as a
/// separate type from the pyclass itself so helper functions can take
/// `&mut SimState` directly rather than the whole `&mut Simulator` -
/// letting the borrow checker see `graph_state`/`enabled_defenses`/`rng`/
/// `attackers`/`defenders` as disjoint fields instead of one opaque `self`.
struct SimState {
    rng: StdRng,
    settings: NativeSettings,
    graph_state: GraphState,
    enabled_defenses: HashSet<AttackGraphNodeId>,
    attackers: HashMap<String, AttackerRuntime>,
    defenders: HashMap<String, DefenderRuntime>,
}

#[pyclass(name = "Simulator", module = "malsim._native", unsendable)]
pub struct Simulator {
    graph: Rc<RefCell<AttackGraph>>,
    state: Option<SimState>,
}

#[pymethods]
impl Simulator {
    #[new]
    fn new(graph: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Simulator {
            graph: extract_shared_graph(graph)?,
            state: None,
        })
    }

    /// Resets the simulator: (re)computes the initial graph state (TTC
    /// values, pre-enabled defenses, impossible attack steps, necessity -
    /// `graph_state::compute_initial_graph_state`, mirroring
    /// `graph_state.py`'s namesake) and (re)builds every agent's runtime
    /// state from scratch, mirroring `reset_agent.py`'s
    /// `reset_attackers`/`reset_defenders` - including entry-point
    /// compromise-at-start (`compromise_entrypoints_at_start`) and the
    /// resulting pre-compromised-nodes feed into every defender's initial
    /// `observed_nodes`/`logs`.
    fn reset_native(
        &mut self,
        py: Python<'_>,
        settings: &Bound<'_, PyDict>,
        agents: &Bound<'_, PyDict>,
        seed: u64,
    ) -> PyResult<Py<PyAny>> {
        let graph_rc = self.graph.clone();
        let native_settings = parse_settings(settings)?;
        let mut rng = StdRng::seed_from_u64(seed);

        let graph_state = {
            let graph = graph_rc.borrow();
            compute_initial_graph_state(
                &graph,
                native_settings.ttc_mode,
                native_settings.run_defense_step_bernoullis,
                native_settings.run_attack_step_bernoullis,
                &mut rng,
            )
            .map_err(to_py_err)?
        };
        let enabled_defenses = graph_state.pre_enabled_defenses.clone();

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

        let mut attackers: HashMap<String, AttackerRuntime> = HashMap::new();
        let mut pre_compromised: HashSet<AttackGraphNodeId> = HashSet::new();

        {
            let graph = graph_rc.borrow();
            for (name, cfg) in &attacker_cfgs {
                let entry_points =
                    extract_id_set(&graph, &require_dict_value(cfg, "entry_points")?)?;
                let goals = extract_optional_id_set(&graph, cfg, "goals")?.unwrap_or_default();
                let actionable_steps = extract_optional_id_set(&graph, cfg, "actionable_steps")?;

                let ttc_dist_overrides =
                    extract_optional_ttc_dist_overrides(&graph, cfg, "ttc_dists")?;
                let (ttc_values, impossible_steps) = match &ttc_dist_overrides {
                    Some(overrides) => attacker_ttc_overrides(
                        &graph,
                        &mut rng,
                        native_settings.ttc_mode,
                        overrides,
                    )?,
                    None => (
                        graph_state.ttc_values.clone(),
                        graph_state.impossible_attack_steps.clone(),
                    ),
                };

                let mut performed_nodes: HashSet<AttackGraphNodeId> = HashSet::new();
                if native_settings.compromise_entrypoints_at_start {
                    for &entry_point in &entry_points {
                        performed_nodes.insert(entry_point);
                        let effects = get_effects_of_attack_step(
                            &graph,
                            entry_point,
                            &performed_nodes,
                            &graph_state.impossible_attack_steps,
                            &enabled_defenses,
                            &graph_state.necessity_per_node,
                        )
                        .map_err(to_py_err)?;
                        performed_nodes.extend(effects);
                    }
                }

                let mut action_surface = get_attack_surface(
                    &graph,
                    native_settings.skip_compromised,
                    native_settings.skip_unnecessary,
                    actionable_steps.as_ref(),
                    &performed_nodes,
                    None,
                    &graph_state.impossible_attack_steps,
                    &enabled_defenses,
                    &graph_state.necessity_per_node,
                )
                .map_err(to_py_err)?;
                if !native_settings.compromise_entrypoints_at_start {
                    action_surface.extend(entry_points.iter().copied());
                }

                pre_compromised.extend(performed_nodes.iter().copied());

                attackers.insert(
                    name.clone(),
                    AttackerRuntime {
                        entry_points,
                        goals,
                        actionable_steps,
                        performed_nodes,
                        attempted_nodes: HashSet::new(),
                        action_surface,
                        num_attempts: HashMap::new(),
                        iteration: 0,
                        ttc_dist_overrides,
                        ttc_values,
                        impossible_steps,
                    },
                );
            }
        }

        let mut defenders: HashMap<String, DefenderRuntime> = HashMap::new();
        {
            let graph = graph_rc.borrow();
            for (name, cfg) in &defender_cfgs {
                let actionable_steps = extract_optional_id_set(&graph, cfg, "actionable_steps")?;
                let observable_steps = extract_optional_id_set(&graph, cfg, "observable_steps")?;
                let false_positive_rates =
                    extract_optional_rate_map(&graph, cfg, "false_positive_rates")?;
                let false_negative_rates =
                    extract_optional_rate_map(&graph, cfg, "false_negative_rates")?;

                let action_surface = get_defense_surface(
                    &graph,
                    actionable_steps.as_ref(),
                    &graph_state.impossible_attack_steps,
                    &enabled_defenses,
                )
                .map_err(to_py_err)?;

                let new_observed_nodes = observed_nodes(
                    observable_steps.as_ref(),
                    false_positive_rates.as_ref(),
                    false_negative_rates.as_ref(),
                    &graph,
                    &pre_compromised,
                    &mut rng,
                );

                let mut logs = collect_logs(
                    0,
                    &graph,
                    pre_compromised.iter().copied(),
                    &HashSet::new(),
                    &mut rng,
                )
                .map_err(to_py_err)?;
                logs.extend(collect_false_positives(0, &graph, &mut rng).map_err(to_py_err)?);

                defenders.insert(
                    name.clone(),
                    DefenderRuntime {
                        actionable_steps,
                        observable_steps,
                        false_positive_rates,
                        false_negative_rates,
                        performed_nodes: enabled_defenses.clone(),
                        compromised_nodes: pre_compromised.clone(),
                        observed_nodes: new_observed_nodes,
                        action_surface,
                        iteration: 0,
                        logs,
                    },
                );
            }
        }

        self.state = Some(SimState {
            rng,
            settings: native_settings,
            graph_state,
            enabled_defenses,
            attackers,
            defenders,
        });

        self.build_output(py)
    }

    /// Steps the simulation: defenders act first (`defender_step`,
    /// mirroring `simulator.py::step`'s own ordering), newly-enabled
    /// defenses are folded into `enabled_defenses` before any attacker
    /// acts, then attackers act (`attacker_step`), then every agent's
    /// runtime state (action surface, observed nodes, logs, ...) is
    /// recomputed from the step's results - same two-phase "compute all
    /// steps, then update all state" shape `simulator.py::step` uses.
    fn step_native(&mut self, py: Python<'_>, actions: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let graph_rc = self.graph.clone();
        let (attacker_names, defender_names) = {
            let state = self.state.as_ref().ok_or_else(not_reset_err)?;
            (
                state.attackers.keys().cloned().collect::<Vec<String>>(),
                state.defenders.keys().cloned().collect::<Vec<String>>(),
            )
        };

        // Mirrors `_pre_step_check`'s `KeyError` on an `actions` key that
        // names no registered agent.
        for key_obj in actions.keys() {
            let name: String = key_obj.extract()?;
            if !attacker_names.contains(&name) && !defender_names.contains(&name) {
                return Err(PyValueError::new_err(format!("No agent has name '{name}'")));
            }
        }

        let mut action_nodes: HashMap<String, Vec<AttackGraphNodeId>> = HashMap::new();
        {
            let graph = graph_rc.borrow();
            for name in attacker_names.iter().chain(defender_names.iter()) {
                let nodes = match actions.get_item(name)? {
                    Some(v) if !v.is_none() => extract_id_list(&graph, &v)?,
                    _ => Vec::new(),
                };
                action_nodes.insert(name.clone(), nodes);
            }
        }

        let state = self.state.as_mut().ok_or_else(not_reset_err)?;

        // --- Defenders act first ---
        let mut step_enabled_defenses: HashSet<AttackGraphNodeId> = HashSet::new();
        {
            let graph = graph_rc.borrow();
            for name in &defender_names {
                let runtime = &state.defenders[name];
                let enabled = defender_step(&graph, &action_nodes[name], &runtime.action_surface)
                    .map_err(to_py_err)?;
                step_enabled_defenses.extend(enabled);
            }
        }
        state
            .enabled_defenses
            .extend(step_enabled_defenses.iter().copied());

        // --- Attackers act afterwards ---
        let mut step_compromised_nodes: HashSet<AttackGraphNodeId> = HashSet::new();
        let mut attacker_results: HashMap<
            String,
            (Vec<AttackGraphNodeId>, Vec<AttackGraphNodeId>),
        > = HashMap::new();
        {
            let graph = graph_rc.borrow();
            for name in &attacker_names {
                let runtime = &state.attackers[name];
                let (compromised, attempted) = attacker_step(
                    &graph,
                    &mut state.rng,
                    state.settings.ttc_mode,
                    &action_nodes[name],
                    &runtime.entry_points,
                    &runtime.action_surface,
                    &runtime.performed_nodes,
                    &runtime.num_attempts,
                    runtime.ttc_dist_overrides.as_ref(),
                    Some(&runtime.ttc_values),
                    &state.graph_state.ttc_values,
                    &state.graph_state.impossible_attack_steps,
                    &state.enabled_defenses,
                    &state.graph_state.necessity_per_node,
                )
                .map_err(to_py_err)?;
                step_compromised_nodes.extend(compromised.iter().copied());
                attacker_results.insert(name.clone(), (compromised, attempted));
            }
        }

        // --- Update attacker runtimes (action surface recompute) ---
        {
            let graph = graph_rc.borrow();
            for name in &attacker_names {
                let (compromised, attempted) = attacker_results.remove(name).unwrap();
                let new_action_surface = {
                    let runtime = &state.attackers[name];
                    let mut performed_nodes = runtime.performed_nodes.clone();
                    performed_nodes.extend(compromised.iter().copied());
                    get_attack_surface(
                        &graph,
                        state.settings.skip_compromised,
                        state.settings.skip_unnecessary,
                        runtime.actionable_steps.as_ref(),
                        &performed_nodes,
                        None,
                        &state.graph_state.impossible_attack_steps,
                        &state.enabled_defenses,
                        &state.graph_state.necessity_per_node,
                    )
                    .map_err(to_py_err)?
                };

                let runtime = state.attackers.get_mut(name).unwrap();
                runtime.performed_nodes.extend(compromised.iter().copied());
                runtime.attempted_nodes.extend(attempted.iter().copied());
                for node_id in attempted {
                    *runtime.num_attempts.entry(node_id).or_insert(0) += 1;
                }
                runtime.action_surface = new_action_surface;
                runtime.iteration += 1;
            }
        }

        // --- Update defender runtimes (action surface, observed nodes, logs) ---
        {
            let graph = graph_rc.borrow();
            let mut pass1: HashMap<String, DefenderStepUpdate> = HashMap::new();

            for name in &defender_names {
                let runtime = &state.defenders[name];
                let previous_performed = runtime.performed_nodes.clone();

                let defense_surface_full = get_defense_surface(
                    &graph,
                    runtime.actionable_steps.as_ref(),
                    &state.graph_state.impossible_attack_steps,
                    &state.enabled_defenses,
                )
                .map_err(to_py_err)?;

                let new_observed = observed_nodes(
                    runtime.observable_steps.as_ref(),
                    runtime.false_positive_rates.as_ref(),
                    runtime.false_negative_rates.as_ref(),
                    &graph,
                    &step_compromised_nodes,
                    &mut state.rng,
                );

                let mut logs = collect_logs(
                    runtime.iteration,
                    &graph,
                    step_compromised_nodes.iter().copied(),
                    &runtime.compromised_nodes,
                    &mut state.rng,
                )
                .map_err(to_py_err)?;
                logs.extend(
                    collect_false_positives(runtime.iteration, &graph, &mut state.rng)
                        .map_err(to_py_err)?,
                );

                pass1.insert(
                    name.clone(),
                    (defense_surface_full, new_observed, logs, previous_performed),
                );
            }

            for name in &defender_names {
                let (defense_surface_full, new_observed, logs, previous_performed) =
                    pass1.remove(name).unwrap();
                let runtime = state.defenders.get_mut(name).unwrap();
                runtime
                    .performed_nodes
                    .extend(step_enabled_defenses.iter().copied());
                runtime.action_surface = defense_surface_full
                    .difference(&previous_performed)
                    .copied()
                    .collect();
                runtime.observed_nodes.extend(new_observed);
                runtime
                    .compromised_nodes
                    .extend(step_compromised_nodes.iter().copied());
                runtime.logs.extend(logs);
                runtime.iteration += 1;
            }
        }

        self.build_output(py)
    }
}

impl Simulator {
    /// Builds the plain-`dict` return value shared by `reset_native`/
    /// `step_native` - see module docs for the exact shape. Deliberately
    /// no custom pyclasses in the return value (per A8's own plan text:
    /// "keep it boring and inspectable").
    fn build_output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = self.state.as_ref().ok_or_else(not_reset_err)?;
        let graph = self.graph.borrow();

        let sim_state = PyDict::new(py);
        sim_state.set_item(
            "enabled_defenses",
            stable_ids(&graph, state.enabled_defenses.iter().copied()),
        )?;
        sim_state.set_item(
            "ttc_values",
            id_value_map(&graph, &state.graph_state.ttc_values),
        )?;
        sim_state.set_item(
            "impossible_attack_steps",
            stable_ids(
                &graph,
                state.graph_state.impossible_attack_steps.iter().copied(),
            ),
        )?;
        sim_state.set_item(
            "necessity_per_node",
            id_value_map(&graph, &state.graph_state.necessity_per_node),
        )?;
        sim_state.set_item(
            "pre_enabled_defenses",
            stable_ids(
                &graph,
                state.graph_state.pre_enabled_defenses.iter().copied(),
            ),
        )?;

        let attacker_triples: Vec<_> = state
            .attackers
            .values()
            .map(|a| (&a.action_surface, &a.goals, &a.performed_nodes))
            .collect();
        let defenders_terminated =
            defender_is_terminated(attacker_triples.iter().map(|&(s, g, p)| (s, g, p)));

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
            let num_attempts: HashMap<i64, u64> = a
                .num_attempts
                .iter()
                .map(|(&id, &n)| (graph.nodes[id].id, n))
                .collect();
            d.set_item("num_attempts", num_attempts)?;
            d.set_item("ttc_values", id_value_map(&graph, &a.ttc_values))?;
            d.set_item(
                "impossible_steps",
                stable_ids(&graph, a.impossible_steps.iter().copied()),
            )?;
            d.set_item("iteration", a.iteration)?;
            d.set_item(
                "terminated",
                attacker_is_terminated(&a.action_surface, &a.goals, &a.performed_nodes),
            )?;
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
}

fn log_entry_to_py(py: Python<'_>, graph: &AttackGraph, log: &LogEntry) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("timestep", log.timestep)?;
    let (detector_node_id, detector_label) = &log.detector_id;
    d.set_item("detector_node_id", graph.nodes[*detector_node_id].id)?;
    d.set_item("detector_label", detector_label)?;
    d.set_item("trigger", graph.nodes[log.trigger].id)?;
    let context: HashMap<String, i64> = log
        .context
        .iter()
        .map(|(label, &id)| (label.clone(), graph.nodes[id].id))
        .collect();
    d.set_item("context", context)?;
    d.set_item("false_positive", log.false_positive)?;
    Ok(d.into())
}
