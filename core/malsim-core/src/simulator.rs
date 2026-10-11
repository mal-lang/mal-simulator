//! The pure-Rust reset/step orchestration of `malsim`'s simulator over a
//! *static* instance model/attack graph: a port of
//! `python/malsim/mal_simulator/simulator.py::reset`/`step`, moved here from
//! `malsim-pyo3`'s `Simulator` pyclass so it can be driven without Python.
//! The dynamic counterpart, where model effects mutate the `Model`/
//! `AttackGraph` mid-episode, is [`crate::dyna_simulator::DynaSimulator`]: it
//! wraps a `Simulator` and reuses its reset and per-agent bookkeeping,
//! swapping in only the dyna step functions (`PORTING_NOTES.md` §11).
//!
//! `malsim-pyo3` is a thin wrapper around both: it parses Python dicts into
//! `MalSimulatorSettings`/`FlatAgentSettings`, calls `reset`/`step`, and
//! builds its plain-`dict` output from `StepOutcome` + `state()`.
//!
//! All node references here are `AttackGraphNodeId` slotmap keys; the
//! stable `AttackGraphNode.id: i64` translation stays at the FFI boundary.
//!
//! Agents are kept in `BTreeMap`s keyed by name, and every loop over agents
//! in `reset`/`step` iterates in that (sorted-name) order, so multi-agent
//! RNG consumption is deterministic across processes for the same seed.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::attack_surface::{get_attack_surface, get_effects_of_attack_step};
use crate::attacker_step::{attacker_step, AttackerStepError};
use crate::defender_step::{defender_step, DefenderStepError};
use crate::defense_surface::get_defense_surface;
use crate::dyna_attacker_step::DynaAttackerStepError;
use crate::dyna_defender_step::DynaDefenderStepError;
use crate::event_logger::{collect_false_positives, collect_logs, EventLoggerError, LogEntry};
use crate::graph_state::{
    attack_step_ttc_value, compute_initial_graph_state, is_impossible_attack_step, GraphState,
    GraphStateError, TtcMode,
};
use crate::graph_utils::GraphUtilsError;
use crate::model_state::ModelStateError;
use crate::observability::observed_nodes;
use crate::settings::{
    FlatAgentSettings, FlatAttackerSettings, FlatDefenderSettings, MalSimulatorSettings,
};
use crate::ttc::TtcDist;

#[derive(Debug)]
pub enum SimulatorError {
    /// `Simulator::step` was called before any successful
    /// `Simulator::reset`.
    NotReset,
    /// Mirrors `_pre_step_check`'s `KeyError` on an `actions` key that
    /// names no registered agent.
    UnknownAgent(String),
    AttackerStep(AttackerStepError),
    DefenderStep(DefenderStepError),
    DynaAttackerStep(DynaAttackerStepError),
    DynaDefenderStep(DynaDefenderStepError),
    GraphState(GraphStateError),
    GraphUtils(GraphUtilsError),
    EventLogger(EventLoggerError),
    /// Boxed (same as `DynaAttackerStepError::ModelEffects`) to keep
    /// `Result<_, SimulatorError>` small - clippy's `result_large_err`.
    ModelState(Box<ModelStateError>),
}

impl From<AttackerStepError> for SimulatorError {
    fn from(e: AttackerStepError) -> Self {
        SimulatorError::AttackerStep(e)
    }
}

impl From<DefenderStepError> for SimulatorError {
    fn from(e: DefenderStepError) -> Self {
        SimulatorError::DefenderStep(e)
    }
}

impl From<DynaAttackerStepError> for SimulatorError {
    fn from(e: DynaAttackerStepError) -> Self {
        SimulatorError::DynaAttackerStep(e)
    }
}

impl From<DynaDefenderStepError> for SimulatorError {
    fn from(e: DynaDefenderStepError) -> Self {
        SimulatorError::DynaDefenderStep(e)
    }
}

impl From<GraphStateError> for SimulatorError {
    fn from(e: GraphStateError) -> Self {
        SimulatorError::GraphState(e)
    }
}

impl From<GraphUtilsError> for SimulatorError {
    fn from(e: GraphUtilsError) -> Self {
        SimulatorError::GraphUtils(e)
    }
}

impl From<EventLoggerError> for SimulatorError {
    fn from(e: EventLoggerError) -> Self {
        SimulatorError::EventLogger(e)
    }
}

impl From<ModelStateError> for SimulatorError {
    fn from(e: ModelStateError) -> Self {
        SimulatorError::ModelState(Box::new(e))
    }
}

impl fmt::Display for SimulatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulatorError::NotReset => write!(f, "Simulator.step() called before reset()"),
            SimulatorError::UnknownAgent(name) => write!(f, "No agent has name '{name}'"),
            SimulatorError::AttackerStep(e) => write!(f, "{e}"),
            SimulatorError::DefenderStep(e) => write!(f, "{e}"),
            SimulatorError::DynaAttackerStep(e) => write!(f, "{e}"),
            SimulatorError::DynaDefenderStep(e) => write!(f, "{e}"),
            SimulatorError::GraphState(e) => write!(f, "{e}"),
            SimulatorError::GraphUtils(e) => write!(f, "{e}"),
            SimulatorError::EventLogger(e) => write!(f, "{e}"),
            SimulatorError::ModelState(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SimulatorError {}

// --- Per-agent runtime state ---

/// The Rust-side equivalent of the mutable parts of `AttackerState` the
/// simulator needs (§3) - not a port of a specific Python type.
#[derive(Debug, Clone, PartialEq)]
pub struct AttackerRuntime {
    pub entry_points: HashSet<AttackGraphNodeId>,
    pub goals: HashSet<AttackGraphNodeId>,
    pub actionable_steps: Option<HashSet<AttackGraphNodeId>>,
    pub performed_nodes: HashSet<AttackGraphNodeId>,
    pub attempted_nodes: HashSet<AttackGraphNodeId>,
    pub action_surface: HashSet<AttackGraphNodeId>,
    pub num_attempts: HashMap<AttackGraphNodeId, u64>,
    pub iteration: u64,
    /// Per-node TTC distribution overrides (`AttackerSettings.ttc_dists`,
    /// already flattened - §2.4) - `None` when this attacker has no
    /// override configured, matching `attempt_attacker_step`'s existing
    /// `ttc_dist_overrides` parameter shape.
    pub ttc_dist_overrides: Option<HashMap<AttackGraphNodeId, TtcDist>>,
    /// The `AttackerState.ttc_values`/`.impossible_steps` fields this
    /// attacker exposes - override-only when `ttc_dist_overrides` is
    /// `Some`, otherwise a clone of the graph-wide values (see
    /// `attacker_ttc_overrides`'s doc comment).
    pub ttc_values: HashMap<AttackGraphNodeId, f64>,
    pub impossible_steps: HashSet<AttackGraphNodeId>,
}

/// The Rust-side equivalent of the mutable parts of `DefenderState` - see
/// `AttackerRuntime`'s doc comment.
#[derive(Debug, Clone, PartialEq)]
pub struct DefenderRuntime {
    pub actionable_steps: Option<HashSet<AttackGraphNodeId>>,
    pub observable_steps: Option<HashSet<AttackGraphNodeId>>,
    pub false_positive_rates: Option<HashMap<AttackGraphNodeId, f64>>,
    pub false_negative_rates: Option<HashMap<AttackGraphNodeId, f64>>,
    pub performed_nodes: HashSet<AttackGraphNodeId>,
    pub compromised_nodes: HashSet<AttackGraphNodeId>,
    pub observed_nodes: HashSet<AttackGraphNodeId>,
    pub action_surface: HashSet<AttackGraphNodeId>,
    pub iteration: u64,
    pub logs: Vec<LogEntry>,
}

/// Everything computed/mutated by `Simulator::reset`/`Simulator::step`, as
/// a separate type from `Simulator` itself so helper functions can take
/// `&mut SimState` directly - letting the borrow checker see `graph_state`/
/// `enabled_defenses`/`rng`/`attackers`/`defenders` as disjoint fields.
#[derive(Debug)]
pub struct SimState {
    pub(crate) rng: StdRng,
    pub settings: MalSimulatorSettings,
    pub graph_state: GraphState,
    pub enabled_defenses: HashSet<AttackGraphNodeId>,
    pub attackers: BTreeMap<String, AttackerRuntime>,
    pub defenders: BTreeMap<String, DefenderRuntime>,
}

impl SimState {
    /// Port of `attacker_is_terminated` for the attacker named `name`;
    /// `None` when no attacker has that name.
    pub fn attacker_is_terminated(&self, name: &str) -> Option<bool> {
        self.attackers.get(name).map(|a| {
            crate::attacker_step::attacker_is_terminated(
                &a.action_surface,
                &a.goals,
                &a.performed_nodes,
            )
        })
    }

    /// Port of `defender_is_terminated`: shared by every defender, true once
    /// every attacker is terminated (vacuously true with no attackers).
    pub fn defender_is_terminated(&self) -> bool {
        crate::defender_step::defender_is_terminated(
            self.attackers
                .values()
                .map(|a| (&a.action_surface, &a.goals, &a.performed_nodes)),
        )
    }
}

/// One attacker's deltas from a single `Simulator::step` (or
/// `DynaSimulator::step`) - the nodes this
/// step compromised/attempted, not the episode-accumulated sets (those
/// live in `AttackerRuntime`).
#[derive(Debug, Clone, PartialEq)]
pub struct AttackerStepOutcome {
    pub step_performed_nodes: HashSet<AttackGraphNodeId>,
    pub step_attempted_nodes: HashSet<AttackGraphNodeId>,
}

/// One defender's deltas from a single `Simulator::step` - newly observed
/// nodes and newly produced logs.
#[derive(Debug, Clone, PartialEq)]
pub struct DefenderStepOutcome {
    pub step_observed_nodes: HashSet<AttackGraphNodeId>,
    pub step_logs: Vec<LogEntry>,
}

/// Everything a single `Simulator::step` produced, as deltas.
#[derive(Debug, Clone)]
pub struct StepOutcome {
    pub attackers: BTreeMap<String, AttackerStepOutcome>,
    pub defenders: BTreeMap<String, DefenderStepOutcome>,
    /// Defenses enabled by this step's defender actions.
    pub step_enabled_defenses: HashSet<AttackGraphNodeId>,
}

/// Port of `MalSimulator`'s reset/step orchestration, over a static
/// instance model/attack graph. See `DynaSimulator` for the dynamic one.
pub struct Simulator {
    graph: Rc<RefCell<AttackGraph>>,
    state: Option<SimState>,
}

impl Simulator {
    /// A simulator over the static `graph`.
    pub fn new(graph: Rc<RefCell<AttackGraph>>) -> Self {
        Simulator { graph, state: None }
    }

    pub fn graph(&self) -> &Rc<RefCell<AttackGraph>> {
        &self.graph
    }

    /// The state built by the last successful `reset` (and advanced by every
    /// `step` since) - `None` before the first successful `reset`.
    pub fn state(&self) -> Option<&SimState> {
        self.state.as_ref()
    }

    /// `state`, mutably, for `DynaSimulator::step`.
    pub(crate) fn state_mut(&mut self) -> Result<&mut SimState, SimulatorError> {
        self.state.as_mut().ok_or(SimulatorError::NotReset)
    }

    /// Resets the simulator: (re)computes the initial graph state (TTC values, pre-enabled
    /// defenses, impossible attack steps, necessity -
    /// `graph_state::compute_initial_graph_state`) and (re)builds every
    /// agent's runtime state from scratch, mirroring `reset_agent.py`'s
    /// `reset_attackers`/`reset_defenders` - including entry-point
    /// compromise-at-start (`compromise_entrypoints_at_start`) and the
    /// resulting pre-compromised-nodes feed into every defender's initial
    /// `observed_nodes`/`logs`. Attackers are reset before defenders, each
    /// in name order. On error, the previous state (if any) is kept.
    pub fn reset(
        &mut self,
        settings: &MalSimulatorSettings,
        agents: Vec<(String, FlatAgentSettings)>,
        seed: u64,
    ) -> Result<&SimState, SimulatorError> {
        let state = reset_state(&self.graph.borrow(), *settings, agents, seed)?;
        Ok(self.state.insert(state))
    }

    /// Steps the simulation: defenders act first (mirroring
    /// `simulator.py::step`'s own ordering), newly-enabled defenses are
    /// folded into `enabled_defenses` before any attacker acts, then
    /// attackers act, then every agent's runtime state (action surface,
    /// observed nodes, logs, ...) is recomputed from the step's results -
    /// same two-phase "compute all steps, then update all state" shape
    /// `simulator.py::step` uses.
    ///
    /// Defenders/attackers act via `defender_step`/`attacker_step`.
    /// `actions` maps agent name to chosen nodes; an agent missing
    /// from it takes no action, and a name that is no registered agent is
    /// an error.
    pub fn step(
        &mut self,
        actions: &HashMap<String, Vec<AttackGraphNodeId>>,
    ) -> Result<StepOutcome, SimulatorError> {
        let state = self.state.as_mut().ok_or(SimulatorError::NotReset)?;
        check_action_agents(state, actions)?;
        plain_step(&self.graph, state, actions)
    }
}

/// Mirrors `_pre_step_check`'s `KeyError` on an `actions` key that names no
/// registered agent. The smallest unknown name is reported, so the error
/// doesn't depend on `HashMap` order.
pub(crate) fn check_action_agents(
    state: &SimState,
    actions: &HashMap<String, Vec<AttackGraphNodeId>>,
) -> Result<(), SimulatorError> {
    match actions
        .keys()
        .filter(|name| !state.attackers.contains_key(*name) && !state.defenders.contains_key(*name))
        .min()
    {
        Some(unknown) => Err(SimulatorError::UnknownAgent(unknown.clone())),
        None => Ok(()),
    }
}

/// The nodes `name` chose this step - none when `actions` doesn't name it.
pub(crate) fn agent_actions<'a>(
    actions: &'a HashMap<String, Vec<AttackGraphNodeId>>,
    name: &str,
) -> &'a [AttackGraphNodeId] {
    actions.get(name).map(Vec::as_slice).unwrap_or(&[])
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
) -> Result<(HashMap<AttackGraphNodeId, f64>, HashSet<AttackGraphNodeId>), SimulatorError> {
    let mut ttc_values = HashMap::new();
    let mut impossible_steps = HashSet::new();
    for (&node_id, dist) in ttc_dist_overrides {
        // `ttc_dist_overrides` is resolved once at reset and never
        // updated - a dyna-path node it names can later be removed from
        // the graph. A removed node has no TTC left to override.
        let Some(node) = graph.nodes.get(node_id) else {
            continue;
        };
        if let Some(v) = attack_step_ttc_value(node, Some(dist), ttc_mode, rng)? {
            ttc_values.insert(node_id, v);
        }
        if is_impossible_attack_step(node, Some(dist), rng)? {
            impossible_steps.insert(node_id);
        }
    }
    Ok((ttc_values, impossible_steps))
}

/// Shared body of `Simulator::reset` and `DynaSimulator::reset` (after its
/// model restore) - see `Simulator::reset`'s doc comment.
pub(crate) fn reset_state(
    graph: &AttackGraph,
    settings: MalSimulatorSettings,
    agents: Vec<(String, FlatAgentSettings)>,
    seed: u64,
) -> Result<SimState, SimulatorError> {
    let mut rng = StdRng::seed_from_u64(seed);

    let graph_state = compute_initial_graph_state(
        graph,
        settings.ttc_mode,
        settings.run_defense_step_bernoullis,
        settings.run_attack_step_bernoullis,
        &mut rng,
    )?;
    let enabled_defenses = graph_state.pre_enabled_defenses.clone();

    let mut attacker_cfgs: BTreeMap<String, FlatAttackerSettings> = BTreeMap::new();
    let mut defender_cfgs: BTreeMap<String, FlatDefenderSettings> = BTreeMap::new();
    for (name, cfg) in agents {
        match cfg {
            FlatAgentSettings::Attacker(a) => {
                attacker_cfgs.insert(name, a);
            }
            FlatAgentSettings::Defender(d) => {
                defender_cfgs.insert(name, d);
            }
        }
    }

    let mut attackers: BTreeMap<String, AttackerRuntime> = BTreeMap::new();
    let mut pre_compromised: HashSet<AttackGraphNodeId> = HashSet::new();

    for (name, cfg) in attacker_cfgs {
        let FlatAttackerSettings {
            entry_points,
            goals,
            actionable_steps,
            ttc_dists: ttc_dist_overrides,
        } = cfg;

        let (ttc_values, impossible_steps) = match &ttc_dist_overrides {
            Some(overrides) => {
                attacker_ttc_overrides(graph, &mut rng, settings.ttc_mode, overrides)?
            }
            None => (
                graph_state.ttc_values.clone(),
                graph_state.impossible_attack_steps.clone(),
            ),
        };

        let mut performed_nodes: HashSet<AttackGraphNodeId> = HashSet::new();
        if settings.compromise_entrypoints_at_start {
            for &entry_point in &entry_points {
                performed_nodes.insert(entry_point);
                let effects = get_effects_of_attack_step(
                    graph,
                    entry_point,
                    &performed_nodes,
                    &graph_state.impossible_attack_steps,
                    &enabled_defenses,
                    &graph_state.necessity_per_node,
                )?;
                performed_nodes.extend(effects);
            }
        }

        let mut action_surface = get_attack_surface(
            graph,
            settings.attack_surface.skip_compromised,
            settings.attack_surface.skip_unnecessary,
            actionable_steps.as_ref(),
            &performed_nodes,
            None,
            &graph_state.impossible_attack_steps,
            &enabled_defenses,
            &graph_state.necessity_per_node,
        )?;
        if !settings.compromise_entrypoints_at_start {
            action_surface.extend(entry_points.iter().copied());
        }

        pre_compromised.extend(performed_nodes.iter().copied());

        attackers.insert(
            name,
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

    let mut defenders: BTreeMap<String, DefenderRuntime> = BTreeMap::new();
    for (name, cfg) in defender_cfgs {
        let FlatDefenderSettings {
            actionable_steps,
            observable_steps,
            false_positive_rates,
            false_negative_rates,
        } = cfg;

        let action_surface = get_defense_surface(
            graph,
            actionable_steps.as_ref(),
            &graph_state.impossible_attack_steps,
            &enabled_defenses,
        )?;

        let new_observed_nodes = observed_nodes(
            observable_steps.as_ref(),
            false_positive_rates.as_ref(),
            false_negative_rates.as_ref(),
            graph,
            &pre_compromised,
            &mut rng,
        );

        let mut logs = collect_logs(
            0,
            graph,
            pre_compromised.iter().copied(),
            &HashSet::new(),
            &mut rng,
        )?;
        logs.extend(collect_false_positives(0, graph, &mut rng)?);

        defenders.insert(
            name,
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

    Ok(SimState {
        rng,
        settings,
        graph_state,
        enabled_defenses,
        attackers,
        defenders,
    })
}

/// Each attacker's `(compromised, attempted)` nodes from this step's
/// stepping pass, before `update_attacker_runtimes` folds them in.
pub(crate) type AttackerResults =
    BTreeMap<String, (Vec<AttackGraphNodeId>, Vec<AttackGraphNodeId>)>;

/// `Simulator::step` for a plain simulator (`defender_step`/
/// `attacker_step`).
fn plain_step(
    graph_rc: &Rc<RefCell<AttackGraph>>,
    state: &mut SimState,
    actions: &HashMap<String, Vec<AttackGraphNodeId>>,
) -> Result<StepOutcome, SimulatorError> {
    let attacker_names: Vec<String> = state.attackers.keys().cloned().collect();
    let defender_names: Vec<String> = state.defenders.keys().cloned().collect();

    // --- Defenders act first ---
    let mut step_enabled_defenses: HashSet<AttackGraphNodeId> = HashSet::new();
    {
        let graph = graph_rc.borrow();
        for name in &defender_names {
            let runtime = &state.defenders[name];
            let enabled = defender_step(
                &graph,
                agent_actions(actions, name),
                &runtime.action_surface,
            )?;
            step_enabled_defenses.extend(enabled);
        }
    }
    state
        .enabled_defenses
        .extend(step_enabled_defenses.iter().copied());

    // --- Attackers act afterwards ---
    let mut step_compromised_nodes: HashSet<AttackGraphNodeId> = HashSet::new();
    let mut attacker_results: AttackerResults = BTreeMap::new();
    {
        let graph = graph_rc.borrow();
        for name in &attacker_names {
            let runtime = &state.attackers[name];
            let (compromised, attempted) = attacker_step(
                &graph,
                &mut state.rng,
                state.settings.ttc_mode,
                agent_actions(actions, name),
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
            )?;
            step_compromised_nodes.extend(compromised.iter().copied());
            attacker_results.insert(name.clone(), (compromised, attempted));
        }
    }

    let graph = graph_rc.borrow();
    let attackers = update_attacker_runtimes(&graph, state, &attacker_names, attacker_results)?;
    let defenders = update_defender_runtimes(
        &graph,
        state,
        &defender_names,
        &step_enabled_defenses,
        &step_compromised_nodes,
    )?;

    Ok(StepOutcome {
        attackers,
        defenders,
        step_enabled_defenses,
    })
}

/// Recomputes each attacker's bookkeeping (`action_surface`,
/// `performed_nodes`, `attempted_nodes`, `num_attempts`, `iteration`)
/// after this step's compromises/attempts - shared by `plain_step` and
/// `DynaSimulator::step`: identical regardless of whether the compromises came from
/// plain `attacker_step` or model-effect-aware `dyna_attacker_step`, since
/// both only ever grow `graph_state`/`enabled_defenses`/the graph itself
/// and this just reads whatever the graph looks like once stepping has
/// finished.
pub(crate) fn update_attacker_runtimes(
    graph: &AttackGraph,
    state: &mut SimState,
    attacker_names: &[String],
    mut attacker_results: AttackerResults,
) -> Result<BTreeMap<String, AttackerStepOutcome>, SimulatorError> {
    let mut attacker_step_deltas = BTreeMap::new();
    for name in attacker_names {
        let (compromised, attempted) = attacker_results
            .remove(name)
            .expect("every attacker stepped this step");
        let new_action_surface = {
            let runtime = &state.attackers[name];
            let mut performed_nodes = runtime.performed_nodes.clone();
            performed_nodes.extend(compromised.iter().copied());
            get_attack_surface(
                graph,
                state.settings.attack_surface.skip_compromised,
                state.settings.attack_surface.skip_unnecessary,
                runtime.actionable_steps.as_ref(),
                &performed_nodes,
                None,
                &state.graph_state.impossible_attack_steps,
                &state.enabled_defenses,
                &state.graph_state.necessity_per_node,
            )?
        };

        let runtime = state
            .attackers
            .get_mut(name)
            .expect("attacker_names come from state.attackers");
        runtime.performed_nodes.extend(compromised.iter().copied());
        runtime.attempted_nodes.extend(attempted.iter().copied());
        for &node_id in &attempted {
            *runtime.num_attempts.entry(node_id).or_insert(0) += 1;
        }
        runtime.action_surface = new_action_surface;
        runtime.iteration += 1;

        attacker_step_deltas.insert(
            name.clone(),
            AttackerStepOutcome {
                step_performed_nodes: compromised.into_iter().collect(),
                step_attempted_nodes: attempted.into_iter().collect(),
            },
        );
    }
    Ok(attacker_step_deltas)
}

/// Per-defender intermediate results computed in
/// `update_defender_runtimes`'s first pass (defense surface, newly observed
/// nodes, new logs, previous `performed_nodes`) before the second pass
/// applies them.
type DefenderStepUpdate = (
    HashSet<AttackGraphNodeId>,
    HashSet<AttackGraphNodeId>,
    Vec<LogEntry>,
    HashSet<AttackGraphNodeId>,
);

/// Recomputes each defender's bookkeeping (`action_surface`,
/// `performed_nodes`, `compromised_nodes`, `observed_nodes`, `logs`,
/// `iteration`) after this step's enables/compromises - shared by
/// `plain_step` and `DynaSimulator::step`, same reasoning as `update_attacker_runtimes`.
pub(crate) fn update_defender_runtimes(
    graph: &AttackGraph,
    state: &mut SimState,
    defender_names: &[String],
    step_enabled_defenses: &HashSet<AttackGraphNodeId>,
    step_compromised_nodes: &HashSet<AttackGraphNodeId>,
) -> Result<BTreeMap<String, DefenderStepOutcome>, SimulatorError> {
    let mut defender_step_deltas = BTreeMap::new();
    let mut pass1: BTreeMap<String, DefenderStepUpdate> = BTreeMap::new();

    for name in defender_names {
        let runtime = &state.defenders[name];
        let previous_performed = runtime.performed_nodes.clone();

        let defense_surface_full = get_defense_surface(
            graph,
            runtime.actionable_steps.as_ref(),
            &state.graph_state.impossible_attack_steps,
            &state.enabled_defenses,
        )?;

        let new_observed = observed_nodes(
            runtime.observable_steps.as_ref(),
            runtime.false_positive_rates.as_ref(),
            runtime.false_negative_rates.as_ref(),
            graph,
            step_compromised_nodes,
            &mut state.rng,
        );

        let mut logs = collect_logs(
            runtime.iteration,
            graph,
            step_compromised_nodes.iter().copied(),
            &runtime.compromised_nodes,
            &mut state.rng,
        )?;
        logs.extend(collect_false_positives(
            runtime.iteration,
            graph,
            &mut state.rng,
        )?);

        pass1.insert(
            name.clone(),
            (defense_surface_full, new_observed, logs, previous_performed),
        );
    }

    for name in defender_names {
        let (defense_surface_full, new_observed, logs, previous_performed) = pass1
            .remove(name)
            .expect("every defender went through pass 1");
        let runtime = state
            .defenders
            .get_mut(name)
            .expect("defender_names come from state.defenders");
        runtime
            .performed_nodes
            .extend(step_enabled_defenses.iter().copied());
        runtime.action_surface = defense_surface_full
            .difference(&previous_performed)
            .copied()
            .collect();
        runtime.observed_nodes.extend(new_observed.iter().copied());
        runtime
            .compromised_nodes
            .extend(step_compromised_nodes.iter().copied());
        runtime.logs.extend(logs.iter().cloned());
        runtime.iteration += 1;

        defender_step_deltas.insert(
            name.clone(),
            DefenderStepOutcome {
                step_observed_nodes: new_observed,
                step_logs: logs,
            },
        );
    }

    Ok(defender_step_deltas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph, wiper_attack_graph};

    /// Settings with every bernoulli off, so tests are not at the mercy of
    /// sampled impossible steps/pre-enabled defenses.
    fn settings() -> MalSimulatorSettings {
        MalSimulatorSettings {
            run_defense_step_bernoullis: false,
            run_attack_step_bernoullis: false,
            ..MalSimulatorSettings::default()
        }
    }

    fn attacker(entry_points: &[AttackGraphNodeId]) -> FlatAgentSettings {
        FlatAgentSettings::Attacker(FlatAttackerSettings {
            entry_points: entry_points.iter().copied().collect(),
            ..FlatAttackerSettings::default()
        })
    }

    fn defender() -> FlatAgentSettings {
        FlatAgentSettings::Defender(FlatDefenderSettings::default())
    }

    /// A dummy graph: `entry -> child` (an OR attack step reachable from the
    /// entry point) plus one unenabled defense.
    struct DummyNodes {
        entry: AttackGraphNodeId,
        child: AttackGraphNodeId,
        defense: AttackGraphNodeId,
    }

    fn dummy_sim() -> (Simulator, DummyNodes) {
        let mut graph = dummy_graph();
        let entry = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let child = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let defense = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        graph.nodes[entry].children.insert(child);
        graph.nodes[child].parents.insert(entry);
        (
            Simulator::new(Rc::new(RefCell::new(graph))),
            DummyNodes {
                entry,
                child,
                defense,
            },
        )
    }

    fn actions(
        entries: &[(&str, Vec<AttackGraphNodeId>)],
    ) -> HashMap<String, Vec<AttackGraphNodeId>> {
        entries
            .iter()
            .map(|(name, nodes)| (name.to_string(), nodes.clone()))
            .collect()
    }

    #[test]
    fn reset_builds_attacker_and_defender_runtimes() {
        let (mut sim, n) = dummy_sim();
        let state = sim
            .reset(
                &settings(),
                vec![
                    ("attacker".to_string(), attacker(&[n.entry])),
                    ("defender".to_string(), defender()),
                ],
                1,
            )
            .unwrap();

        let a = &state.attackers["attacker"];
        assert!(a.performed_nodes.contains(&n.entry));
        assert_eq!(a.action_surface, [n.child].into_iter().collect());
        assert_eq!(a.iteration, 0);
        assert_eq!(state.attacker_is_terminated("attacker"), Some(false));
        assert_eq!(state.attacker_is_terminated("nobody"), None);

        let d = &state.defenders["defender"];
        assert_eq!(d.action_surface, [n.defense].into_iter().collect());
        assert!(d.compromised_nodes.contains(&n.entry));
        assert!(d.performed_nodes.is_empty());
        assert!(!state.defender_is_terminated());
    }

    #[test]
    fn reset_on_wiper_fixture_without_entrypoint_compromise() {
        let (graph, _model) = wiper_attack_graph();
        let infect = graph.full_name_to_node["InfectedDevice:infect"];
        let mut sim = Simulator::new(Rc::new(RefCell::new(graph)));
        let s = MalSimulatorSettings {
            compromise_entrypoints_at_start: false,
            ..settings()
        };
        let state = sim
            .reset(
                &s,
                vec![
                    ("WiperController".to_string(), attacker(&[infect])),
                    ("defender".to_string(), defender()),
                ],
                42,
            )
            .unwrap();
        let a = &state.attackers["WiperController"];
        assert!(a.performed_nodes.is_empty());
        assert!(a.action_surface.contains(&infect));
    }

    #[test]
    fn step_compromises_node_on_attack_surface() {
        let (mut sim, n) = dummy_sim();
        sim.reset(
            &settings(),
            vec![
                ("attacker".to_string(), attacker(&[n.entry])),
                ("defender".to_string(), defender()),
            ],
            1,
        )
        .unwrap();

        let outcome = sim.step(&actions(&[("attacker", vec![n.child])])).unwrap();
        let delta = &outcome.attackers["attacker"];
        assert_eq!(delta.step_performed_nodes, [n.child].into_iter().collect());
        // Only failed attempts are recorded as attempts.
        assert!(delta.step_attempted_nodes.is_empty());
        assert!(outcome.step_enabled_defenses.is_empty());

        let state = sim.state().unwrap();
        let a = &state.attackers["attacker"];
        assert!(a.performed_nodes.contains(&n.child));
        assert!(a.action_surface.is_empty());
        assert_eq!(a.iteration, 1);
        assert_eq!(state.attacker_is_terminated("attacker"), Some(true));
        let d = &state.defenders["defender"];
        assert!(d.compromised_nodes.contains(&n.child));
        assert_eq!(d.iteration, 1);
    }

    #[test]
    fn defender_step_enables_defense() {
        let (mut sim, n) = dummy_sim();
        sim.reset(
            &settings(),
            vec![
                ("attacker".to_string(), attacker(&[n.entry])),
                ("defender".to_string(), defender()),
            ],
            1,
        )
        .unwrap();

        let outcome = sim
            .step(&actions(&[("defender", vec![n.defense])]))
            .unwrap();
        assert_eq!(
            outcome.step_enabled_defenses,
            [n.defense].into_iter().collect()
        );
        let state = sim.state().unwrap();
        assert!(state.enabled_defenses.contains(&n.defense));
        assert!(state.defenders["defender"]
            .performed_nodes
            .contains(&n.defense));
        // The attacker took no action this step.
        assert!(outcome.attackers["attacker"]
            .step_performed_nodes
            .is_empty());
    }

    #[test]
    fn step_rejects_unknown_agent() {
        let (mut sim, n) = dummy_sim();
        sim.reset(
            &settings(),
            vec![("attacker".to_string(), attacker(&[n.entry]))],
            1,
        )
        .unwrap();
        let err = sim
            .step(&actions(&[("nobody", vec![]), ("attacker", vec![])]))
            .unwrap_err();
        assert!(matches!(&err, SimulatorError::UnknownAgent(name) if name == "nobody"));
        assert_eq!(err.to_string(), "No agent has name 'nobody'");
    }

    #[test]
    fn step_before_reset_fails() {
        let (mut sim, _) = dummy_sim();
        assert!(sim.state().is_none());
        assert!(matches!(
            sim.step(&HashMap::new()),
            Err(SimulatorError::NotReset)
        ));
    }

    /// The static `Simulator` never runs model effects: compromising
    /// `InfectedDevice:infect`, whose model effect adds a `Wiper` asset
    /// under `DynaSimulator` (see `dyna_simulator.rs`), leaves the graph
    /// exactly as it was.
    #[test]
    fn step_does_not_run_model_effects() {
        let (graph, _model) = wiper_attack_graph();
        let infect = graph.full_name_to_node["InfectedDevice:infect"];
        let pristine: Vec<_> = graph.nodes.keys().collect();
        let mut sim = Simulator::new(Rc::new(RefCell::new(graph)));
        let s = MalSimulatorSettings {
            compromise_entrypoints_at_start: false,
            ..settings()
        };
        sim.reset(
            &s,
            vec![("WiperController".to_string(), attacker(&[infect]))],
            42,
        )
        .unwrap();

        let outcome = sim
            .step(&actions(&[("WiperController", vec![infect])]))
            .unwrap();
        assert!(outcome.attackers["WiperController"]
            .step_performed_nodes
            .contains(&infect));
        assert_eq!(
            sim.graph().borrow().nodes.keys().collect::<Vec<_>>(),
            pristine
        );
    }

    /// Runs reset + two steps on a fresh dummy simulator with two attackers
    /// and a defender, in `PER_STEP_SAMPLE` mode so the RNG is consumed by
    /// every attacker's attempts.
    fn run_dummy_episode(seed: u64) -> (Simulator, Vec<StepOutcome>) {
        let (mut sim, n) = dummy_sim();
        let s = MalSimulatorSettings {
            ttc_mode: TtcMode::PerStepSample,
            ..MalSimulatorSettings::default()
        };
        sim.reset(
            &s,
            vec![
                ("attacker_b".to_string(), attacker(&[n.entry])),
                ("attacker_a".to_string(), attacker(&[n.entry])),
                ("defender".to_string(), defender()),
            ],
            seed,
        )
        .unwrap();
        let mut outcomes = Vec::new();
        for _ in 0..2 {
            outcomes.push(
                sim.step(&actions(&[
                    ("attacker_a", vec![n.child]),
                    ("attacker_b", vec![n.child]),
                ]))
                .unwrap(),
            );
        }
        (sim, outcomes)
    }

    #[test]
    fn same_seed_gives_identical_results() {
        let (sim1, outcomes1) = run_dummy_episode(7);
        let (sim2, outcomes2) = run_dummy_episode(7);
        let (state1, state2) = (sim1.state().unwrap(), sim2.state().unwrap());

        assert_eq!(state1.graph_state, state2.graph_state);
        assert_eq!(state1.enabled_defenses, state2.enabled_defenses);
        assert_eq!(state1.attackers, state2.attackers);
        assert_eq!(state1.defenders, state2.defenders);
        assert_eq!(outcomes1.len(), outcomes2.len());
        for (o1, o2) in outcomes1.iter().zip(&outcomes2) {
            assert_eq!(o1.attackers, o2.attackers);
            assert_eq!(o1.defenders, o2.defenders);
            assert_eq!(o1.step_enabled_defenses, o2.step_enabled_defenses);
        }
        assert_eq!(
            state1.attackers.keys().collect::<Vec<_>>(),
            ["attacker_a", "attacker_b"]
        );
    }
}
