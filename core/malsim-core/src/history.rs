//! Per-episode step history kept inside [`SimState`], so the state at any
//! earlier iteration of the current episode can be rebuilt on demand
//! ([`SimState::snapshot_at`]), plus the per-step `recording` and the
//! dyna path's accumulated modification record.
//!
//! This is what lets the Python `MalSimulator`/`DynaMalSimulator` hold no
//! state of their own: `agent_states`, `sim_state` and `recording` are
//! translated from the native simulator on access, and an
//! `AttackerState`/`DefenderState`'s `previous_state` is rebuilt from this
//! history when it is read (`PORTING_NOTES.md` §14).
//!
//! The history stores, per step, only what that step *added* (and the
//! action-surface diff), so rebuilding iteration `k` is "clone the current
//! state, undo the steps after `k`, newest first". Rebuilding the previous
//! iteration, the common case, therefore costs one undo.

use std::collections::BTreeMap;

use maltoolbox_attackgraph::AttackGraphNodeId;

use crate::event_logger::LogEntry;
use crate::graph_state::GraphState;
use crate::model_effects::ModEffectOp;
use crate::simulator::{AttackerRuntime, DefenderRuntime, SimState};
use crate::types::{NodeMap, NodeSet};

/// What one step changed for one attacker.
#[derive(Debug, Clone, Default)]
pub(crate) struct AttackerStepRecord {
    /// The nodes the step reported as compromised, as reported (the
    /// `step_performed_nodes` delta) - what `performed_nodes_order` keys.
    pub(crate) step_performed_nodes: NodeSet,
    /// `performed_history.len()` before the step; the nodes after it are
    /// the ones this step newly added to `performed_nodes`.
    pub(crate) performed_history_len: usize,
    /// Nodes this step newly added to `attempted_nodes`.
    pub(crate) new_attempted_nodes: NodeSet,
    /// Every node attempted this step (each bumped `num_attempts` by one).
    pub(crate) step_attempted_nodes: NodeSet,
    pub(crate) surface_added: NodeSet,
    pub(crate) surface_removed: NodeSet,
}

/// What one step changed for one defender.
#[derive(Debug, Clone, Default)]
pub(crate) struct DefenderStepRecord {
    /// The defenses the step reported as enabled, as reported (the
    /// `step_performed_nodes` delta) - what `performed_nodes_order` keys.
    pub(crate) step_performed_nodes: NodeSet,
    pub(crate) new_performed_nodes: NodeSet,
    pub(crate) new_compromised_nodes: NodeSet,
    pub(crate) new_observed_nodes: NodeSet,
    /// `logs.len()` before the step.
    pub(crate) logs_len: usize,
    pub(crate) surface_added: NodeSet,
    pub(crate) surface_removed: NodeSet,
}

/// The shared (not per-agent) state before a step, kept only when the
/// step could not be undone from deltas: a dyna step whose model effects
/// folded new nodes into `graph_state`/`enabled_defenses`.
#[derive(Debug, Clone)]
pub(crate) struct PriorSharedState {
    pub(crate) graph_state: GraphState,
    pub(crate) enabled_defenses: NodeSet,
}

/// What one step changed, for undoing it.
#[derive(Debug, Clone, Default)]
pub(crate) struct StepRecord {
    pub(crate) attackers: BTreeMap<String, AttackerStepRecord>,
    pub(crate) defenders: BTreeMap<String, DefenderStepRecord>,
    /// Defenses this step newly added to `enabled_defenses` (unused when
    /// `prior_shared` is set).
    pub(crate) new_enabled_defenses: NodeSet,
    pub(crate) prior_shared: Option<PriorSharedState>,
    /// `modification_record.len()` before the step.
    pub(crate) modification_record_len: usize,
    /// This step's `recording` entry: per agent, the nodes it newly
    /// performed, requested actions first (in request order), then the
    /// rest (effects) in the order they were performed.
    pub(crate) recording: BTreeMap<String, Vec<AttackGraphNodeId>>,
}

/// The step history of the current episode - see module docs.
#[derive(Debug, Clone, Default)]
pub struct EpisodeHistory {
    /// Per agent, the nodes it had performed right after reset (what
    /// `performed_nodes_order[0]` holds).
    reset_performed: BTreeMap<String, NodeSet>,
    steps: Vec<StepRecord>,
    modification_record: Vec<ModEffectOp>,
}

impl EpisodeHistory {
    pub(crate) fn from_reset(
        attackers: &BTreeMap<String, AttackerRuntime>,
        defenders: &BTreeMap<String, DefenderRuntime>,
    ) -> EpisodeHistory {
        let reset_performed = attackers
            .iter()
            .map(|(name, a)| (name.clone(), a.performed_nodes.clone()))
            .chain(
                defenders
                    .iter()
                    .map(|(name, d)| (name.clone(), d.performed_nodes.clone())),
            )
            .collect();
        EpisodeHistory {
            reset_performed,
            steps: Vec::new(),
            modification_record: Vec::new(),
        }
    }

    /// The current iteration: the number of steps since reset.
    pub fn iteration(&self) -> u64 {
        self.steps.len() as u64
    }

    /// Every model modification of this episode, in the order applied.
    /// Always empty for the static `Simulator`.
    pub fn modification_record(&self) -> &[ModEffectOp] {
        &self.modification_record
    }

    /// The recording of step `iteration` (1-based: the step that took the
    /// simulator from `iteration - 1` to `iteration`): per agent, the nodes
    /// it newly performed, requested actions first. `None` for an
    /// iteration that has not happened.
    pub fn recording(&self, iteration: u64) -> Option<&BTreeMap<String, Vec<AttackGraphNodeId>>> {
        let index = usize::try_from(iteration).ok()?.checked_sub(1)?;
        self.steps.get(index).map(|s| &s.recording)
    }

    pub(crate) fn push_step(&mut self, record: StepRecord) {
        self.steps.push(record);
    }

    pub(crate) fn extend_modification_record(&mut self, ops: Vec<ModEffectOp>) {
        self.modification_record.extend(ops);
    }

    /// `performed_nodes_order` of agent `name` at `iteration`: iteration ->
    /// nodes performed then, for every iteration up to `iteration` in which
    /// it performed something (0 is reset).
    fn performed_nodes_order(&self, name: &str, iteration: u64) -> BTreeMap<u64, NodeSet> {
        let mut order = BTreeMap::new();
        if let Some(at_reset) = self.reset_performed.get(name) {
            if !at_reset.is_empty() {
                order.insert(0, at_reset.clone());
            }
        }
        for (index, step) in self.steps.iter().take(iteration as usize).enumerate() {
            let performed = step
                .attackers
                .get(name)
                .map(|a| &a.step_performed_nodes)
                .or_else(|| step.defenders.get(name).map(|d| &d.step_performed_nodes));
            if let Some(performed) = performed {
                if !performed.is_empty() {
                    order.insert(index as u64 + 1, performed.clone());
                }
            }
        }
        order
    }
}

/// The state of one attacker at some iteration - the fields of
/// `AttackerRuntime` an `AttackerState` exposes, plus its
/// `performed_nodes_order`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttackerSnapshot {
    pub entry_points: NodeSet,
    pub goals: NodeSet,
    pub performed_nodes: NodeSet,
    pub attempted_nodes: NodeSet,
    pub action_surface: NodeSet,
    pub num_attempts: NodeMap<u64>,
    pub ttc_values: NodeMap<f64>,
    pub impossible_steps: NodeSet,
    pub performed_nodes_order: BTreeMap<u64, NodeSet>,
    pub iteration: u64,
}

/// The state of one defender at some iteration - see `AttackerSnapshot`.
#[derive(Debug, Clone, PartialEq)]
pub struct DefenderSnapshot {
    pub performed_nodes: NodeSet,
    pub compromised_nodes: NodeSet,
    pub observed_nodes: NodeSet,
    pub action_surface: NodeSet,
    pub logs: Vec<LogEntry>,
    pub performed_nodes_order: BTreeMap<u64, NodeSet>,
    pub iteration: u64,
}

/// The whole simulator state at some iteration of the current episode -
/// what [`SimState::snapshot_at`] rebuilds.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub iteration: u64,
    pub graph_state: GraphState,
    pub enabled_defenses: NodeSet,
    /// The modification record up to this iteration (dyna only).
    pub modification_record: Vec<ModEffectOp>,
    pub attackers: BTreeMap<String, AttackerSnapshot>,
    pub defenders: BTreeMap<String, DefenderSnapshot>,
}

fn undo_surface(surface: &mut NodeSet, added: &NodeSet, removed: &NodeSet) {
    for id in added {
        surface.remove(id);
    }
    surface.extend(removed.iter().copied());
}

/// `(added, removed)` going from `before` to `after`.
pub(crate) fn surface_diff(before: &NodeSet, after: &NodeSet) -> (NodeSet, NodeSet) {
    (
        after.difference(before).copied().collect(),
        before.difference(after).copied().collect(),
    )
}

/// The recording order of `new_nodes`: the `requested` ones first, in
/// request order, then the rest in `rest_order`'s order.
pub(crate) fn recording_order(
    requested: &[AttackGraphNodeId],
    new_nodes: &NodeSet,
    rest_order: impl IntoIterator<Item = AttackGraphNodeId>,
) -> Vec<AttackGraphNodeId> {
    let mut out: Vec<AttackGraphNodeId> = requested
        .iter()
        .copied()
        .filter(|id| new_nodes.contains(id))
        .collect();
    out.extend(
        rest_order
            .into_iter()
            .filter(|id| new_nodes.contains(id) && !requested.contains(id)),
    );
    out
}

impl SimState {
    /// The step history of the current episode.
    pub fn history(&self) -> &EpisodeHistory {
        &self.history
    }

    /// The state at `iteration` (0 is right after reset) of the current
    /// episode, rebuilt from the current state and the step history.
    /// `None` for an iteration after the current one.
    pub fn snapshot_at(&self, iteration: u64) -> Option<StateSnapshot> {
        if iteration > self.history.iteration() {
            return None;
        }
        let mut attackers = self.attackers.clone();
        let mut defenders = self.defenders.clone();
        let mut graph_state = self.graph_state.clone();
        let mut enabled_defenses = self.enabled_defenses.clone();
        let mut modification_record_len = self.history.modification_record.len();

        for step in self.history.steps[iteration as usize..].iter().rev() {
            for (name, record) in &step.attackers {
                let Some(a) = attackers.get_mut(name) else {
                    continue;
                };
                for id in &a.performed_history[record.performed_history_len..] {
                    a.performed_nodes.remove(id);
                }
                a.performed_history.truncate(record.performed_history_len);
                for id in &record.new_attempted_nodes {
                    a.attempted_nodes.remove(id);
                }
                for id in &record.step_attempted_nodes {
                    if let Some(count) = a.num_attempts.get_mut(id) {
                        *count -= 1;
                        if *count == 0 {
                            a.num_attempts.remove(id);
                        }
                    }
                }
                undo_surface(
                    &mut a.action_surface,
                    &record.surface_added,
                    &record.surface_removed,
                );
                a.iteration -= 1;
            }
            for (name, record) in &step.defenders {
                let Some(d) = defenders.get_mut(name) else {
                    continue;
                };
                for id in &record.new_performed_nodes {
                    d.performed_nodes.remove(id);
                }
                for id in &record.new_compromised_nodes {
                    d.compromised_nodes.remove(id);
                }
                for id in &record.new_observed_nodes {
                    d.observed_nodes.remove(id);
                }
                d.logs.truncate(record.logs_len);
                undo_surface(
                    &mut d.action_surface,
                    &record.surface_added,
                    &record.surface_removed,
                );
                d.iteration -= 1;
            }
            match &step.prior_shared {
                Some(prior) => {
                    graph_state = prior.graph_state.clone();
                    enabled_defenses = prior.enabled_defenses.clone();
                }
                None => {
                    for id in &step.new_enabled_defenses {
                        enabled_defenses.remove(id);
                    }
                }
            }
            modification_record_len = step.modification_record_len;
        }

        let attackers = attackers
            .into_iter()
            .map(|(name, a)| {
                let performed_nodes_order = self.history.performed_nodes_order(&name, iteration);
                let snapshot = AttackerSnapshot {
                    entry_points: a.entry_points,
                    goals: a.goals,
                    performed_nodes: a.performed_nodes,
                    attempted_nodes: a.attempted_nodes,
                    action_surface: a.action_surface,
                    num_attempts: a.num_attempts,
                    ttc_values: a.ttc_values,
                    impossible_steps: a.impossible_steps,
                    performed_nodes_order,
                    iteration: a.iteration,
                };
                (name, snapshot)
            })
            .collect();
        let defenders = defenders
            .into_iter()
            .map(|(name, d)| {
                let performed_nodes_order = self.history.performed_nodes_order(&name, iteration);
                let snapshot = DefenderSnapshot {
                    performed_nodes: d.performed_nodes,
                    compromised_nodes: d.compromised_nodes,
                    observed_nodes: d.observed_nodes,
                    action_surface: d.action_surface,
                    logs: d.logs,
                    performed_nodes_order,
                    iteration: d.iteration,
                };
                (name, snapshot)
            })
            .collect();

        Some(StateSnapshot {
            iteration,
            graph_state,
            enabled_defenses,
            modification_record: self.history.modification_record[..modification_record_len]
                .to_vec(),
            attackers,
            defenders,
        })
    }
}
