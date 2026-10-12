//! Read-only per-agent views handed to policies and reward functions - the
//! Rust counterpart of `agent_state.py`/`attacker_state.py`/
//! `defender_state.py`.
//!
//! Python's `AttackerState`/`DefenderState` are owned dataclasses rebuilt
//! every step. Here they are borrowed views over what the `Simulator`
//! already holds (`SimState`, the agent's runtime struct and the last
//! step's `StepOutcome`) plus the agent's scenario settings, so building
//! one copies nothing but the defender's `step_compromised_nodes`. See
//! `PORTING_NOTES.md` §11.

use std::collections::HashSet;
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_model::Model;

use crate::graph_state::TtcMode;
use crate::scenario::agent_settings::{AgentSettings, AttackerSettings, DefenderSettings};
use crate::simulator::{AttackerRuntime, DefenderRuntime, SimState, StepOutcome};

#[derive(Debug, Clone, PartialEq)]
pub enum AgentStateError {
    /// The agent is not registered in the simulator state (or has no entry
    /// in the step outcome).
    UnknownAgent(String),
    /// Mirrors `node_ttc_value`'s assert on the TTC mode.
    TtcValuesUnavailable(TtcMode),
    /// Mirrors `node_ttc_value`'s assert that the node has a TTC value.
    MissingTtcValue(String),
}

impl fmt::Display for AgentStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentStateError::UnknownAgent(name) => {
                write!(f, "agent '{name}' is not part of the simulator state")
            }
            AgentStateError::TtcValuesUnavailable(mode) => write!(
                f,
                "TTC value only when TTCMode is PRE_SAMPLE or EXPECTED_VALUE, not {}",
                mode.name()
            ),
            AgentStateError::MissingTtcValue(node) => {
                write!(f, "Node {node} does not have a ttc value")
            }
        }
    }
}

impl std::error::Error for AgentStateError {}

/// Port of `AttackerState`, as a borrowed view.
#[derive(Clone)]
pub struct AttackerState<'a> {
    pub name: &'a str,
    pub attack_graph: &'a AttackGraph,
    pub model: &'a Model,
    pub sim_state: &'a SimState,
    pub settings: &'a AttackerSettings<AttackGraphNodeId>,
    /// The accumulated fields: `performed_nodes`, `action_surface`,
    /// `num_attempts`, `goals`, ...
    pub runtime: &'a AttackerRuntime,
    pub step_performed_nodes: &'a HashSet<AttackGraphNodeId>,
    pub step_attempted_nodes: &'a HashSet<AttackGraphNodeId>,
}

impl AttackerState<'_> {
    /// Port of `state_query.py::node_ttc_value`: the attacker's own
    /// (override) TTC value if it has one, else the graph-wide one.
    pub fn node_ttc_value(&self, node: AttackGraphNodeId) -> Result<f64, AgentStateError> {
        let ttc_mode = self.sim_state.settings.ttc_mode;
        if !matches!(ttc_mode, TtcMode::PreSample | TtcMode::ExpectedValue) {
            return Err(AgentStateError::TtcValuesUnavailable(ttc_mode));
        }
        self.runtime
            .ttc_values
            .get(&node)
            .or_else(|| self.sim_state.graph_state.ttc_values.get(&node))
            .copied()
            .ok_or_else(|| {
                AgentStateError::MissingTtcValue(
                    self.attack_graph.full_name_of(node, Some(self.model)),
                )
            })
    }
}

/// Port of `DefenderState`, as a borrowed view.
#[derive(Clone)]
pub struct DefenderState<'a> {
    pub name: &'a str,
    pub attack_graph: &'a AttackGraph,
    pub model: &'a Model,
    pub sim_state: &'a SimState,
    pub settings: &'a DefenderSettings,
    /// The accumulated fields: `performed_nodes`, `compromised_nodes`,
    /// `observed_nodes`, `action_surface`, ...
    pub runtime: &'a DefenderRuntime,
    /// Defenses enabled this step (by any defender, as on the Python side).
    pub step_performed_nodes: &'a HashSet<AttackGraphNodeId>,
    /// Nodes any attacker compromised this step.
    pub step_compromised_nodes: HashSet<AttackGraphNodeId>,
    pub step_observed_nodes: &'a HashSet<AttackGraphNodeId>,
}

/// `AttackerState | DefenderState`.
#[derive(Clone)]
pub enum AgentState<'a> {
    Attacker(AttackerState<'a>),
    Defender(DefenderState<'a>),
}

impl<'a> AgentState<'a> {
    /// The state of the agent `settings` describes, after the reset or
    /// step that produced `outcome` (`StepOutcome::from_reset` for a
    /// reset).
    pub fn new(
        settings: &'a AgentSettings<AttackGraphNodeId>,
        attack_graph: &'a AttackGraph,
        model: &'a Model,
        sim_state: &'a SimState,
        outcome: &'a StepOutcome,
    ) -> Result<AgentState<'a>, AgentStateError> {
        let unknown = || AgentStateError::UnknownAgent(settings.name().to_string());
        match settings {
            AgentSettings::Attacker(settings) => {
                let runtime = sim_state
                    .attackers
                    .get(&settings.name)
                    .ok_or_else(unknown)?;
                let step = outcome.attackers.get(&settings.name).ok_or_else(unknown)?;
                Ok(AgentState::Attacker(AttackerState {
                    name: &settings.name,
                    attack_graph,
                    model,
                    sim_state,
                    settings,
                    runtime,
                    step_performed_nodes: &step.step_performed_nodes,
                    step_attempted_nodes: &step.step_attempted_nodes,
                }))
            }
            AgentSettings::Defender(settings) => {
                let runtime = sim_state
                    .defenders
                    .get(&settings.name)
                    .ok_or_else(unknown)?;
                let step = outcome.defenders.get(&settings.name).ok_or_else(unknown)?;
                Ok(AgentState::Defender(DefenderState {
                    name: &settings.name,
                    attack_graph,
                    model,
                    sim_state,
                    settings,
                    runtime,
                    step_performed_nodes: &outcome.step_enabled_defenses,
                    step_compromised_nodes: outcome
                        .attackers
                        .values()
                        .flat_map(|a| a.step_performed_nodes.iter().copied())
                        .collect(),
                    step_observed_nodes: &step.step_observed_nodes,
                }))
            }
        }
    }

    pub fn name(&self) -> &'a str {
        match self {
            AgentState::Attacker(a) => a.name,
            AgentState::Defender(d) => d.name,
        }
    }

    pub fn attack_graph(&self) -> &'a AttackGraph {
        match self {
            AgentState::Attacker(a) => a.attack_graph,
            AgentState::Defender(d) => d.attack_graph,
        }
    }

    pub fn model(&self) -> &'a Model {
        match self {
            AgentState::Attacker(a) => a.model,
            AgentState::Defender(d) => d.model,
        }
    }

    pub fn action_surface(&self) -> &'a HashSet<AttackGraphNodeId> {
        match self {
            AgentState::Attacker(a) => &a.runtime.action_surface,
            AgentState::Defender(d) => &d.runtime.action_surface,
        }
    }

    /// `action_surface` ordered by node id (`sorted(.., key=lambda n:
    /// n.id)`), the deterministic order policies iterate it in.
    pub fn sorted_action_surface(&self) -> Vec<AttackGraphNodeId> {
        sorted_by_id(self.attack_graph(), self.action_surface().iter().copied())
    }
}

/// `nodes` ordered by `AttackGraphNode.id`. Nodes no longer in the graph
/// are dropped.
pub fn sorted_by_id(
    attack_graph: &AttackGraph,
    nodes: impl IntoIterator<Item = AttackGraphNodeId>,
) -> Vec<AttackGraphNodeId> {
    let mut nodes: Vec<(i64, AttackGraphNodeId)> = nodes
        .into_iter()
        .filter_map(|key| attack_graph.nodes.get(key).map(|node| (node.id, key)))
        .collect();
    nodes.sort_unstable_by_key(|&(id, _)| id);
    nodes.into_iter().map(|(_, key)| key).collect()
}
