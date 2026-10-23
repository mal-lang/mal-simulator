//! Port of `rewards.py` (plus `graph_utils.py::node_reward`): per-step
//! rewards for attackers and defenders.
//!
//! Python builds one reward closure per agent at simulator construction
//! (`attacker_step_reward_fn`/`defender_step_reward_fn`) and
//! `MalSimulator.agent_reward(state)` calls it. Here the agent state view
//! already carries the settings the closure captured, so these are plain
//! functions of the state.

use crate::types::NodeSet;
use std::fmt;

use maltoolbox_attackgraph::AttackGraphNode;
use maltoolbox_model::Model;
use rand::Rng;

use crate::agent_state::{sorted_by_id, AgentState, AttackerState, DefenderState};
use crate::graph_state::TtcMode;
use crate::scenario::node_property_rule::NodePropertyRule;
use crate::settings::RewardMode;
use crate::ttc::{TtcDist, TtcDistError};

#[derive(Debug, Clone, PartialEq)]
pub enum RewardError {
    /// Python's `ENABLED_DEFENSES_FUNCS[reward_mode]` `KeyError`: defenders
    /// only support `CUMULATIVE` and `ONE_OFF`.
    UnsupportedDefenderRewardMode(RewardMode),
    TtcDist(TtcDistError),
}

impl fmt::Display for RewardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RewardError::UnsupportedDefenderRewardMode(mode) => write!(
                f,
                "reward mode {} is not supported for defenders",
                mode.name()
            ),
            RewardError::TtcDist(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RewardError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RewardError::TtcDist(e) => Some(e),
            RewardError::UnsupportedDefenderRewardMode(_) => None,
        }
    }
}

impl From<TtcDistError> for RewardError {
    fn from(e: TtcDistError) -> Self {
        RewardError::TtcDist(e)
    }
}

/// Port of `node_reward`: the node's reward from the agent's rule, `0.0`
/// without a (non-empty) rule or a matching entry.
pub fn node_reward(
    node: &AttackGraphNode,
    reward_rule: Option<&NodePropertyRule<f64>>,
    model: &Model,
) -> f64 {
    match reward_rule {
        Some(rule) if !rule.is_empty() => rule.value(node, model).unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Port of `defender_step_reward`: the defender is penalized for enabled
/// defenses and compromised steps - all of them for `CUMULATIVE`, this
/// step's for `ONE_OFF`.
pub fn defender_step_reward(defender_state: &DefenderState<'_>) -> Result<f64, RewardError> {
    let runtime = defender_state.runtime;
    let (enabled_defenses, compromised_nodes) = match defender_state.settings.reward_mode {
        RewardMode::Cumulative => (&runtime.performed_nodes, &runtime.compromised_nodes),
        RewardMode::OneOff => (
            defender_state.step_performed_nodes,
            &defender_state.step_compromised_nodes,
        ),
        mode => return Err(RewardError::UnsupportedDefenderRewardMode(mode)),
    };

    let nodes: NodeSet = enabled_defenses.union(compromised_nodes).copied().collect();
    let graph = defender_state.attack_graph;
    let step_reward: f64 = sorted_by_id(graph, nodes)
        .into_iter()
        .map(|n| {
            node_reward(
                &graph.nodes[n],
                defender_state.settings.rewards.as_ref(),
                defender_state.model,
            )
        })
        .sum();
    Ok(-step_reward)
}

/// Port of `attacker_step_reward`: the attacker is rewarded for compromised
/// nodes (all of them for `CUMULATIVE`, otherwise this step's) and
/// penalized per attempt when TTCs are enabled, or by the node's TTC for
/// the `EXPECTED_TTC`/`SAMPLE_TTC` reward modes when they are disabled.
pub fn attacker_step_reward(
    attacker_state: &AttackerState<'_>,
    rng: &mut impl Rng,
) -> Result<f64, RewardError> {
    let reward_mode = attacker_state.settings.reward_mode;
    let performed_steps = match reward_mode {
        RewardMode::Cumulative => &attacker_state.runtime.performed_nodes,
        RewardMode::OneOff | RewardMode::ExpectedTtc | RewardMode::SampleTtc => {
            attacker_state.step_performed_nodes
        }
    };
    let graph = attacker_state.attack_graph;
    // Node-id order, so the float sum and the `SAMPLE_TTC` draws don't
    // depend on `HashSet` order.
    let performed_steps = sorted_by_id(graph, performed_steps.iter().copied());

    let mut step_reward: f64 = performed_steps
        .iter()
        .map(|&n| {
            node_reward(
                &graph.nodes[n],
                attacker_state.settings.rewards.as_ref(),
                attacker_state.model,
            )
        })
        .sum();

    if attacker_state.sim_state.settings.ttc_mode != TtcMode::Disabled {
        // If TTC Mode is not disabled, attacker is penalized for each attempt
        step_reward -= attacker_state.step_attempted_nodes.len() as f64;
    } else if matches!(reward_mode, RewardMode::ExpectedTtc | RewardMode::SampleTtc) {
        // If TTC Mode is disabled but reward mode uses TTCs, penalize with TTCs
        for &n in &performed_steps {
            // `TTCDist.from_node(node) if node.ttc else 0`
            let Some(ttc) = graph.nodes[n].ttc.as_ref().filter(|ttc| is_truthy(ttc)) else {
                continue;
            };
            let ttc_dist = TtcDist::from_dict(ttc)?;
            step_reward -= match reward_mode {
                RewardMode::ExpectedTtc => ttc_dist.expected_value(),
                _ => ttc_dist.sample_value(rng),
            };
        }
    }

    Ok(step_reward)
}

/// Port of `MalSimulator.agent_reward`.
pub fn agent_reward(agent_state: &AgentState<'_>, rng: &mut impl Rng) -> Result<f64, RewardError> {
    match agent_state {
        AgentState::Attacker(attacker_state) => attacker_step_reward(attacker_state, rng),
        AgentState::Defender(defender_state) => defender_step_reward(defender_state),
    }
}

/// Python truthiness of a node's `ttc` value (`None` and `{}` are falsy).
fn is_truthy(ttc: &serde_json::Value) -> bool {
    match ttc {
        serde_json::Value::Null => false,
        serde_json::Value::Object(map) => !map.is_empty(),
        _ => true,
    }
}
