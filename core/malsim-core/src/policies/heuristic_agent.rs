//! Port of `defenders/heuristic_agent.py`.

use std::collections::HashSet;

use maltoolbox_attackgraph::AttackGraphNodeId;
use serde_json::{Map, Value};

use crate::agent_state::{AgentState, DefenderState};
use crate::policies::{DecisionAgent, PolicyError};
use crate::rewards::node_reward;

/// The loop both defenders share: the cheapest defense on the action
/// surface (ties: lowest node id) whose node satisfies `worth_enabling`,
/// given the nodes observed as compromised so far.
fn cheapest_defense(
    policy: &'static str,
    compromised_nodes: &mut HashSet<AttackGraphNodeId>,
    agent_state: &AgentState<'_>,
    worth_enabling: impl Fn(&DefenderState<'_>, AttackGraphNodeId, &HashSet<AttackGraphNodeId>) -> bool,
) -> Result<Option<AttackGraphNodeId>, PolicyError> {
    let AgentState::Defender(defender_state) = agent_state else {
        return Err(PolicyError::WrongAgentType {
            policy,
            expected: "defender",
        });
    };
    compromised_nodes.extend(defender_state.step_observed_nodes.iter().copied());

    let mut selected_node_cost = f64::INFINITY;
    let mut selected_node = None;

    // To make it deterministic
    for node in agent_state.sorted_action_surface() {
        if compromised_nodes.contains(&node) {
            continue;
        }

        let node_cost = node_reward(
            &defender_state.attack_graph.nodes[node],
            defender_state.settings.rewards.as_ref(),
            defender_state.model,
        );

        if node_cost < selected_node_cost && worth_enabling(defender_state, node, compromised_nodes)
        {
            selected_node = Some(node);
            selected_node_cost = node_cost;
        }
    }

    Ok(selected_node)
}

/// A defender that defends compromised assets using notPresent.
///
/// Strategy: enable the cheapest defense node that has compromised child
/// nodes.
#[derive(Debug, Clone, Default)]
pub struct DefendCompromisedDefender {
    compromised_nodes: HashSet<AttackGraphNodeId>,
}

impl DefendCompromisedDefender {
    /// `agent_config`'s `seed`/`randomize` are accepted but, as in Python,
    /// not used.
    pub fn new(_agent_config: &Map<String, Value>) -> Self {
        Self::default()
    }
}

impl DecisionAgent for DefendCompromisedDefender {
    /// Return an action that disables a compromised node.
    fn get_next_action(
        &mut self,
        agent_state: &AgentState<'_>,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        cheapest_defense(
            "DefendCompromisedDefender",
            &mut self.compromised_nodes,
            agent_state,
            |defender_state, node, compromised_nodes| {
                defender_state.attack_graph.nodes[node]
                    .children
                    .iter()
                    .any(|child_node| compromised_nodes.contains(child_node))
            },
        )
    }
}

/// A defender that defends assets about to be compromised.
///
/// Strategy: enable the cheapest defense node that has a non compromised
/// child that has a compromised parent.
#[derive(Debug, Clone, Default)]
pub struct DefendFutureCompromisedDefender {
    compromised_nodes: HashSet<AttackGraphNodeId>,
}

impl DefendFutureCompromisedDefender {
    /// `agent_config`'s `seed`/`randomize` are accepted but, as in Python,
    /// not used.
    pub fn new(_agent_config: &Map<String, Value>) -> Self {
        Self::default()
    }
}

impl DecisionAgent for DefendFutureCompromisedDefender {
    /// Return an action that disables a node that could be compromised
    /// next.
    fn get_next_action(
        &mut self,
        agent_state: &AgentState<'_>,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        cheapest_defense(
            "DefendFutureCompromisedDefender",
            &mut self.compromised_nodes,
            agent_state,
            |defender_state, node, compromised_nodes| {
                let graph = defender_state.attack_graph;
                graph.nodes[node].children.iter().any(|&child_node| {
                    !compromised_nodes.contains(&child_node)
                        && graph.nodes.get(child_node).is_some_and(|child| {
                            child.parents.iter().any(|p| compromised_nodes.contains(p))
                        })
                })
            },
        )
    }
}
