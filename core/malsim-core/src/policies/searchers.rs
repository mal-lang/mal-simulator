//! Port of `attackers/searchers.py`: breadth-first and depth-first agents.

use std::collections::{HashSet, VecDeque};

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde_json::{Map, Value};

use crate::agent_state::{sorted_by_id, AgentState};
use crate::policies::{rng_from_config, DecisionAgent, PolicyError};

/// How new targets are ordered before they are queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActionOrdering {
    /// Randomize the order of new targets at each step.
    Random,
    /// Sort by node id.
    Sorted,
    /// Python relies on the action surface set's built-in order here. A
    /// Rust `HashSet` has no stable one, so this sorts by node id.
    #[default]
    Nothing,
    /// Python shuffles without sorting first. Without a stable set order
    /// that is no different from `Random`, which is what this does.
    ListRandom,
}

impl ActionOrdering {
    /// Parses the Python enum value (`ActionOrdering(value)`).
    pub fn from_value(value: &str) -> Option<ActionOrdering> {
        match value {
            "random" => Some(ActionOrdering::Random),
            "sorted" => Some(ActionOrdering::Sorted),
            "nothing" => Some(ActionOrdering::Nothing),
            "list_random" => Some(ActionOrdering::ListRandom),
            _ => None,
        }
    }
}

/// Port of `parse_agent_config`'s `action_ordering` half (the seed is
/// handled by `rng_from_config`).
fn parse_action_ordering(agent_config: &Map<String, Value>) -> Result<ActionOrdering, PolicyError> {
    match agent_config.get("action_ordering") {
        None => Ok(ActionOrdering::Nothing),
        Some(value) => value
            .as_str()
            .and_then(ActionOrdering::from_value)
            .ok_or_else(|| PolicyError::InvalidConfig {
                field: "action_ordering",
                message: format!(
                    "Invalid action_ordering: {value}. Valid options are: \
                     [\"random\", \"sorted\", \"nothing\", \"list_random\"]"
                ),
            }),
    }
}

/// Where `extend_method` puts new targets: the two searchers differ only
/// in this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchOrder {
    BreadthFirst,
    DepthFirst,
}

/// The state both searchers share (Python: `DepthFirstAttacker` subclasses
/// `BreadthFirstAttacker`).
#[derive(Debug)]
struct Searcher {
    search_order: SearchOrder,
    action_ordering: ActionOrdering,
    rng: StdRng,
    targets: VecDeque<AttackGraphNodeId>,
    current_target: Option<AttackGraphNodeId>,
    /// `_prev_state.action_surface`; `None` before the first call.
    prev_action_surface: Option<HashSet<AttackGraphNodeId>>,
}

impl Searcher {
    fn new(
        search_order: SearchOrder,
        agent_config: &Map<String, Value>,
    ) -> Result<Self, PolicyError> {
        Ok(Searcher {
            search_order,
            action_ordering: parse_action_ordering(agent_config)?,
            rng: rng_from_config(agent_config)?,
            targets: VecDeque::new(),
            current_target: None,
            prev_action_surface: None,
        })
    }

    /// Port of `order_func`.
    fn order(
        &mut self,
        attack_graph: &AttackGraph,
        nodes: impl IntoIterator<Item = AttackGraphNodeId>,
    ) -> Vec<AttackGraphNodeId> {
        // sort to ensure deterministic order before shuffling
        let mut nodes = sorted_by_id(attack_graph, nodes);
        match self.action_ordering {
            ActionOrdering::Random | ActionOrdering::ListRandom => nodes.shuffle(&mut self.rng),
            ActionOrdering::Sorted | ActionOrdering::Nothing => {}
        }
        nodes
    }

    /// Port of `extend_method`: extend the deque of targets with new nodes
    /// according to the policy of the agent (BFS or DFS).
    fn extend_method(&mut self, new_nodes: Vec<AttackGraphNodeId>) {
        match self.search_order {
            // `targets.extendleft(new_nodes)`
            SearchOrder::BreadthFirst => {
                for node in new_nodes {
                    self.targets.push_front(node);
                }
            }
            // `targets.extend(new_nodes)`
            SearchOrder::DepthFirst => self.targets.extend(new_nodes),
        }
    }

    /// Port of `_update_targets`.
    fn update_targets(
        &mut self,
        new_targets: Vec<AttackGraphNodeId>,
        disabled_nodes: &HashSet<AttackGraphNodeId>,
    ) {
        if let Some(current_target) = self.current_target {
            if !disabled_nodes.contains(&current_target) {
                // If the current target was not compromised, e.g. due to
                // TTCs, it remains in action surface and should be added as
                // a target.
                self.targets.push_back(current_target);
            }
        }

        // Enabled defenses may remove previously possible attack steps.
        if !disabled_nodes.is_empty() {
            self.targets.retain(|n| !disabled_nodes.contains(n));
        }

        // Use the extend method to add new targets
        self.extend_method(new_targets);
    }

    /// Receive the next action according to agent policy (bfs/dfs).
    fn get_next_action(&mut self, agent_state: &AgentState<'_>) -> Option<AttackGraphNodeId> {
        let action_surface = agent_state.action_surface();
        let (new_nodes, disabled_nodes): (Vec<_>, HashSet<_>) = match &self.prev_action_surface {
            Some(prev) => (
                action_surface.difference(prev).copied().collect(),
                prev.difference(action_surface).copied().collect(),
            ),
            None => (action_surface.iter().copied().collect(), HashSet::new()),
        };

        let new_targets = self.order(agent_state.attack_graph(), new_nodes);
        self.update_targets(new_targets, &disabled_nodes);

        // `_select_next_target`
        self.current_target = self.targets.pop_back();
        self.prev_action_surface = Some(action_surface.clone());
        self.current_target
    }
}

/// A Breadth-First agent, with possible randomization at each level.
#[derive(Debug)]
pub struct BreadthFirstAttacker(Searcher);

impl BreadthFirstAttacker {
    /// A human-friendly name for the agent.
    pub const NAME: &'static str = "Breadth First Attacker";

    pub fn new(agent_config: &Map<String, Value>) -> Result<Self, PolicyError> {
        Searcher::new(SearchOrder::BreadthFirst, agent_config).map(BreadthFirstAttacker)
    }
}

impl DecisionAgent for BreadthFirstAttacker {
    fn get_next_action(
        &mut self,
        agent_state: &AgentState<'_>,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        Ok(self.0.get_next_action(agent_state))
    }
}

/// A Depth-First agent, with possible randomization at each level.
#[derive(Debug)]
pub struct DepthFirstAttacker(Searcher);

impl DepthFirstAttacker {
    /// A human-friendly name for the agent.
    pub const NAME: &'static str = "Depth First Attacker";

    pub fn new(agent_config: &Map<String, Value>) -> Result<Self, PolicyError> {
        Searcher::new(SearchOrder::DepthFirst, agent_config).map(DepthFirstAttacker)
    }
}

impl DecisionAgent for DepthFirstAttacker {
    fn get_next_action(
        &mut self,
        agent_state: &AgentState<'_>,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        Ok(self.0.get_next_action(agent_state))
    }
}
