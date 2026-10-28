//! Port of `attackers/shortest_path.py`.

use maltoolbox_attackgraph::AttackGraphNodeId;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use serde_json::{Map, Value};

use crate::agent_state::AgentState;
use crate::policies::path_finding::get_shortest_paths_for_attacker;
use crate::policies::{rng_from_config, DecisionAgent, PolicyError};

/// An agent that finds shortest path in respect to TTC.
/// Requires that the agent has goal set in agent state,
/// and that TTCMode is PRE_SAMPLE or EXPECTED_VALUE.
///
/// Note: Experimental, not proven correct
#[derive(Debug)]
pub struct ShortestPathAttacker {
    rng: StdRng,
}

impl ShortestPathAttacker {
    pub fn new(agent_config: &Map<String, Value>) -> Result<Self, PolicyError> {
        println!("Note: ShortestPathAttacker is in experimental mode");
        Ok(ShortestPathAttacker {
            rng: rng_from_config(agent_config)?,
        })
    }
}

impl DecisionAgent for ShortestPathAttacker {
    /// The first step of the shortest path to a randomly chosen goal.
    fn get_next_action(
        &mut self,
        agent_state: &AgentState<'_>,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        let AgentState::Attacker(attacker_state) = agent_state else {
            return Err(PolicyError::WrongAgentType {
                policy: "ShortestPathAttacker",
                expected: "attacker",
            });
        };

        let shortest_paths_per_goal = get_shortest_paths_for_attacker(attacker_state)?;
        let shortest_path_random_goal = shortest_paths_per_goal
            .choose(&mut self.rng)
            .map(|(_goal, path)| path);
        Ok(shortest_path_random_goal.and_then(|path| path.first().copied()))
    }
}
