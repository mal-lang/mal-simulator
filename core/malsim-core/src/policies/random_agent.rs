//! Port of `random_agent.py`.

use maltoolbox_attackgraph::AttackGraphNodeId;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::RngExt;
use serde_json::{Map, Value};

use crate::agent_state::AgentState;
use crate::policies::{f64_from_config, rng_from_config, DecisionAgent, PolicyError};

/// An agent that selects random actions.
#[derive(Debug)]
pub struct RandomAgent {
    rng: StdRng,
    wait_prob: f64,
}

impl RandomAgent {
    pub fn new(agent_config: &Map<String, Value>) -> Result<Self, PolicyError> {
        Ok(RandomAgent {
            rng: rng_from_config(agent_config)?,
            wait_prob: f64_from_config(agent_config, "wait_prob", 0.0)?,
        })
    }
}

impl DecisionAgent for RandomAgent {
    /// Return a random node from the action surface.
    fn get_next_action(
        &mut self,
        agent_state: &AgentState<'_>,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        let possible_choices = agent_state.sorted_action_surface();
        if possible_choices.is_empty() || self.rng.random::<f64>() < self.wait_prob {
            return Ok(None);
        }
        Ok(possible_choices.choose(&mut self.rng).copied())
    }
}
