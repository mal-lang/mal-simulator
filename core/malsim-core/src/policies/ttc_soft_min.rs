//! Port of `attackers/ttc_soft_min.py`: use softargmax to prioritize low
//! TTC attack steps when running attacker.

use maltoolbox_attackgraph::AttackGraphNodeId;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::StdRng;
use serde_json::{Map, Value};

use crate::agent_state::AgentState;
use crate::policies::{f64_from_config, rng_from_config, DecisionAgent, PolicyError};

/// An agent that samples its next step with a probability that falls off
/// with the TTC the step has left.
#[derive(Debug)]
pub struct TTCSoftMinAttacker {
    rng: StdRng,
    /// Sharpness of the softargmax.
    beta: f64,
}

impl TTCSoftMinAttacker {
    pub fn new(agent_config: &Map<String, Value>) -> Result<Self, PolicyError> {
        Ok(TTCSoftMinAttacker {
            rng: rng_from_config(agent_config)?,
            beta: f64_from_config(agent_config, "beta", 1.0)?,
        })
    }
}

impl DecisionAgent for TTCSoftMinAttacker {
    /// Sample node from the action surface based on ttc softargmax.
    fn get_next_action(
        &mut self,
        agent_state: &AgentState<'_>,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        let AgentState::Attacker(attacker_state) = agent_state else {
            return Err(PolicyError::WrongAgentType {
                policy: "TTCSoftMinAttacker",
                expected: "attacker",
            });
        };

        let possible_choices = agent_state.sorted_action_surface();
        if possible_choices.is_empty() {
            return Ok(None);
        }

        let weights = possible_choices
            .iter()
            .map(|&n| {
                let num_attempts = attacker_state.runtime.num_attempts.get(&n).copied();
                let ttc_left = attacker_state.node_ttc_value(n)? - num_attempts.unwrap_or(0) as f64;
                Ok((-self.beta * ttc_left).exp())
            })
            .collect::<Result<Vec<f64>, PolicyError>>()?;

        // `WeightedIndex` normalizes the weights itself.
        let index = WeightedIndex::new(&weights)
            .map_err(|e| PolicyError::InvalidWeights(e.to_string()))?
            .sample(&mut self.rng);
        Ok(Some(possible_choices[index]))
    }
}
