//! Port of `passive_agent.py`: a passive agent that always chooses to do
//! nothing.

use maltoolbox_attackgraph::AttackGraphNodeId;

use crate::agent_state::AgentState;
use crate::policies::{DecisionAgent, PolicyError};

#[derive(Debug, Clone, Copy, Default)]
pub struct PassiveAgent;

impl DecisionAgent for PassiveAgent {
    fn get_next_action(
        &mut self,
        _agent_state: &AgentState<'_>,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        // A passive agent never does anything
        Ok(None)
    }
}
