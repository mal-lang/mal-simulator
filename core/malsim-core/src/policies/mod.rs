//! Port of `python/malsim/policies`: the `DecisionAgent` interface and the
//! built-in heuristic agents a scenario's `policy` field can name.
//!
//! Wherever a Python policy iterates an action surface (a `set`), the Rust
//! port iterates it in node-id order: `HashSet` order is random per
//! process, so anything else would make seeded runs irreproducible. See
//! `PORTING_NOTES.md` §12.

use std::fmt;

use maltoolbox_attackgraph::AttackGraphNodeId;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{Map, Value};

use crate::agent_state::{AgentState, AgentStateError};

pub mod heuristic_agent;
pub mod keyboard_input;
pub mod passive_agent;
pub mod path_finding;
pub mod random_agent;
pub mod searchers;
pub mod shortest_path;
pub mod ttc_soft_min;

pub use heuristic_agent::{DefendCompromisedDefender, DefendFutureCompromisedDefender};
pub use keyboard_input::KeyboardAgent;
pub use passive_agent::PassiveAgent;
pub use path_finding::get_shortest_path_to;
pub use random_agent::RandomAgent;
pub use searchers::{BreadthFirstAttacker, DepthFirstAttacker};
pub use shortest_path::ShortestPathAttacker;
pub use ttc_soft_min::TTCSoftMinAttacker;

#[derive(Debug)]
pub enum PolicyError {
    /// Python's `LookupError` for a name outside `policy_name_to_class`.
    UnknownPolicy(String),
    /// A value in the agent's `config` has the wrong type or is not one of
    /// the allowed values.
    InvalidConfig {
        field: &'static str,
        message: String,
    },
    /// An attacker-only policy was handed a defender state, or the reverse.
    WrongAgentType {
        policy: &'static str,
        expected: &'static str,
    },
    /// `ShortestPathAttacker`'s assert that the attacker has goals.
    NoGoals,
    /// `TTCSoftMinAttacker` could not turn its TTCs into sampling weights.
    InvalidWeights(String),
    AgentState(AgentStateError),
    /// `KeyboardAgent` could not read from or write to its terminal.
    Io(std::io::Error),
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyError::UnknownPolicy(name) => write!(
                f,
                "Policy class '{name}' not supported. Must be one of: {POLICY_NAMES:?}"
            ),
            PolicyError::InvalidConfig { field, message } => {
                write!(f, "invalid agent config '{field}': {message}")
            }
            PolicyError::WrongAgentType { policy, expected } => {
                write!(f, "{policy} can only be used by {expected} agents")
            }
            PolicyError::NoGoals => {
                write!(f, "Attacker needs goal set for shortest path calculation")
            }
            PolicyError::InvalidWeights(message) => {
                write!(f, "could not sample from TTC weights: {message}")
            }
            PolicyError::AgentState(e) => write!(f, "{e}"),
            PolicyError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PolicyError::AgentState(e) => Some(e),
            PolicyError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<AgentStateError> for PolicyError {
    fn from(e: AgentStateError) -> Self {
        PolicyError::AgentState(e)
    }
}

impl From<std::io::Error> for PolicyError {
    fn from(e: std::io::Error) -> Self {
        PolicyError::Io(e)
    }
}

/// Port of `DecisionAgent`: a heuristic agent.
pub trait DecisionAgent {
    /// Select the next action the agent will work with: the selected node,
    /// or `None` if there are no actions to select from (or the agent
    /// chooses to wait).
    fn get_next_action(
        &mut self,
        agent_state: &AgentState<'_>,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError>;
}

/// The keys of Python's `policy_name_to_class`.
pub const POLICY_NAMES: [&str; 9] = [
    "KeyboardAgent",
    "PassiveAgent",
    "DepthFirstAttacker",
    "BreadthFirstAttacker",
    "TTCSoftMinAttacker",
    "ShortestPathAttacker",
    "DefendCompromisedDefender",
    "DefendFutureCompromisedDefender",
    "RandomAgent",
];

/// Port of `policy_name_to_class[policy_name](agent_config)`: instantiates
/// the built-in policy a scenario's `policy` field names, with the agent's
/// `config`.
pub fn policy_from_name(
    policy_name: &str,
    agent_config: &Map<String, Value>,
) -> Result<Box<dyn DecisionAgent>, PolicyError> {
    Ok(match policy_name {
        "KeyboardAgent" => Box::new(KeyboardAgent::new()),
        "PassiveAgent" => Box::new(PassiveAgent),
        "DepthFirstAttacker" => Box::new(DepthFirstAttacker::new(agent_config)?),
        "BreadthFirstAttacker" => Box::new(BreadthFirstAttacker::new(agent_config)?),
        "TTCSoftMinAttacker" => Box::new(TTCSoftMinAttacker::new(agent_config)?),
        "ShortestPathAttacker" => Box::new(ShortestPathAttacker::new(agent_config)?),
        "DefendCompromisedDefender" => Box::new(DefendCompromisedDefender::new(agent_config)),
        "DefendFutureCompromisedDefender" => {
            Box::new(DefendFutureCompromisedDefender::new(agent_config))
        }
        "RandomAgent" => Box::new(RandomAgent::new(agent_config)?),
        other => return Err(PolicyError::UnknownPolicy(other.to_string())),
    })
}

/// `random.Random(agent_config.get('seed'))`: seeded when `seed` is an
/// integer, from OS entropy when it is absent or `null`.
pub(crate) fn rng_from_config(agent_config: &Map<String, Value>) -> Result<StdRng, PolicyError> {
    match agent_config.get("seed") {
        None | Some(Value::Null) => Ok(StdRng::from_rng(&mut rand::rng())),
        Some(seed) => match seed.as_i64() {
            // Negative seeds are legal in Python; reinterpret the bits.
            Some(seed) => Ok(StdRng::seed_from_u64(seed as u64)),
            None => Err(PolicyError::InvalidConfig {
                field: "seed",
                message: format!("Seed must be an integer or None, got {seed}"),
            }),
        },
    }
}

/// `agent_config.get(field, default)` for a float setting.
pub(crate) fn f64_from_config(
    agent_config: &Map<String, Value>,
    field: &'static str,
    default: f64,
) -> Result<f64, PolicyError> {
    match agent_config.get(field) {
        None => Ok(default),
        Some(value) => value.as_f64().ok_or_else(|| PolicyError::InvalidConfig {
            field,
            message: format!("expected a number, got {value}"),
        }),
    }
}

#[cfg(test)]
mod tests;
