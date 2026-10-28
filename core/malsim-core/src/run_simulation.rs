//! Port of `run_simulation.py`: the reference simulation loop the CLI
//! uses. Every agent's policy picks one action per step until the
//! simulation is done.
//!
//! Python's `run_simulation(sim)` reads agent settings, policies and
//! rewards off the `MalSimulator`. The Rust `Simulator` holds none of
//! those, so the `Scenario` it was built from is passed alongside it. See
//! `PORTING_NOTES.md` §11.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::rc::Rc;

use maltoolbox_attackgraph::AttackGraphNodeId;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::agent_state::{AgentState, AgentStateError};
use crate::policies::{policy_from_name, DecisionAgent, PolicyError};
use crate::rewards::{agent_reward, RewardError};
use crate::scenario::agent_settings::{AgentSettings, AgentSettingsError};
use crate::scenario::Scenario;
use crate::simulator::{Simulator, SimulatorError, StepOutcome};

#[derive(Debug)]
pub enum RunSimulationError {
    /// The simulator does not run on the scenario's attack graph.
    GraphMismatch,
    AgentSettings(AgentSettingsError),
    AgentState(AgentStateError),
    Policy(PolicyError),
    Reward(RewardError),
    Simulator(SimulatorError),
}

impl fmt::Display for RunSimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunSimulationError::GraphMismatch => write!(
                f,
                "the simulator was not built from the scenario's attack graph"
            ),
            RunSimulationError::AgentSettings(e) => write!(f, "{e}"),
            RunSimulationError::AgentState(e) => write!(f, "{e}"),
            RunSimulationError::Policy(e) => write!(f, "{e}"),
            RunSimulationError::Reward(e) => write!(f, "{e}"),
            RunSimulationError::Simulator(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RunSimulationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RunSimulationError::GraphMismatch => None,
            RunSimulationError::AgentSettings(e) => Some(e),
            RunSimulationError::AgentState(e) => Some(e),
            RunSimulationError::Policy(e) => Some(e),
            RunSimulationError::Reward(e) => Some(e),
            RunSimulationError::Simulator(e) => Some(e),
        }
    }
}

impl From<AgentSettingsError> for RunSimulationError {
    fn from(e: AgentSettingsError) -> Self {
        RunSimulationError::AgentSettings(e)
    }
}

impl From<AgentStateError> for RunSimulationError {
    fn from(e: AgentStateError) -> Self {
        RunSimulationError::AgentState(e)
    }
}

impl From<PolicyError> for RunSimulationError {
    fn from(e: PolicyError) -> Self {
        RunSimulationError::Policy(e)
    }
}

impl From<RewardError> for RunSimulationError {
    fn from(e: RewardError) -> Self {
        RunSimulationError::Reward(e)
    }
}

impl From<SimulatorError> for RunSimulationError {
    fn from(e: SimulatorError) -> Self {
        RunSimulationError::Simulator(e)
    }
}

/// The decision agents of a scenario, by agent name: Python's lazily
/// instantiated `AgentRuntimeMixin.agent`. An agent without a `policy` has
/// no entry.
pub type DecisionAgents = BTreeMap<String, Box<dyn DecisionAgent>>;

/// Instantiates the built-in policy each of the scenario's agents names.
pub fn decision_agents(scenario: &Scenario) -> Result<DecisionAgents, PolicyError> {
    let mut agents = DecisionAgents::new();
    for settings in &scenario.agent_settings {
        let (policy, config) = match settings {
            AgentSettings::Attacker(a) => (&a.policy, &a.config),
            AgentSettings::Defender(d) => (&d.policy, &d.config),
        };
        if let Some(policy) = policy {
            agents.insert(
                settings.name().to_string(),
                policy_from_name(policy, config)?,
            );
        }
    }
    Ok(agents)
}

/// Run a simulation with the agents (and their built-in policies) of
/// `scenario`, on a `sim` built from `scenario.attack_graph`.
///
/// `max_steps`: if set, stop after this many steps even if the simulation
/// isn't done.
///
/// Returns the selected actions by each agent in each step.
pub fn run_simulation(
    sim: &mut Simulator,
    scenario: &Scenario,
    max_steps: Option<u64>,
) -> Result<BTreeMap<String, Vec<AttackGraphNodeId>>, RunSimulationError> {
    let mut agents = decision_agents(scenario)?;
    run_simulation_with_agents(sim, scenario, &mut agents, max_steps)
}

/// `run_simulation` with caller-supplied decision agents, e.g. policies
/// that are not built in. Agents of the scenario without an entry in
/// `agents` wait.
pub fn run_simulation_with_agents(
    sim: &mut Simulator,
    scenario: &Scenario,
    agents: &mut DecisionAgents,
    max_steps: Option<u64>,
) -> Result<BTreeMap<String, Vec<AttackGraphNodeId>>, RunSimulationError> {
    if !Rc::ptr_eq(sim.graph(), &scenario.attack_graph) {
        return Err(RunSimulationError::GraphMismatch);
    }
    let mut agent_actions: BTreeMap<String, Vec<AttackGraphNodeId>> = BTreeMap::new();
    let mut total_rewards: Vec<f64> = vec![0.0; scenario.agent_settings.len()];

    println!("Starting CLI env simulator.");
    // One generator for what Python's `MalSimulator.rng` is used for:
    // sampling entry points, seeding the simulator and `SAMPLE_TTC`
    // rewards.
    let mut rng = match scenario.sim_settings.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(&mut rand::rng()),
    };
    let flat_agents = scenario.flatten_agents(&mut rng)?;
    let state = sim.reset(&scenario.sim_settings, flat_agents, rng.random())?;
    let mut outcome = StepOutcome::from_reset(state);

    let mut iteration: u64 = 0;
    while !sim.state().is_some_and(|state| state.done())
        && max_steps.is_none_or(|max_steps| iteration < max_steps)
    {
        println!("Iteration {iteration}");
        let mut actions: HashMap<String, Vec<AttackGraphNodeId>> = HashMap::new();

        // Select actions for each agent
        {
            let attack_graph = scenario.attack_graph.borrow();
            let model = scenario.model.borrow();
            let state = sim.state().ok_or(SimulatorError::NotReset)?;
            for settings in &scenario.agent_settings {
                let agent_name = settings.name();
                let Some(decision_agent) = agents.get_mut(agent_name) else {
                    println!(
                        "Agent \"{agent_name}\" has no decision agent class \
                         specified in scenario. Waiting."
                    );
                    continue;
                };

                let agent_state =
                    AgentState::new(settings, &attack_graph, &model, state, &outcome)?;
                if let Some(agent_action) = decision_agent.get_next_action(&agent_state)? {
                    actions.insert(agent_name.to_string(), vec![agent_action]);

                    // Store agent action
                    agent_actions
                        .entry(agent_name.to_string())
                        .or_default()
                        .push(agent_action);
                }
            }
        }

        // Perform next step of simulation
        outcome = sim.step(&actions)?;
        {
            let attack_graph = scenario.attack_graph.borrow();
            let model = scenario.model.borrow();
            let state = sim.state().ok_or(SimulatorError::NotReset)?;
            for (settings, total_reward) in scenario.agent_settings.iter().zip(&mut total_rewards) {
                let agent_state =
                    AgentState::new(settings, &attack_graph, &model, state, &outcome)?;
                *total_reward += agent_reward(&agent_state, &mut rng)?;
            }
        }
        iteration += 1;
        println!("---");
    }

    println!("Simulation over after {iteration} steps.");

    // Print total rewards
    for (settings, total_reward) in scenario.agent_settings.iter().zip(&total_rewards) {
        println!("Total reward \"{}\" {total_reward:?}", settings.name());
    }

    Ok(agent_actions)
}
