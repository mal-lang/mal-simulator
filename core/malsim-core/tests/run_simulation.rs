//! Rewards, scenario-driven policies and `run_simulation`, end to end on
//! real scenario fixtures through `malsim-core`'s public API. Rust ports
//! of `tests/test_main.py`, `tests/agents/test_path_finders.py`,
//! `tests/agents/test_ttc_avoider.py`, `test_agents.py::
//! test_random_agent_wait_prob` and `test_mal_simulator.py`'s
//! `test_*_step_rewards_*`.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use malsim_core::agent_state::{AgentState, AgentStateError};
use malsim_core::graph_state::TtcMode;
use malsim_core::policies::{
    get_shortest_path_to, policy_from_name, DecisionAgent, KeyboardAgent, PolicyError, RandomAgent,
    ShortestPathAttacker, TTCSoftMinAttacker,
};
use malsim_core::rewards::{agent_reward, RewardError};
use malsim_core::run_simulation::{
    decision_agents, run_simulation, run_simulation_with_agents, DecisionAgents, RunSimulationError,
};
use malsim_core::scenario::{load_scenario_dict, Scenario};
use malsim_core::simulator::{Simulator, StepOutcome};
use malsim_core::ttc::TtcDist;
use malsim_core::types::NodeMap;
use maltoolbox_attackgraph::AttackGraphNodeId;
use maltoolbox_language::graph::attack_step::AttackStepType;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Map, Value};

fn testdata(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/testdata")
        .join(path)
}

fn load_scenario(name: &str) -> Scenario {
    Scenario::load_from_file(testdata("scenarios").join(name)).unwrap()
}

/// The scenario file `name` with its `agents` replaced.
fn load_scenario_with_agents(name: &str, agents: Value) -> Scenario {
    let mut scenario_dict = load_scenario_dict(testdata("scenarios").join(name)).unwrap();
    scenario_dict.insert("agents".to_string(), agents);
    Scenario::from_dict(&scenario_dict).unwrap()
}

/// coreLang + `simple_test_model.yml` (Python's `corelang_lang_graph` and
/// `model` fixtures) with the given agents.
fn corelang_scenario(agents: Value) -> Scenario {
    let scenario_dict = json!({
        "lang_file": testdata("langs/org.mal-lang.coreLang-1.0.0.mar").to_string_lossy(),
        "model_file": testdata("models/simple_test_model.yml").to_string_lossy(),
        "agents": agents,
    });
    Scenario::from_dict(scenario_dict.as_object().unwrap()).unwrap()
}

fn config(value: Value) -> Map<String, Value> {
    value.as_object().unwrap().clone()
}

/// A `MalSimulator.from_scenario` stand-in: a `Simulator` over a
/// scenario, plus the last step outcome agent states are built from.
struct Sim {
    scenario: Scenario,
    sim: Simulator,
    outcome: StepOutcome,
    rng: StdRng,
}

impl Sim {
    fn new(scenario: Scenario) -> Self {
        let mut sim = Sim {
            sim: Simulator::new(scenario.attack_graph.clone()),
            scenario,
            outcome: StepOutcome {
                attackers: BTreeMap::new(),
                defenders: BTreeMap::new(),
                step_enabled_defenses: Default::default(),
            },
            rng: StdRng::seed_from_u64(0),
        };
        sim.reset();
        sim
    }

    fn reset(&mut self) {
        let seed = self.scenario.sim_settings.seed.unwrap_or(0);
        let agents = self
            .scenario
            .flatten_agents(&mut StdRng::seed_from_u64(seed))
            .unwrap();
        let state = self
            .sim
            .reset(&self.scenario.sim_settings, agents, seed)
            .unwrap();
        self.outcome = StepOutcome::from_reset(state);
    }

    fn node(&self, full_name: &str) -> AttackGraphNodeId {
        self.scenario
            .attack_graph
            .borrow()
            .get_node_by_full_name(full_name)
            .unwrap()
    }

    fn name(&self, node: AttackGraphNodeId) -> String {
        self.scenario.attack_graph.borrow().nodes[node].name.clone()
    }

    fn with_state<R>(&self, agent_name: &str, f: impl FnOnce(&AgentState<'_>) -> R) -> R {
        let attack_graph = self.scenario.attack_graph.borrow();
        let model = self.scenario.model.borrow();
        let settings = self
            .scenario
            .agent_settings
            .iter()
            .find(|s| s.name() == agent_name)
            .unwrap();
        let state = AgentState::new(
            settings,
            &attack_graph,
            &model,
            self.sim.state().unwrap(),
            &self.outcome,
        )
        .unwrap();
        f(&state)
    }

    fn next_action(
        &self,
        agent_name: &str,
        agent: &mut dyn DecisionAgent,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        self.with_state(agent_name, |state| agent.get_next_action(state))
    }

    fn step(&mut self, agent_name: &str, actions: Vec<AttackGraphNodeId>) {
        self.outcome = self
            .sim
            .step(&HashMap::from([(agent_name.to_string(), actions)]))
            .unwrap();
    }

    fn reward(&mut self, agent_name: &str) -> Result<f64, RewardError> {
        let mut rng = StdRng::seed_from_u64(0);
        std::mem::swap(&mut rng, &mut self.rng);
        let reward = self.with_state(agent_name, |state| agent_reward(state, &mut rng));
        self.rng = rng;
        reward
    }

    fn done(&self) -> bool {
        self.sim.state().unwrap().done()
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

// --- Rewards ---

const ATTACKER: &str = "Test Attacker";

#[test]
fn attacker_step_rewards_cumulative() {
    let mut sim = Sim::new(corelang_scenario(json!({
        ATTACKER: {
            "type": "attacker",
            "entry_points": ["OS App:fullAccess"],
            "reward_mode": "CUMULATIVE",
            "rewards": {"by_asset_name": {"OS App": {
                "fullAccess": 10.0,
                "attemptRead": 100.0,
                "accessNetworkAndConnections": 50.4,
            }}},
        },
    })));
    // After reset only the entry point is compromised.
    assert_close(sim.reward(ATTACKER).unwrap(), 10.0);

    sim.step(ATTACKER, vec![sim.node("OS App:attemptRead")]);
    assert_close(sim.reward(ATTACKER).unwrap(), 10.0 + 100.0);

    sim.step(
        ATTACKER,
        vec![sim.node("OS App:accessNetworkAndConnections")],
    );
    assert_close(sim.reward(ATTACKER).unwrap(), 10.0 + 100.0 + 50.4);
}

#[test]
fn attacker_step_rewards_one_off() {
    let mut sim = Sim::new(corelang_scenario(json!({
        ATTACKER: {
            "type": "attacker",
            "entry_points": ["OS App:fullAccess"],
            "reward_mode": "ONE_OFF",
            "rewards": {"by_asset_type": {"Application": {
                "fullAccess": 10.0,
                "attemptRead": 100.0,
                "accessNetworkAndConnections": 50.4,
            }}},
        },
    })));

    // TTCs are disabled, so attempts are not penalized.
    sim.step(ATTACKER, vec![sim.node("OS App:attemptRead")]);
    assert_close(sim.reward(ATTACKER).unwrap(), 100.0);

    sim.step(
        ATTACKER,
        vec![sim.node("OS App:accessNetworkAndConnections")],
    );
    assert_close(sim.reward(ATTACKER).unwrap(), 50.4);

    // A step that compromises nothing is worth nothing.
    sim.step(ATTACKER, vec![]);
    assert_close(sim.reward(ATTACKER).unwrap(), 0.0);
}

#[test]
fn attacker_step_rewards_penalize_attempts_when_ttcs_are_enabled() {
    let mut scenario = corelang_scenario(json!({
        ATTACKER: {
            "type": "attacker",
            "entry_points": ["OS App:fullAccess"],
            "reward_mode": "ONE_OFF",
            "rewards": {"by_asset_name": {"OS App": {"attemptRead": 100.0}}},
        },
    }));
    scenario.sim_settings.ttc_mode = TtcMode::PerStepSample;
    scenario.sim_settings.seed = Some(1);
    let mut sim = Sim::new(scenario);

    let attempt_read = sim.node("OS App:attemptRead");
    sim.step(ATTACKER, vec![attempt_read]);
    let compromised = sim.outcome.attackers[ATTACKER]
        .step_performed_nodes
        .contains(&attempt_read);
    // Only a failed attempt counts as attempted (and is penalized).
    let expected = if compromised { 100.0 } else { -1.0 };
    assert_close(sim.reward(ATTACKER).unwrap(), expected);
}

/// Runs an episode stepping on the whole action surface, checking each
/// step's reward against the TTC penalty `penalty` gives a node's
/// `TtcDist`. Returns the summed penalty.
fn check_ttc_penalties(reward_mode: &str, penalty: impl Fn(&TtcDist) -> Option<f64>) -> f64 {
    // ttc_lang has "hard" steps with a TTC and "easy" ones without.
    let agents = json!({
        ATTACKER: {
            "type": "attacker",
            "entry_points": ["Net1:easyAccess"],
            "reward_mode": reward_mode,
            "rewards": {"by_asset_type": {"Network": {"hardAccess": 100.0}}},
        },
    });
    let mut sim = Sim::new(load_scenario_with_agents("ttc_lang_scenario.yml", agents));

    let mut total_penalty = 0.0;
    let mut steps = 0;
    while !sim.done() {
        assert!(steps < 1000, "episode did not terminate");
        steps += 1;
        let action_surface = sim.with_state(ATTACKER, |state| state.sorted_action_surface());
        sim.step(ATTACKER, action_surface);

        let (node_rewards, ttc_penalty, exact) = {
            let graph = sim.scenario.attack_graph.borrow();
            let model = sim.scenario.model.borrow();
            let mut node_rewards = 0.0;
            let mut ttc_penalty = 0.0;
            let mut exact = true;
            for &n in &sim.outcome.attackers[ATTACKER].step_performed_nodes {
                let node = &graph.nodes[n];
                let asset = model.get_asset_by_id(node.model_asset.unwrap()).unwrap();
                if asset.asset_type == "Network" && node.name == "hardAccess" {
                    node_rewards += 100.0;
                }
                if let Some(ttc) = &node.ttc {
                    match penalty(&TtcDist::from_dict(ttc).unwrap()) {
                        Some(p) => ttc_penalty += p,
                        None => exact = false,
                    }
                }
            }
            (node_rewards, ttc_penalty, exact)
        };
        let reward = sim.reward(ATTACKER).unwrap();
        if exact {
            // Penalized with the TTC value (since ttc mode is disabled)
            assert!((reward - (node_rewards - ttc_penalty)).abs() < 1e-6);
        } else {
            assert!(reward <= node_rewards);
        }
        total_penalty += node_rewards - reward;
    }
    total_penalty
}

#[test]
fn attacker_step_rewards_expected_ttc() {
    let total_penalty = check_ttc_penalties("EXPECTED_TTC", |dist| Some(dist.expected_value()));
    assert!(total_penalty > 0.0);
}

#[test]
fn attacker_step_rewards_sample_ttc() {
    // Sampled values can't be predicted here, only that they penalize.
    let total_penalty = check_ttc_penalties("SAMPLE_TTC", |_| None);
    assert!(total_penalty > 0.0);
}

const DEFENDER: &str = "defender";

fn defender_rewards_scenario(reward_mode: &str) -> Scenario {
    corelang_scenario(json!({
        DEFENDER: {
            "type": "defender",
            "reward_mode": reward_mode,
            "rewards": {"by_asset_name": {"OS App": {
                "notPresent": 100,
                "fullAccess": 10.0,
                "attemptRead": 105.0,
                "accessNetworkAndConnections": 35.04,
            }}},
        },
        // To be able to step
        ATTACKER: {"type": "attacker", "entry_points": ["OS App:fullAccess"]},
    }))
}

#[test]
fn defender_step_rewards_cumulative() {
    let mut sim = Sim::new(defender_rewards_scenario("CUMULATIVE"));

    sim.step(ATTACKER, vec![sim.node("OS App:attemptRead")]);
    assert_close(sim.reward(DEFENDER).unwrap(), -(10.0 + 105.0));

    sim.step(
        ATTACKER,
        vec![sim.node("OS App:accessNetworkAndConnections")],
    );
    assert_close(sim.reward(DEFENDER).unwrap(), -(10.0 + 105.0 + 35.04));

    // Enabling a defense costs its reward too.
    sim.step(DEFENDER, vec![sim.node("OS App:notPresent")]);
    assert_close(
        sim.reward(DEFENDER).unwrap(),
        -(10.0 + 105.0 + 35.04 + 100.0),
    );
}

#[test]
fn defender_step_rewards_one_off() {
    let mut sim = Sim::new(defender_rewards_scenario("ONE_OFF"));

    sim.step(ATTACKER, vec![sim.node("OS App:attemptRead")]);
    assert_close(sim.reward(DEFENDER).unwrap(), -105.0);

    sim.step(
        ATTACKER,
        vec![sim.node("OS App:accessNetworkAndConnections")],
    );
    assert_close(sim.reward(DEFENDER).unwrap(), -35.04);

    sim.step(DEFENDER, vec![sim.node("OS App:notPresent")]);
    assert_close(sim.reward(DEFENDER).unwrap(), -100.0);
}

#[test]
fn defender_step_rewards_reject_ttc_reward_modes() {
    let mut sim = Sim::new(defender_rewards_scenario("EXPECTED_TTC"));
    assert!(matches!(
        sim.reward(DEFENDER),
        Err(RewardError::UnsupportedDefenderRewardMode(_))
    ));
}

// --- Path finding ---

#[test]
fn path_finding() {
    let sim = Sim::new(load_scenario("traininglang_scenario.yml"));
    let graph = sim.scenario.attack_graph.borrow();
    let data_2_read = graph.get_node_by_full_name("Data:2:read").unwrap();

    let performed_nodes: Vec<_> = sim.sim.state().unwrap().attackers["Attacker1"]
        .performed_nodes
        .iter()
        .copied()
        .collect();
    let ttc_values = graph.nodes.keys().map(|n| (n, 1.0)).collect();
    let path = get_shortest_path_to(&graph, &performed_nodes, data_2_read, &ttc_values).unwrap();

    let model = sim.scenario.model.borrow();
    let full_names: Vec<_> = path
        .iter()
        .map(|&n| graph.full_name_of(n, Some(&model)))
        .collect();
    assert_eq!(full_names, ["Host:0:access", "Data:2:read"]);
}

#[test]
fn path_finding_ttc_lang() {
    let mut scenario = load_scenario("ttc_lang_scenario.yml");
    scenario.sim_settings.ttc_mode = TtcMode::ExpectedValue;
    scenario.sim_settings.seed = Some(100);
    let sim = Sim::new(scenario);
    let graph = sim.scenario.attack_graph.borrow();

    let entry_point = graph.get_node_by_full_name("Net1:easyAccess").unwrap();
    let goal = graph.get_node_by_full_name("DataD:read").unwrap();
    let ttc_values: NodeMap<_> = graph
        .nodes
        .iter()
        .filter(|(_, n)| matches!(n.step_type, AttackStepType::Or | AttackStepType::And))
        .map(|(key, _)| (key, sim.sim.state().unwrap().graph_state.ttc_values[&key]))
        .collect();
    assert_close(ttc_values.values().sum(), 2021.0);

    let path = get_shortest_path_to(&graph, &[entry_point], goal, &ttc_values).unwrap();

    assert!(!path.is_empty());
    assert_eq!(path.last(), Some(&goal));
    for &node in &path {
        // Should only have picked the low ttc steps
        assert!(graph.nodes[node].name.contains("easy") || node == goal);
    }

    // No path from a node the goal isn't reachable from, and nothing left
    // to do when the goal is already compromised.
    assert!(
        get_shortest_path_to(&graph, &[goal], entry_point, &ttc_values)
            .unwrap()
            .is_empty()
    );
    assert!(get_shortest_path_to(&graph, &[goal], goal, &ttc_values)
        .unwrap()
        .is_empty());
}

#[test]
fn shortest_path_attacker_reaches_its_goal_on_easy_steps() {
    let mut scenario = load_scenario("ttc_lang_scenario.yml");
    scenario.sim_settings.ttc_mode = TtcMode::ExpectedValue;
    scenario.sim_settings.seed = Some(100);
    let mut sim = Sim::new(scenario);
    let goal = sim.node("DataD:read");
    let mut attacker_agent = ShortestPathAttacker::new(&config(json!({"seed": 0}))).unwrap();

    let mut picked = Vec::new();
    for _ in 0..10_000 {
        if sim.done() {
            break;
        }
        let action = sim.next_action("Attacker", &mut attacker_agent).unwrap();
        let action = action.expect("a path to the goal");
        assert!(sim.name(action).contains("easy") || action == goal);
        picked.push(action);
        sim.step("Attacker", vec![action]);
    }
    assert!(sim.done());
    assert_eq!(picked.last(), Some(&goal));
}

#[test]
fn shortest_path_attacker_needs_goals_and_ttc_values() {
    let goalless = json!({"Attacker": {
        "type": "attacker", "entry_points": ["ComputerA:easyAccess"],
    }});
    let mut scenario = load_scenario_with_agents("ttc_lang_scenario.yml", goalless);
    scenario.sim_settings.ttc_mode = TtcMode::ExpectedValue;
    let sim = Sim::new(scenario);
    let mut attacker_agent = ShortestPathAttacker::new(&Map::new()).unwrap();
    assert!(matches!(
        sim.next_action("Attacker", &mut attacker_agent),
        Err(PolicyError::NoGoals)
    ));

    // TTC values only exist for PRE_SAMPLE and EXPECTED_VALUE.
    let sim = Sim::new(load_scenario("ttc_lang_scenario.yml"));
    assert!(matches!(
        sim.next_action("Attacker", &mut attacker_agent),
        Err(PolicyError::AgentState(
            AgentStateError::TtcValuesUnavailable(TtcMode::Disabled)
        ))
    ));
}

// --- TTC soft min ---

const TTC_AVOIDER: &str = "TTCAvoidingAttacker";

fn ttc_avoider_sim(seed: u64) -> Sim {
    let agents = json!({TTC_AVOIDER: {
        "type": "attacker",
        "entry_points": ["Net1:easyAccess"],
        "goals": ["DataD:read"],
    }});
    let mut scenario = load_scenario_with_agents("ttc_lang_scenario.yml", agents);
    scenario.sim_settings.seed = Some(seed);
    scenario.sim_settings.ttc_mode = TtcMode::PreSample;
    scenario.sim_settings.attack_surface.skip_unnecessary = false;
    Sim::new(scenario)
}

#[test]
fn ttc_avoider() {
    // `TTCSoftMinAttacker` samples from a softmax, not an argmin, so a
    // "hard" step whose pre-sampled TTC came out low has a real chance of
    // being drawn. As in the Python test, the simulator and policy seeds
    // pin a run where that doesn't happen (with these TTCs, 18 of the
    // policy seeds 0..20 pick only easy steps).
    let mut sim = ttc_avoider_sim(2);
    let goal = sim.node("DataD:read");
    let mut attacker_agent = TTCSoftMinAttacker::new(&config(json!({"seed": 0}))).unwrap();

    while !sim.done() {
        // Run the simulation until agents are terminated/truncated
        let attacker_node = sim.next_action(TTC_AVOIDER, &mut attacker_agent).unwrap();
        let attacker_node = attacker_node.expect("an action");

        // Should always pick the easy path or the goal
        assert!(sim.name(attacker_node).contains("easy") || attacker_node == goal);
        sim.step(TTC_AVOIDER, vec![attacker_node]);
    }
}

#[test]
fn ttc_avoider_low_sharpness() {
    // TTC Avoider with low beta/sharpness explores both easy and hard paths.
    let num_trials = 20;
    let mut easy_count = 0;
    let mut hard_count = 0;

    for trial in 0..num_trials {
        let mut sim = ttc_avoider_sim(trial);
        let mut attacker_agent =
            TTCSoftMinAttacker::new(&config(json!({"beta": 0.1, "seed": trial}))).unwrap();

        let mut trial_picked_easy = false;
        let mut trial_picked_hard = false;
        while !sim.done() {
            let attacker_node = sim.next_action(TTC_AVOIDER, &mut attacker_agent).unwrap();
            let attacker_node = attacker_node.expect("an action");

            trial_picked_easy |= sim.name(attacker_node).contains("easy");
            trial_picked_hard |= sim.name(attacker_node).contains("hard");
            sim.step(TTC_AVOIDER, vec![attacker_node]);
        }
        easy_count += u64::from(trial_picked_easy);
        hard_count += u64::from(trial_picked_hard);
    }

    let min_trials = num_trials * 8 / 10;
    assert!(easy_count >= min_trials, "{easy_count}/{num_trials}");
    assert!(hard_count >= min_trials, "{hard_count}/{num_trials}");
}

// --- Random agent from a scenario ---

#[test]
fn random_agent_wait_prob() {
    let mut sim = Sim::new(load_scenario("simple_random_defender.yml"));
    let defender_name = "Defender";
    let defender_config = sim
        .scenario
        .defender_settings()
        .next()
        .unwrap()
        .config
        .clone();
    let wait_prob = defender_config["wait_prob"].as_f64().unwrap();
    let mut defender_ai = RandomAgent::new(&defender_config).unwrap();

    let mut num_actions = 0;
    let mut wait_count = 0;
    for _ in 0..100 {
        sim.reset();
        while !sim.with_state(defender_name, |s| s.action_surface().is_empty()) {
            let action = sim.next_action(defender_name, &mut defender_ai).unwrap();
            num_actions += 1;
            wait_count += u32::from(action.is_none());
            sim.step(defender_name, action.into_iter().collect());
        }
    }

    let wait_freq = f64::from(wait_count) / f64::from(num_actions);
    assert!((wait_freq - wait_prob).abs() < 0.01, "{wait_freq}");
}

// --- run_simulation ---

/// Every recorded action must be a node of the scenario's graph.
fn assert_actions_are_graph_nodes(
    scenario: &Scenario,
    agent_actions: &BTreeMap<String, Vec<AttackGraphNodeId>>,
) {
    let graph = scenario.attack_graph.borrow();
    for actions in agent_actions.values() {
        assert!(actions.iter().all(|&n| graph.nodes.contains_key(n)));
    }
}

#[test]
fn run_simulation_with_defender_agent() {
    // Make sure we can run simulation with defender agent registered in
    // scenario
    let scenario = load_scenario("bfs_vs_bfs_scenario.yml");
    let mut sim = Simulator::new(scenario.attack_graph.clone());
    let agent_actions = run_simulation(&mut sim, &scenario, None).unwrap();

    assert!(sim.state().unwrap().done());
    assert!(!agent_actions["attacker1"].is_empty());
    assert!(!agent_actions["defender1"].is_empty());
    assert_actions_are_graph_nodes(&scenario, &agent_actions);

    // Every attacker action was compromised (or at least attempted), and
    // every defender action enabled a defense.
    let state = sim.state().unwrap();
    let attacker = &state.attackers["attacker1"];
    for action in &agent_actions["attacker1"] {
        assert!(
            attacker.performed_nodes.contains(action) || attacker.attempted_nodes.contains(action)
        );
    }
    for action in &agent_actions["defender1"] {
        assert!(state.enabled_defenses.contains(action));
    }
}

#[test]
fn run_simulation_without_defender_agent() {
    // Make sure we can run simulation without defender agent registered in
    // scenario
    let scenario = load_scenario("no_defender_agent_scenario.yml");
    let mut sim = Simulator::new(scenario.attack_graph.clone());
    let agent_actions = run_simulation(&mut sim, &scenario, None).unwrap();

    assert!(sim.state().unwrap().done());
    assert_eq!(
        agent_actions.keys().collect::<Vec<_>>(),
        ["attacker1"],
        "only the attacker acts"
    );
    assert_actions_are_graph_nodes(&scenario, &agent_actions);
}

#[test]
fn run_simulation_is_reproducible_for_a_seed() {
    let run = || {
        let mut scenario = load_scenario("bfs_vs_bfs_scenario.yml");
        scenario.sim_settings.seed = Some(5);
        scenario.sim_settings.ttc_mode = TtcMode::PerStepSample;
        let mut sim = Simulator::new(scenario.attack_graph.clone());
        let agent_actions = run_simulation(&mut sim, &scenario, None).unwrap();
        let graph = scenario.attack_graph.borrow();
        agent_actions
            .into_iter()
            .map(|(name, actions)| {
                let ids: Vec<i64> = actions.iter().map(|&n| graph.nodes[n].id).collect();
                (name, ids)
            })
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn run_simulation_stops_at_max_steps() {
    let scenario = load_scenario("bfs_vs_bfs_scenario.yml");
    let mut sim = Simulator::new(scenario.attack_graph.clone());
    let agent_actions = run_simulation(&mut sim, &scenario, Some(3)).unwrap();

    assert!(!sim.state().unwrap().done());
    assert_eq!(sim.state().unwrap().attackers["attacker1"].iteration, 3);
    assert_eq!(agent_actions["attacker1"].len(), 3);
}

#[test]
fn run_simulation_agents_without_policy_wait() {
    // `agent_class: null` for the attacker: nothing ever happens, so the
    // run only ends at `max_steps`.
    let scenario = load_scenario("simple_ai_scenario_with_heuristic_defender.yml");
    assert!(!decision_agents(&scenario)
        .unwrap()
        .contains_key("Attacker1"));
    let mut sim = Simulator::new(scenario.attack_graph.clone());
    let agent_actions = run_simulation(&mut sim, &scenario, Some(5)).unwrap();
    assert!(!agent_actions.contains_key("Attacker1"));
}

#[test]
fn run_simulation_with_keyboard_agent() {
    // Python patches `input()` to return an empty line; an agent fed empty
    // lines always waits.
    let scenario = load_scenario("no_defender_agent_scenario.yml");
    let mut sim = Simulator::new(scenario.attack_graph.clone());
    let input: &[u8] = b"\n\n\n";
    let mut agents = DecisionAgents::new();
    agents.insert(
        "attacker1".to_string(),
        Box::new(KeyboardAgent::with_io(input, std::io::sink())),
    );
    let agent_actions =
        run_simulation_with_agents(&mut sim, &scenario, &mut agents, Some(3)).unwrap();
    assert!(agent_actions.is_empty());
}

#[test]
fn run_simulation_rejects_unknown_policies_and_foreign_simulators() {
    // The loader carries unknown policy names; instantiating them fails.
    let scenario = load_scenario("wrong_agent_classes_scenario.yml");
    let mut sim = Simulator::new(scenario.attack_graph.clone());
    assert!(matches!(
        run_simulation(&mut sim, &scenario, None),
        Err(RunSimulationError::Policy(PolicyError::UnknownPolicy(_)))
    ));
    assert!(policy_from_name("BreadthFirstAttacker", &Map::new()).is_ok());

    let scenario = load_scenario("bfs_vs_bfs_scenario.yml");
    let other = load_scenario("bfs_vs_bfs_scenario.yml");
    let mut sim = Simulator::new(other.attack_graph.clone());
    assert!(matches!(
        run_simulation(&mut sim, &scenario, None),
        Err(RunSimulationError::GraphMismatch)
    ));
}
