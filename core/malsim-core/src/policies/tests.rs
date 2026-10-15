//! Rust ports of `tests/agents/test_searchers.py` and the hand-built-graph
//! tests of `tests/agents/test_agents.py`. The scenario-file tests of that
//! directory are in `core/malsim-core/tests/run_simulation.rs`.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_model::Model;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Map, Value};

use super::*;
use crate::agent_state::AgentState;
use crate::scenario::agent_settings::{
    AgentSettings, AttackerSettings, DefenderSettings, EntryPoints,
};
use crate::scenario::flatten::{
    flatten_attacker_settings, flatten_defender_settings, get_entry_points,
};
use crate::scenario::node_property_rule::NodePropertyRule;
use crate::settings::{FlatAgentSettings, MalSimulatorSettings, RewardMode};
use crate::simulator::{Simulator, StepOutcome};
use crate::test_fixtures::dummy_graph_and_model;

/// A `MalSimulator` stand-in for a hand-built graph: a `Simulator` plus
/// the agent settings and last step outcome agent states are built from.
struct Sim {
    graph: Rc<RefCell<AttackGraph>>,
    model: Model,
    settings: Vec<AgentSettings<AttackGraphNodeId>>,
    sim: Simulator,
    outcome: StepOutcome,
}

impl Sim {
    fn new(
        graph: AttackGraph,
        model: Model,
        settings: Vec<AgentSettings<AttackGraphNodeId>>,
    ) -> Self {
        let graph = Rc::new(RefCell::new(graph));
        let mut sim = Sim {
            sim: Simulator::new(graph.clone()),
            graph,
            model,
            settings,
            outcome: StepOutcome {
                attackers: Default::default(),
                defenders: Default::default(),
                step_enabled_defenses: Default::default(),
            },
        };
        sim.reset();
        sim
    }

    fn reset(&mut self) {
        let graph = self.graph.borrow();
        let mut rng = StdRng::seed_from_u64(0);
        let agents = self
            .settings
            .iter()
            .map(|settings| {
                let flat = match settings {
                    AgentSettings::Attacker(a) => {
                        FlatAgentSettings::Attacker(flatten_attacker_settings(
                            a,
                            &graph,
                            &self.model,
                            &get_entry_points(a, &mut rng),
                        ))
                    }
                    AgentSettings::Defender(d) => FlatAgentSettings::Defender(
                        flatten_defender_settings(d, &graph, &self.model),
                    ),
                };
                (settings.name().to_string(), flat)
            })
            .collect();
        drop(graph);
        let state = self
            .sim
            .reset(&MalSimulatorSettings::default(), agents, 0)
            .unwrap();
        self.outcome = StepOutcome::from_reset(state);
    }

    fn next_action(
        &self,
        name: &str,
        agent: &mut dyn DecisionAgent,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        let graph = self.graph.borrow();
        let settings = self.settings.iter().find(|s| s.name() == name).unwrap();
        let agent_state = AgentState::new(
            settings,
            &graph,
            &self.model,
            self.sim.state().unwrap(),
            &self.outcome,
        )
        .unwrap();
        agent.get_next_action(&agent_state)
    }

    /// Steps `name` on `action` and returns the ids of the nodes that
    /// step compromised.
    fn step(&mut self, name: &str, action: AttackGraphNodeId) -> Vec<i64> {
        self.outcome = self
            .sim
            .step(&HashMap::from([(name.to_string(), vec![action])]))
            .unwrap();
        let graph = self.graph.borrow();
        let mut ids: Vec<i64> = self.outcome.attackers[name]
            .step_performed_nodes
            .iter()
            .map(|&n| graph.nodes[n].id)
            .collect();
        ids.sort_unstable();
        ids
    }

    fn id(&self, node: AttackGraphNodeId) -> i64 {
        self.graph.borrow().nodes[node].id
    }
}

/// Adds a node of the `DummyAsset` attack step `step_name`, backed by a
/// new model asset named `asset<node id>`.
fn add_node(graph: &mut AttackGraph, model: &mut Model, step_name: &str) -> AttackGraphNodeId {
    let asset_id = graph.lang_graph.asset_id("DummyAsset").unwrap();
    let step_id = graph.lang_graph.assets[asset_id].attack_steps[step_name];
    let name = format!("asset{}", graph.nodes.len());
    let model_asset = model
        .add_asset("DummyAsset", Some(name), None, None, None, false)
        .unwrap();
    graph
        .add_node(
            step_id,
            Some(model),
            None,
            Some(model_asset),
            None,
            None,
            None,
        )
        .unwrap()
}

/// Port of `tests/conftest.py::connect_nodes`.
fn connect_nodes(graph: &mut AttackGraph, parent: AttackGraphNodeId, child: AttackGraphNodeId) {
    graph.nodes[parent].children.insert(child);
    graph.nodes[child].parents.insert(parent);
}

fn attacker(name: &str, entry_point: AttackGraphNodeId) -> AgentSettings<AttackGraphNodeId> {
    AgentSettings::Attacker(AttackerSettings {
        name: name.to_string(),
        entry_points: EntryPoints::Single(BTreeSet::from([entry_point])),
        policy: None,
        actionable_steps: None,
        rewards: None,
        config: Map::new(),
        reward_mode: RewardMode::Cumulative,
        goals: BTreeSet::new(),
        ttc_dists: None,
    })
}

fn defender(name: &str, rewards: Option<Value>) -> AgentSettings<AttackGraphNodeId> {
    AgentSettings::Defender(DefenderSettings {
        name: name.to_string(),
        policy: None,
        observable_steps: None,
        actionable_steps: None,
        rewards: rewards.map(|r| NodePropertyRule::from_value(&r).unwrap()),
        false_positive_rates: None,
        false_negative_rates: None,
        config: Map::new(),
        reward_mode: RewardMode::Cumulative,
    })
}

fn config(value: Value) -> Map<String, Value> {
    value.as_object().unwrap().clone()
}

/// `num_nodes` OR nodes connected by `edges` (pairs of node indexes), with
/// an attacker named `name` entering at node 0.
fn or_graph_sim(name: &str, num_nodes: usize, edges: &[(usize, usize)]) -> Sim {
    let (mut graph, mut model) = dummy_graph_and_model();
    let nodes: Vec<_> = (0..num_nodes)
        .map(|_| add_node(&mut graph, &mut model, "DummyOrAttackStep"))
        .collect();
    for &(parent, child) in edges {
        connect_nodes(&mut graph, nodes[parent], nodes[child]);
    }
    let settings = vec![attacker(name, nodes[0])];
    Sim::new(graph, model, settings)
}

/// ```text
///                 node0 ______________
///               /       \             \
///         node1          node2        node7
///         /   \           /   \
///     node3   node4    node5  node6
/// ```
const COMPLICATED: [(usize, usize); 7] = [(0, 1), (0, 2), (0, 7), (1, 3), (1, 4), (2, 5), (2, 6)];

/// Lets `agent` pick `steps` actions, stepping on each, and returns the
/// compromised node ids in order.
fn traversal_order(
    sim: &mut Sim,
    name: &str,
    agent: &mut dyn DecisionAgent,
    steps: usize,
) -> Vec<i64> {
    let mut actual_order = Vec::new();
    for _ in 0..steps {
        let action_node = sim.next_action(name, agent).unwrap().expect("an action");
        let performed = sim.step(name, action_node);
        assert_eq!(performed.len(), 1);
        actual_order.push(performed[0]);
    }
    actual_order
}

#[test]
fn breadth_first_traversal_simple() {
    // node0 -> node1 -> node2 -> node3
    let mut sim = or_graph_sim("bfs", 4, &[(0, 1), (1, 2), (2, 3)]);
    let mut attacker_ai =
        BreadthFirstAttacker::new(&config(json!({"seed": 42, "randomize": false}))).unwrap();

    assert_eq!(
        traversal_order(&mut sim, "bfs", &mut attacker_ai, 3),
        [1, 2, 3]
    );
    // Nothing left to do.
    assert_eq!(sim.next_action("bfs", &mut attacker_ai).unwrap(), None);
}

#[test]
fn breadth_first_traversal_complicated() {
    for action_ordering in ["nothing", "sorted", "random", "list_random"] {
        let mut sim = or_graph_sim("bfs", 8, &COMPLICATED);
        let mut attacker_ai = BreadthFirstAttacker::new(&config(
            json!({"seed": 42, "action_ordering": action_ordering}),
        ))
        .unwrap();

        let mut actual_order = traversal_order(&mut sim, "bfs", &mut attacker_ai, 7);
        if matches!(action_ordering, "nothing" | "sorted") {
            assert_eq!(actual_order, [1, 2, 7, 3, 4, 5, 6]);
        }
        // Each level is exhausted before the next one starts.
        actual_order[..3].sort_unstable();
        actual_order[3..].sort_unstable();
        assert_eq!(actual_order, [1, 2, 7, 3, 4, 5, 6], "{action_ordering}");
    }
}

#[test]
fn depth_first_traversal_complicated() {
    for action_ordering in ["nothing", "sorted", "random", "list_random"] {
        let mut sim = or_graph_sim("dfs", 8, &COMPLICATED);
        let mut attacker_ai = DepthFirstAttacker::new(&config(
            json!({"seed": 42, "action_ordering": action_ordering}),
        ))
        .unwrap();

        let actual_order = traversal_order(&mut sim, "dfs", &mut attacker_ai, 7);

        // All children of 1 must come directly after it
        let index1 = actual_order.iter().position(|&id| id == 1).unwrap();
        let mut children = actual_order[index1 + 1..index1 + 3].to_vec();
        children.sort_unstable();
        assert_eq!(children, [3, 4], "{action_ordering}");

        // All children of 2 must come directly after it
        let index2 = actual_order.iter().position(|&id| id == 2).unwrap();
        let mut children = actual_order[index2 + 1..index2 + 3].to_vec();
        children.sort_unstable();
        assert_eq!(children, [5, 6], "{action_ordering}");
    }
}

#[test]
fn searcher_seeded_random_ordering_is_reproducible() {
    let order = || {
        let mut sim = or_graph_sim("bfs", 8, &COMPLICATED);
        let mut attacker_ai =
            BreadthFirstAttacker::new(&config(json!({"seed": 7, "action_ordering": "random"})))
                .unwrap();
        traversal_order(&mut sim, "bfs", &mut attacker_ai, 7)
    };
    assert_eq!(order(), order());
}

#[test]
fn searcher_rejects_invalid_config() {
    assert!(matches!(
        BreadthFirstAttacker::new(&config(json!({"action_ordering": "alphabetical"}))),
        Err(PolicyError::InvalidConfig {
            field: "action_ordering",
            ..
        })
    ));
    assert!(matches!(
        DepthFirstAttacker::new(&config(json!({"seed": "one"}))),
        Err(PolicyError::InvalidConfig { field: "seed", .. })
    ));
}

/// ```text
/// node0 -------+---------|
///  |   node1   |  node2  |
///  |   /    \  |  /   \  |
/// node3      node4    node5
/// ```
/// node1/node2 are defenses, the rest AND steps; the attacker enters at
/// node4. `rewards` are the defender's.
fn defend_compromised_sim(rewards: Option<Value>) -> (Sim, Vec<AttackGraphNodeId>) {
    let (mut graph, mut model) = dummy_graph_and_model();
    let steps = ["And", "Defense", "Defense", "And", "And", "And"];
    let nodes: Vec<_> = steps
        .iter()
        .map(|step| add_node(&mut graph, &mut model, &format!("Dummy{step}AttackStep")))
        .collect();
    for (parent, child) in [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (2, 4), (2, 5)] {
        connect_nodes(&mut graph, nodes[parent], nodes[child]);
    }
    let settings = vec![attacker("bfs", nodes[4]), defender("def_comp", rewards)];
    (Sim::new(graph, model, settings), nodes)
}

fn defense_costs(node1: f64, node2: f64) -> Option<Value> {
    Some(json!({"by_asset_name": {
        "asset1": {"DummyDefenseAttackStep": node1},
        "asset2": {"DummyDefenseAttackStep": node2},
    }}))
}

#[test]
fn defend_compromised_defender() {
    let agent_config = config(json!({"seed": 42, "randomize": false}));

    // Both defenses have the compromised node4 as a child: without
    // rewards they cost the same and the lowest node id wins.
    let (sim, nodes) = defend_compromised_sim(None);
    let mut defender_ai = DefendCompromisedDefender::new(&agent_config);
    let action_node = sim.next_action("def_comp", &mut defender_ai).unwrap();
    assert_eq!(action_node, Some(nodes[1]));
    assert_eq!(sim.id(nodes[1]), 1);

    // Should pick cheapest one
    let (sim, nodes) = defend_compromised_sim(defense_costs(100.0, 10.0));
    let mut defender_ai = DefendCompromisedDefender::new(&agent_config);
    let action_node = sim.next_action("def_comp", &mut defender_ai).unwrap();
    assert_eq!(action_node, Some(nodes[2]));

    let (sim, nodes) = defend_compromised_sim(defense_costs(10.0, 100.0));
    let mut defender_ai = DefendCompromisedDefender::new(&agent_config);
    let action_node = sim.next_action("def_comp", &mut defender_ai).unwrap();
    assert_eq!(action_node, Some(nodes[1]));
}

#[test]
fn defend_future_compromised_defender() {
    // node0 -------+-----------|
    //  |   node1   |    node2  |
    //  |   /    \  |    /   \  |
    // node3      node4  |   node5
    //              |    |
    //              \    /
    //              node 6
    let (mut graph, mut model) = dummy_graph_and_model();
    let steps = ["And", "Defense", "Defense", "And", "And", "And", "And"];
    let nodes: Vec<_> = steps
        .iter()
        .map(|step| add_node(&mut graph, &mut model, &format!("Dummy{step}AttackStep")))
        .collect();
    for (parent, child) in [
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 3),
        (1, 4),
        (2, 5),
        (2, 6),
        (4, 6),
    ] {
        connect_nodes(&mut graph, nodes[parent], nodes[child]);
    }
    let settings = vec![attacker("bfs", nodes[4]), defender("def_future_comp", None)];
    let sim = Sim::new(graph, model, settings);

    let mut defender_ai =
        DefendFutureCompromisedDefender::new(&config(json!({"seed": 42, "randomize": false})));

    // Should pick node 2 either way: only its child node6 is not
    // compromised but has a compromised parent (node4).
    let action_node = sim
        .next_action("def_future_comp", &mut defender_ai)
        .unwrap();
    assert_eq!(action_node, Some(nodes[2]));
}

#[test]
fn defender_policies_reject_attacker_states() {
    let (sim, _) = defend_compromised_sim(None);
    let agent_config = Map::new();
    assert!(matches!(
        sim.next_action("bfs", &mut DefendCompromisedDefender::new(&agent_config)),
        Err(PolicyError::WrongAgentType {
            expected: "defender",
            ..
        })
    ));
    assert!(matches!(
        sim.next_action(
            "def_comp",
            &mut TTCSoftMinAttacker::new(&agent_config).unwrap()
        ),
        Err(PolicyError::WrongAgentType {
            expected: "attacker",
            ..
        })
    ));
    assert!(matches!(
        sim.next_action(
            "def_comp",
            &mut ShortestPathAttacker::new(&agent_config).unwrap()
        ),
        Err(PolicyError::WrongAgentType {
            expected: "attacker",
            ..
        })
    ));
}

#[test]
fn random_agent() {
    // node0 connected to all other nodes
    let mut sim = or_graph_sim("random_attacker", 4, &[(0, 1), (0, 2), (0, 3)]);
    let attacker_name = "random_attacker";

    let mut attacker_ai = RandomAgent::new(&config(json!({"seed": 0}))).unwrap();
    let mut counter: HashMap<Option<i64>, u32> = HashMap::new();
    for _ in 0..1_500 {
        sim.reset();
        let action_node = sim.next_action(attacker_name, &mut attacker_ai).unwrap();
        *counter.entry(action_node.map(|n| sim.id(n))).or_default() += 1;
    }
    assert_eq!(counter.len(), 3);
    assert!(!counter.contains_key(&None));
    for &count in counter.values() {
        let prob = f64::from(count) / 1_500.0;
        assert!((prob - 1.0 / 3.0).abs() < 0.04, "{counter:?}");
    }

    let mut attacker_ai = RandomAgent::new(&config(json!({"seed": 0, "wait_prob": 0.05}))).unwrap();
    let mut waits = 0;
    for _ in 0..1_500 {
        sim.reset();
        if sim
            .next_action(attacker_name, &mut attacker_ai)
            .unwrap()
            .is_none()
        {
            waits += 1;
        }
    }
    let wait_prob = f64::from(waits) / 1_500.0;
    assert!((wait_prob - 0.05).abs() < 0.04, "{wait_prob}");
}

#[test]
fn passive_agent_never_acts() {
    let sim = or_graph_sim("passive", 2, &[(0, 1)]);
    assert_eq!(sim.next_action("passive", &mut PassiveAgent).unwrap(), None);
}

/// A `KeyboardAgent` fed `input`, with what it printed.
fn keyboard_action(
    sim: &Sim,
    input: &str,
) -> (Result<Option<AttackGraphNodeId>, PolicyError>, String) {
    let mut output = Vec::new();
    let mut agent = KeyboardAgent::with_io(input.as_bytes(), &mut output);
    let action = sim.next_action("keyboard", &mut agent);
    (action, String::from_utf8(output).unwrap())
}

#[test]
fn keyboard_agent_reads_an_action_index() {
    let sim = or_graph_sim("keyboard", 4, &[(0, 1), (0, 2), (0, 3)]);
    let surface = {
        let graph = sim.graph.borrow();
        crate::agent_state::sorted_by_id(
            &graph,
            sim.sim.state().unwrap().attackers["keyboard"]
                .action_surface
                .iter()
                .copied(),
        )
    };
    assert_eq!(surface.len(), 3);

    let (action, output) = keyboard_action(&sim, "1\n");
    assert_eq!(action.unwrap(), Some(surface[1]));
    assert!(output.contains("Available actions:"));
    assert!(output.contains("0. asset1:DummyOrAttackStep"), "{output}");
    assert!(output.contains("Selected action: asset2:DummyOrAttackStep"));

    // An empty line waits.
    let (action, output) = keyboard_action(&sim, "\n");
    assert_eq!(action.unwrap(), None);
    assert!(output.ends_with("wait\n"));

    // Invalid entries (not a number, out of range) are asked again.
    let (action, output) = keyboard_action(&sim, "x\n3\n-1\n2\n");
    assert_eq!(action.unwrap(), Some(surface[2]));
    assert_eq!(output.matches("Invalid action.").count(), 3);

    // Running out of input is an error, not a wait.
    let (action, _) = keyboard_action(&sim, "");
    assert!(matches!(action, Err(PolicyError::Io(_))));
}

#[test]
fn keyboard_agent_without_actions_does_not_prompt() {
    let sim = or_graph_sim("keyboard", 1, &[]);
    let (action, output) = keyboard_action(&sim, "");
    assert_eq!(action.unwrap(), None);
    assert_eq!(output, "No actions to pick for defender\n");
}

#[test]
fn policy_from_name_knows_every_python_policy() {
    for name in POLICY_NAMES {
        assert!(policy_from_name(name, &Map::new()).is_ok(), "{name}");
    }
    assert!(matches!(
        policy_from_name("NotARealPolicy", &Map::new()),
        Err(PolicyError::UnknownPolicy(name)) if name == "NotARealPolicy"
    ));
}
