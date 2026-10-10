//! Phase C5 smoke test (`PORTING_NOTES.md` §7): loads real scenario
//! fixtures with the Rust-only scenario loader, builds a `Simulator` from
//! them and steps it with hand-picked actions - using only
//! `malsim-core`'s public API, with no Python or PyO3 anywhere in the
//! dependency graph. This is the embedding-Rust-program story end to end.

use std::collections::HashMap;
use std::path::PathBuf;

use malsim_core::scenario::Scenario;
use malsim_core::simulator::Simulator;
use maltoolbox_attackgraph::AttackGraphNodeId;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn scenario_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/testdata/scenarios")
        .join(name)
}

fn full_name(scenario: &Scenario, id: AttackGraphNodeId) -> String {
    scenario
        .attack_graph
        .borrow()
        .full_name_of(id, Some(&scenario.model.borrow()))
}

fn node(scenario: &Scenario, full_name: &str) -> AttackGraphNodeId {
    scenario
        .attack_graph
        .borrow()
        .get_node_by_full_name(full_name)
        .unwrap()
}

/// A deterministic stand-in for a policy: the action-surface node with the
/// smallest full name.
fn first_by_name(
    scenario: &Scenario,
    surface: impl IntoIterator<Item = AttackGraphNodeId>,
) -> Option<AttackGraphNodeId> {
    surface
        .into_iter()
        .min_by_key(|&id| full_name(scenario, id))
}

#[test]
fn load_scenario_reset_and_step() {
    let scenario = Scenario::load_from_file(scenario_path("traininglang_scenario.yml")).unwrap();
    let mut sim = Simulator::new(scenario.attack_graph.clone());
    let agents = scenario.flatten_agents(&mut StdRng::seed_from_u64(1));

    let state = sim.reset(&scenario.sim_settings, agents, 1).unwrap();

    // Entry points are compromised at start (default settings) and the
    // attacker has somewhere to go next.
    let attacker = &state.attackers["Attacker1"];
    assert!(attacker
        .performed_nodes
        .contains(&node(&scenario, "User:3:phishing")));
    assert!(attacker
        .performed_nodes
        .contains(&node(&scenario, "Host:0:connect")));
    assert!(!attacker.action_surface.is_empty());
    assert_eq!(state.attacker_is_terminated("Attacker1"), Some(false));
    assert!(state.defenders.contains_key("Defender1"));

    // Attacker steps on a node from its surface; the defender does nothing.
    let target = first_by_name(&scenario, attacker.action_surface.iter().copied()).unwrap();
    let actions = HashMap::from([("Attacker1".to_string(), vec![target])]);
    let outcome = sim.step(&actions).unwrap();

    // TTCs are disabled by default, so the attempt always succeeds.
    assert!(outcome.attackers["Attacker1"]
        .step_performed_nodes
        .contains(&target));
    let state = sim.state().unwrap();
    assert!(state.attackers["Attacker1"]
        .performed_nodes
        .contains(&target));
    assert_eq!(state.attackers["Attacker1"].iteration, 1);
    assert_eq!(state.defenders["Defender1"].iteration, 1);
    assert!(state.defenders["Defender1"]
        .compromised_nodes
        .contains(&target));
}

#[test]
fn defender_action_enables_defense() {
    let scenario = Scenario::load_from_file(scenario_path("traininglang_scenario.yml")).unwrap();
    let mut sim = Simulator::new(scenario.attack_graph.clone());
    let agents = scenario.flatten_agents(&mut StdRng::seed_from_u64(2));
    let state = sim.reset(&scenario.sim_settings, agents, 2).unwrap();

    let defense = first_by_name(
        &scenario,
        state.defenders["Defender1"].action_surface.iter().copied(),
    )
    .expect("defender has a defense to enable");
    let actions = HashMap::from([("Defender1".to_string(), vec![defense])]);
    let outcome = sim.step(&actions).unwrap();

    assert!(outcome.step_enabled_defenses.contains(&defense));
    let state = sim.state().unwrap();
    assert!(state.enabled_defenses.contains(&defense));
    assert!(state.defenders["Defender1"]
        .performed_nodes
        .contains(&defense));
    assert!(!state.defenders["Defender1"]
        .action_surface
        .contains(&defense));
}

#[test]
fn run_until_attacker_terminates() {
    // A full episode: the attacker always takes its first surface node by
    // name until it runs out of moves, with a step bound as a safety net.
    let scenario = Scenario::load_from_file(scenario_path("traininglang_scenario.yml")).unwrap();
    let mut sim = Simulator::new(scenario.attack_graph.clone());
    let agents = scenario.flatten_agents(&mut StdRng::seed_from_u64(3));
    sim.reset(&scenario.sim_settings, agents, 3).unwrap();

    let max_steps = scenario.attack_graph.borrow().nodes.len();
    let mut steps = 0;
    while sim.state().unwrap().attacker_is_terminated("Attacker1") == Some(false) {
        assert!(steps < max_steps, "attacker did not terminate");
        let surface = sim.state().unwrap().attackers["Attacker1"]
            .action_surface
            .clone();
        let target = first_by_name(&scenario, surface).unwrap();
        sim.step(&HashMap::from([("Attacker1".to_string(), vec![target])]))
            .unwrap();
        steps += 1;
    }
    assert!(steps > 0);
    assert!(sim.state().unwrap().defender_is_terminated());
}

/// The `Wiper-<id>:activate` node a fired wiperLang model effect adds.
fn wiper_activate(scenario: &Scenario) -> Option<AttackGraphNodeId> {
    let graph = scenario.attack_graph.borrow();
    let model = scenario.model.borrow();
    graph.nodes.keys().find(|&id| {
        let name = graph.full_name_of(id, Some(&model));
        name.starts_with("Wiper-") && name.ends_with(":activate")
    })
}

#[test]
fn dyna_scenario_reset_and_step() {
    // The DynaMalSimulator flavor, mirroring `test_dyna_mal_simulator.py::
    // test_inherited_query_methods_follow_graph_mutated_by_model_effects`
    // stepping on `InfectedDevice:infect` fires wiperLang's model effect,
    // which adds a Wiper asset (and its attack steps) at runtime through
    // the shared graph/model handles.
    let scenario = Scenario::load_from_file(scenario_path("wiper_scenario.yml")).unwrap();
    let nodes_before = scenario.attack_graph.borrow().nodes.len();
    let infect = node(&scenario, "InfectedDevice:infect");
    let mut sim = Simulator::new_dyna(scenario.attack_graph.clone(), scenario.model.clone());

    for seed in 0..2 {
        let agents = scenario.flatten_agents(&mut StdRng::seed_from_u64(seed));
        sim.reset(&scenario.sim_settings, agents, seed).unwrap();
        // Every reset restores the pristine model/graph.
        assert_eq!(scenario.attack_graph.borrow().nodes.len(), nodes_before);
        assert_eq!(wiper_activate(&scenario), None);

        let attacker = "WiperController".to_string();
        let outcome = sim
            .step(&HashMap::from([(attacker.clone(), vec![infect])]))
            .unwrap();
        assert!(!outcome.step_modification_record.is_empty());
        assert!(scenario.attack_graph.borrow().nodes.len() > nodes_before);

        // The new node is on the attacker's surface and can be compromised.
        // Asset ids keep counting across resets (Wiper-7, Wiper-8, ...), the
        // same as on the Python side, so look the new node up by pattern.
        let activate = wiper_activate(&scenario).expect("the effect added a Wiper asset");
        assert!(sim.state().unwrap().attackers[&attacker]
            .action_surface
            .contains(&activate));
        let outcome = sim
            .step(&HashMap::from([(attacker.clone(), vec![activate])]))
            .unwrap();
        assert!(outcome.attackers[&attacker]
            .step_performed_nodes
            .contains(&activate));
        assert!(sim.state().unwrap().attackers[&attacker]
            .performed_nodes
            .contains(&activate));
    }
}
