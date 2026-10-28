//! `SimState::snapshot_at` and the episode history (`PORTING_NOTES.md`
//! §14): the state rebuilt for an earlier iteration must equal the state
//! the simulator actually had at that iteration, on the static and the
//! dyna path (including steps whose model effects change the graph).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

use malsim_core::dyna_simulator::DynaSimulator;
use malsim_core::history::StateSnapshot;
use malsim_core::model_effects::ModEffectOp;
use malsim_core::scenario::{load_scenario_dict, Scenario};
use malsim_core::simulator::{SimState, Simulator};
use malsim_core::types::NodeSet;
use maltoolbox_attackgraph::AttackGraphNodeId;
use maltoolbox_language::graph::model_effect::ModelEffectType;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use slotmap::Key;

fn testdata(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/testdata")
        .join(path)
}

fn scenario(file: &str) -> Scenario {
    let dict = load_scenario_dict(testdata("scenarios").join(file)).unwrap();
    Scenario::from_dict(&dict).unwrap()
}

enum Driver {
    Static(Simulator),
    Dyna(DynaSimulator),
}

impl Driver {
    fn state(&self) -> &SimState {
        match self {
            Driver::Static(s) => s.state().unwrap(),
            Driver::Dyna(s) => s.state().unwrap(),
        }
    }

    fn step(&mut self, actions: &HashMap<String, Vec<AttackGraphNodeId>>) -> Vec<ModEffectOp> {
        match self {
            Driver::Static(s) => {
                s.step(actions).unwrap();
                Vec::new()
            }
            Driver::Dyna(s) => s.step(actions).unwrap().step_modification_record,
        }
    }
}

fn current(state: &SimState) -> StateSnapshot {
    state.snapshot_at(state.history().iteration()).unwrap()
}

fn sorted(ids: &NodeSet) -> Vec<AttackGraphNodeId> {
    let mut v: Vec<_> = ids.iter().copied().collect();
    v.sort();
    v
}

fn assert_same(label: &str, expected: &StateSnapshot, actual: &StateSnapshot) {
    assert_eq!(expected.iteration, actual.iteration, "{label}: iteration");
    assert_eq!(
        expected.graph_state, actual.graph_state,
        "{label}: graph_state"
    );
    assert_eq!(
        sorted(&expected.enabled_defenses),
        sorted(&actual.enabled_defenses),
        "{label}: enabled_defenses"
    );
    assert_eq!(
        expected.modification_record.len(),
        actual.modification_record.len(),
        "{label}: modification_record"
    );
    assert_eq!(expected.attackers, actual.attackers, "{label}: attackers");
    assert_eq!(expected.defenders, actual.defenders, "{label}: defenders");
}

/// What a run exercised, so the dyna test can check that the delta-based
/// `snapshot_at` was tested on the cases that matter.
#[derive(Debug, Default, Clone, Copy)]
struct Coverage {
    /// Model operations seen.
    modifications: usize,
    /// Steps with at least one additive operation.
    additive_steps: usize,
    /// Steps with at least one subtractive operation.
    subtractive_steps: usize,
    /// New node keys reusing the slot index of an earlier key.
    reused_slots: usize,
    /// Surviving nodes whose necessity changed between two captures.
    necessity_changes: usize,
    /// New keys in `pre_enabled_defenses`.
    new_pre_enabled: usize,
    /// New keys in `impossible_attack_steps`.
    new_impossible: usize,
    /// Steps where `enabled_defenses` grew by an action.
    action_enabled_steps: usize,
}

impl Coverage {
    fn add(&mut self, other: Coverage) {
        self.modifications += other.modifications;
        self.additive_steps += other.additive_steps;
        self.subtractive_steps += other.subtractive_steps;
        self.reused_slots += other.reused_slots;
        self.necessity_changes += other.necessity_changes;
        self.new_pre_enabled += other.new_pre_enabled;
        self.new_impossible += other.new_impossible;
        self.action_enabled_steps += other.action_enabled_steps;
    }
}

fn slot(id: AttackGraphNodeId) -> u32 {
    id.data().as_ffi() as u32
}

/// Runs `file` for up to `steps` steps with random actions,
/// capturing the live state after reset and after every step, then checks
/// `snapshot_at` against every capture. Returns what the run covered, so
/// dyna callers can check effects actually ran.
fn check(file: &str, dyna: bool, seed: u64, steps: u64) -> Coverage {
    let scenario = scenario(file);
    let mut driver = if dyna {
        let mut sim = DynaSimulator::new(scenario.attack_graph.clone(), scenario.model.clone());
        sim.restore_model().unwrap();
        Driver::Dyna(sim)
    } else {
        Driver::Static(Simulator::new(scenario.attack_graph.clone()))
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let agents = scenario.flatten_agents(&mut rng).unwrap();
    match &mut driver {
        Driver::Static(sim) => sim.reset(&scenario.sim_settings, agents, seed).map(|_| ()),
        Driver::Dyna(sim) => sim.reset(&scenario.sim_settings, agents, seed).map(|_| ()),
    }
    .unwrap();

    let mut captured = vec![current(driver.state())];
    let mut coverage = Coverage::default();
    let mut seen_keys: HashSet<AttackGraphNodeId> =
        captured[0].graph_state.necessity_per_node.keys().collect();
    let mut seen_slots: HashSet<u32> = seen_keys.iter().map(|&k| slot(k)).collect();
    for _ in 0..steps {
        if driver.state().done() {
            break;
        }
        let state = driver.state();
        let mut actions: HashMap<String, Vec<AttackGraphNodeId>> = HashMap::new();
        let surfaces = state
            .attackers
            .iter()
            .map(|(n, a)| (n, &a.action_surface))
            .chain(state.defenders.iter().map(|(n, d)| (n, &d.action_surface)));
        for (name, surface) in surfaces {
            // At most one action per agent, as the policies do: on the dyna
            // path an earlier action's model effect may remove a later one.
            let surface = sorted(surface);
            if !surface.is_empty() && rng.random_bool(0.8) {
                let pick = surface[rng.random_range(0..surface.len())];
                actions.insert(name.clone(), vec![pick]);
            }
        }
        let ops = driver.step(&actions);
        coverage.modifications += ops.len();
        let is_additive = |op: &&ModEffectOp| matches!(op.effect_type(), ModelEffectType::Additive);
        if ops.iter().any(|op| is_additive(&op)) {
            coverage.additive_steps += 1;
        }
        if ops.iter().any(|op| !is_additive(&op)) {
            coverage.subtractive_steps += 1;
        }
        captured.push(current(driver.state()));

        let prev = &captured[captured.len() - 2].graph_state;
        let prev_enabled = &captured[captured.len() - 2].enabled_defenses;
        let cur = &captured[captured.len() - 1];
        let now = &cur.graph_state;
        for key in now.necessity_per_node.keys() {
            if seen_keys.insert(key) && !seen_slots.insert(slot(key)) {
                coverage.reused_slots += 1;
            }
        }
        for (key, necessary) in &now.necessity_per_node {
            if prev
                .necessity_per_node
                .get(key)
                .is_some_and(|p| p != necessary)
            {
                coverage.necessity_changes += 1;
            }
        }
        coverage.new_pre_enabled += now
            .pre_enabled_defenses
            .difference(&prev.pre_enabled_defenses)
            .count();
        coverage.new_impossible += now
            .impossible_attack_steps
            .difference(&prev.impossible_attack_steps)
            .count();
        if cur
            .enabled_defenses
            .iter()
            .any(|d| !prev_enabled.contains(d) && !now.pre_enabled_defenses.contains(d))
        {
            coverage.action_enabled_steps += 1;
        }
    }

    let state = driver.state();
    let record = state.episode_record();
    for (k, expected) in captured.iter().enumerate() {
        let actual = state.snapshot_at(k as u64).unwrap();
        assert_same(&format!("{file} iteration {k}"), expected, &actual);
        let archived = record.snapshot_at(k as u64).unwrap();
        assert_same(
            &format!("{file} archived iteration {k}"),
            expected,
            &archived,
        );
    }
    assert!(state.snapshot_at(state.history().iteration() + 1).is_none());
    coverage
}

#[test]
fn static_snapshots_match_live_state() {
    check("traininglang_scenario.yml", false, 1, 30);
    check("traininglang_observability_scenario.yml", false, 2, 30);
}

#[test]
fn dyna_snapshots_match_live_state() {
    let mut files = vec![
        "actions_effects_scenario.yml".to_string(),
        "dynamic_remove_many_assoc_scenario.yml".to_string(),
        "dyna_defense_scenario.yml".to_string(),
    ];
    let root = testdata("scenarios/dynamal_example_scenarios");
    for dir in ["basic", "intermediate", "advanced"] {
        let mut entries: Vec<String> = std::fs::read_dir(root.join(dir))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        entries.sort();
        for name in entries {
            files.push(format!("dynamal_example_scenarios/{dir}/{name}"));
        }
    }
    let mut coverage = Coverage::default();
    for (i, file) in files.iter().enumerate() {
        for seed in 0..3 {
            coverage.add(check(file, true, i as u64 * 10 + seed, 60));
        }
    }
    eprintln!("dyna snapshot coverage: {coverage:?}");
    assert!(
        coverage.modifications > 0,
        "no model effect ran; test proves nothing"
    );
    assert!(coverage.additive_steps > 0, "no step added assets");
    assert!(coverage.subtractive_steps > 0, "no step removed assets");
    assert!(coverage.reused_slots > 0, "no node reused a freed slot");
    assert!(
        coverage.necessity_changes > 0,
        "no surviving node changed necessity"
    );
    assert!(coverage.new_pre_enabled > 0, "no new pre-enabled defense");
    assert!(coverage.new_impossible > 0, "no new impossible attack step");
    assert!(
        coverage.action_enabled_steps > 0,
        "no defense was enabled by an action"
    );
}

/// `recording(k)` lists each agent's newly performed nodes, requested
/// actions first, in request order.
#[test]
fn recording_lists_requested_actions_first() {
    let scenario = scenario("traininglang_scenario.yml");
    let mut sim = Simulator::new(scenario.attack_graph.clone());
    let agents = scenario
        .flatten_agents(&mut StdRng::seed_from_u64(0))
        .unwrap();
    sim.reset(&scenario.sim_settings, agents, 0).unwrap();
    assert!(sim.state().unwrap().history().recording(0).is_none());
    assert!(sim.state().unwrap().history().recording(1).is_none());

    let state = sim.state().unwrap();
    let (attacker, runtime) = state.attackers.iter().next().unwrap();
    let mut requested = sorted(&runtime.action_surface);
    requested.reverse();
    let attacker = attacker.clone();
    let actions = HashMap::from([(attacker.clone(), requested.clone())]);
    let outcome = sim.step(&actions).unwrap();

    let state = sim.state().unwrap();
    let recording: &BTreeMap<String, Vec<AttackGraphNodeId>> =
        state.history().recording(1).unwrap();
    let recorded = &recording[&attacker];
    let new_nodes = &outcome.attackers[&attacker].step_performed_nodes;
    let direct: Vec<AttackGraphNodeId> = requested
        .iter()
        .copied()
        .filter(|id| new_nodes.contains(id))
        .collect();
    assert!(!direct.is_empty());
    assert_eq!(&recorded[..direct.len()], direct.as_slice());
    assert_eq!(
        sorted(&recorded.iter().copied().collect()),
        sorted(new_nodes)
    );
    // Every agent gets an entry, also one that took no action.
    assert_eq!(
        recording.len(),
        state.attackers.len() + state.defenders.len()
    );
}
