//! `update_attack_surface` must always give exactly what a full
//! `get_attack_surface` rebuild gives. Two sweeps check that:
//!
//! - `simulated_surfaces_match_full_rebuild`: every test scenario, under
//!   many combinations of settings and seeds, played by random attackers
//!   and defenders on the plain and dyna simulators. Every attacker's
//!   surface is compared with a full rebuild after every step.
//! - `random_updates_match_full_rebuild`: `update_attack_surface` called
//!   directly on real graphs with random performed nodes and defenses -
//!   also ones no attacker could reach - and random actionability,
//!   necessity and impossible steps.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use malsim_core::attack_surface::{get_attack_surface, update_attack_surface};
use malsim_core::dyna_simulator::DynaSimulator;
use malsim_core::graph_state::{compute_initial_graph_state, TtcMode};
use malsim_core::graph_utils::sort_by_node_id;
use malsim_core::scenario::Scenario;
use malsim_core::settings::{FlatAgentSettings, FlatDefenderSettings, MalSimulatorSettings};
use malsim_core::simulator::{SimState, Simulator};
use malsim_core::types::{NodeDenseMap, NodeSet};
use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_language::graph::attack_step::AttackStepType;
use rand::rngs::StdRng;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{RngExt, SeedableRng};

/// Coverage counters for `simulated_surfaces_match_full_rebuild`, so a
/// pass can't hide a sweep that never hit the interesting cases.
static EPISODES: AtomicUsize = AtomicUsize::new(0);
static RESET_FAILURES: AtomicUsize = AtomicUsize::new(0);
static STEPS: AtomicUsize = AtomicUsize::new(0);
static STEPS_WITH_NEW_DEFENSES: AtomicUsize = AtomicUsize::new(0);
static STEPS_WITH_SURFACE_REMOVALS: AtomicUsize = AtomicUsize::new(0);
static STEPS_WITH_MODEL_EFFECTS: AtomicUsize = AtomicUsize::new(0);

fn bump(counter: &AtomicUsize) {
    counter.fetch_add(1, Ordering::Relaxed);
}

fn scenario_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/testdata/scenarios")
}

fn scenario_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            scenario_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "yml" || e == "yaml") {
            out.push(path);
        }
    }
}

/// Every test scenario that loads offline (some need the network or are
/// deliberately broken), with whether it is a dyna-MAL scenario.
fn scenarios() -> Vec<(PathBuf, bool)> {
    let mut files = Vec::new();
    scenario_files(&scenario_dir(), &mut files);
    let loaded: Vec<(PathBuf, bool)> = files
        .into_iter()
        .filter(|path| !path.to_string_lossy().contains("git_url"))
        .filter(|path| Scenario::load_from_file(path).is_ok())
        .map(|path| {
            let name = path.to_string_lossy();
            let dyna = name.contains("dynamal") || name.contains("dynamic_");
            (path, dyna)
        })
        .collect();
    loaded
}

fn sorted(
    graph: &AttackGraph,
    ids: impl IntoIterator<Item = AttackGraphNodeId>,
) -> Vec<AttackGraphNodeId> {
    let mut ids: Vec<AttackGraphNodeId> = ids.into_iter().collect();
    sort_by_node_id(graph, &mut ids);
    ids
}

/// Asserts every attacker's surface equals a full rebuild.
fn check_surfaces(graph: &AttackGraph, state: &SimState, label: &str, step: u64) {
    for (name, runtime) in &state.attackers {
        let mut expected = get_attack_surface(
            graph,
            state.settings.attack_surface.skip_compromised,
            state.settings.attack_surface.skip_unnecessary,
            runtime.actionable_steps.as_ref(),
            &runtime.performed_nodes,
            None,
            &state.graph_state.impossible_attack_steps,
            &state.enabled_defenses,
            &state.graph_state.necessity_per_node,
        )
        .unwrap();
        if runtime.iteration == 0 && !state.settings.compromise_entrypoints_at_start {
            expected.extend(runtime.entry_points.iter().copied());
        }
        assert_eq!(
            runtime.action_surface, expected,
            "{label}: attacker {name} at step {step}"
        );
    }
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

    /// Steps; returns whether a model effect ran.
    fn step(&mut self, actions: &HashMap<String, Vec<AttackGraphNodeId>>) -> bool {
        match self {
            Driver::Static(s) => {
                s.step(actions).unwrap();
                false
            }
            Driver::Dyna(s) => !s.step(actions).unwrap().step_modification_record.is_empty(),
        }
    }
}

/// Random actions: attackers mostly pick from their surface (one or
/// several nodes), sometimes an arbitrary node or nothing; defenders
/// sometimes enable one or more defenses from their surface. On dyna
/// runs every agent picks at most one node: a model effect of one chosen
/// node can delete another, which the simulator rejects.
fn random_actions(
    graph: &AttackGraph,
    state: &SimState,
    rng: &mut StdRng,
    dyna: bool,
) -> HashMap<String, Vec<AttackGraphNodeId>> {
    let all_nodes = sorted(graph, graph.nodes.keys());
    let mut actions = HashMap::new();
    for (name, runtime) in &state.attackers {
        let surface = sorted(graph, runtime.action_surface.iter().copied());
        let mut chosen = Vec::new();
        let roll: f64 = rng.random();
        if roll < 0.75 {
            let count = if !dyna && rng.random::<f64>() < 0.2 {
                3
            } else {
                1
            };
            chosen.extend(surface.sample(rng, count).copied());
        } else if roll < 0.9 {
            chosen.extend(all_nodes.choose(rng).copied());
        }
        actions.insert(name.clone(), chosen);
    }
    for (name, runtime) in &state.defenders {
        let surface = sorted(graph, runtime.action_surface.iter().copied());
        let mut chosen = Vec::new();
        let roll: f64 = rng.random();
        if roll < 0.25 {
            chosen.extend(surface.choose(rng).copied());
        } else if roll < 0.3 && !dyna {
            chosen.extend(surface.sample(rng, 3).copied());
        }
        actions.insert(name.clone(), chosen);
    }
    actions
}

fn run_case(path: &Path, dyna: bool, settings: MalSimulatorSettings, seed: u64, steps: u64) {
    let label = format!(
        "{} dyna={dyna} seed={seed} {settings:?}",
        path.file_name().unwrap().to_string_lossy()
    );
    let scenario = Scenario::load_from_file(path).unwrap();
    let mut driver = if dyna {
        let mut sim = DynaSimulator::new(scenario.attack_graph.clone(), scenario.model.clone());
        sim.restore_model().unwrap();
        Driver::Dyna(sim)
    } else {
        Driver::Static(Simulator::new(scenario.attack_graph.clone()))
    };
    let mut agents = scenario
        .flatten_agents(&mut StdRng::seed_from_u64(seed))
        .unwrap();
    // Without a defender no defense is ever enabled mid-episode.
    if !agents
        .iter()
        .any(|(_, a)| matches!(a, FlatAgentSettings::Defender(_)))
    {
        agents.push((
            "TestDefender".to_string(),
            FlatAgentSettings::Defender(FlatDefenderSettings {
                actionable_steps: None,
                observable_steps: None,
                false_positive_rates: None,
                false_negative_rates: None,
            }),
        ));
    }
    let reset = match &mut driver {
        Driver::Static(sim) => sim.reset(&settings, agents, seed).map(|_| ()),
        Driver::Dyna(sim) => sim.reset(&settings, agents, seed).map(|_| ()),
    };
    if reset.is_err() {
        // Some settings combinations are invalid for some scenarios.
        bump(&RESET_FAILURES);
        return;
    }
    bump(&EPISODES);

    let graph_rc = scenario.attack_graph.clone();
    check_surfaces(&graph_rc.borrow(), driver.state(), &label, 0);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xa77ac);
    for step in 1..=steps {
        if driver.state().done() {
            break;
        }
        let actions = random_actions(&graph_rc.borrow(), driver.state(), &mut rng, dyna);
        let defenses_before = driver.state().enabled_defenses.len();
        let surfaces_before: Vec<NodeSet> = driver
            .state()
            .attackers
            .values()
            .map(|a| a.action_surface.clone())
            .collect();
        let effects = driver.step(&actions);
        bump(&STEPS);
        if effects {
            bump(&STEPS_WITH_MODEL_EFFECTS);
        }
        if driver.state().enabled_defenses.len() > defenses_before {
            bump(&STEPS_WITH_NEW_DEFENSES);
        }
        if driver
            .state()
            .attackers
            .values()
            .zip(&surfaces_before)
            .any(|(a, before)| before.iter().any(|n| !a.action_surface.contains(n)))
        {
            bump(&STEPS_WITH_SURFACE_REMOVALS);
        }
        check_surfaces(&graph_rc.borrow(), driver.state(), &label, step);
    }
}

#[test]
fn simulated_surfaces_match_full_rebuild() {
    let scenarios = scenarios();
    let mut cases = 0;
    for (path, dyna) in &scenarios {
        let base = Scenario::load_from_file(path).unwrap().sim_settings;
        for bits in 0..32u32 {
            let mut settings = base;
            settings.attack_surface.skip_compromised = bits & 1 != 0;
            settings.attack_surface.skip_unnecessary = bits & 2 != 0;
            settings.compromise_entrypoints_at_start = bits & 4 != 0;
            settings.ttc_mode = if bits & 8 != 0 {
                TtcMode::PerStepSample
            } else {
                TtcMode::Disabled
            };
            settings.run_defense_step_bernoullis = bits & 16 != 0;
            settings.run_attack_step_bernoullis = bits & 16 != 0;
            for seed in 0..3 {
                // Every scenario on the plain simulator; dyna scenarios
                // also on the dyna one.
                run_case(path, false, settings, seed, 100);
                if *dyna {
                    run_case(path, true, settings, seed, 100);
                }
                cases += 1;
            }
        }
    }
    let get = |c: &AtomicUsize| c.load(Ordering::Relaxed);
    eprintln!(
        "{} scenarios, {cases} cases: {} episodes ({} invalid resets), {} steps, \
         {} with new defenses, {} with surface removals, {} with model effects",
        scenarios.len(),
        get(&EPISODES),
        get(&RESET_FAILURES),
        get(&STEPS),
        get(&STEPS_WITH_NEW_DEFENSES),
        get(&STEPS_WITH_SURFACE_REMOVALS),
        get(&STEPS_WITH_MODEL_EFFECTS),
    );
    assert!(get(&STEPS_WITH_NEW_DEFENSES) > 1000);
    assert!(get(&STEPS_WITH_SURFACE_REMOVALS) > 1000);
    assert!(get(&STEPS_WITH_MODEL_EFFECTS) > 100);
}

/// Random sets over the graph, built from node-id-sorted lists so the test
/// is reproducible.
fn random_subset(nodes: &[AttackGraphNodeId], p: f64, rng: &mut StdRng) -> NodeSet {
    nodes
        .iter()
        .copied()
        .filter(|_| rng.random::<f64>() < p)
        .collect()
}

fn random_update_case(path: &Path, seed: u64) {
    let label = format!(
        "{} seed={seed}",
        path.file_name().unwrap().to_string_lossy()
    );
    let scenario = Scenario::load_from_file(path).unwrap();
    let graph = scenario.attack_graph.borrow();
    let mut rng = StdRng::seed_from_u64(seed);
    let nodes = sorted(&graph, graph.nodes.keys());
    if nodes.is_empty() {
        return;
    }
    let defenses: Vec<AttackGraphNodeId> = nodes
        .iter()
        .copied()
        .filter(|&n| graph.nodes[n].step_type == AttackStepType::Defense)
        .collect();

    let realistic = rng.random::<f64>() < 0.5;
    let (impossible, necessity): (NodeSet, NodeDenseMap<bool>) = if realistic {
        let graph_state =
            compute_initial_graph_state(&graph, TtcMode::Disabled, true, true, &mut rng).unwrap();
        (
            graph_state.impossible_attack_steps,
            graph_state.necessity_per_node,
        )
    } else {
        (
            random_subset(&nodes, 0.1, &mut rng),
            nodes
                .iter()
                .map(|&n| (n, rng.random::<f64>() < 0.8))
                .collect(),
        )
    };
    let actionable = (rng.random::<f64>() < 0.3).then(|| random_subset(&nodes, 0.7, &mut rng));
    let actionable = actionable.as_ref();

    for (skip_compromised, skip_unnecessary) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let mut performed = random_subset(&nodes, 0.02, &mut rng);
        let mut enabled = random_subset(&defenses, 0.1, &mut rng);
        let full = |performed: &NodeSet, enabled: &NodeSet| {
            get_attack_surface(
                &graph,
                skip_compromised,
                skip_unnecessary,
                actionable,
                performed,
                None,
                &impossible,
                enabled,
                &necessity,
            )
            .unwrap()
        };
        let mut surface = full(&performed, &enabled);
        for round in 0..40 {
            // Mostly nodes from the surface, like an attacker would pick,
            // plus some arbitrary ones (entry points, effects, defenses,
            // already performed nodes, ...).
            let surface_nodes = sorted(&graph, surface.iter().copied());
            let count = rng.random_range(0..4);
            let mut new_performed: Vec<AttackGraphNodeId> =
                surface_nodes.sample(&mut rng, count).copied().collect();
            if rng.random::<f64>() < 0.3 {
                let count = rng.random_range(1..4);
                new_performed.extend(nodes.sample(&mut rng, count));
            }
            new_performed.shuffle(&mut rng);
            let new_defenses: Vec<AttackGraphNodeId> = if rng.random::<f64>() < 0.3 {
                let count = rng.random_range(1..3);
                defenses.sample(&mut rng, count).copied().collect()
            } else {
                Vec::new()
            };
            performed.extend(new_performed.iter().copied());
            enabled.extend(new_defenses.iter().copied());

            let before = surface.clone();
            let (added, removed) = update_attack_surface(
                &graph,
                &mut surface,
                skip_compromised,
                skip_unnecessary,
                actionable,
                &performed,
                new_performed.iter().copied(),
                &impossible,
                &enabled,
                new_defenses.iter().copied(),
                &necessity,
            )
            .unwrap();
            let expected = full(&performed, &enabled);
            let ctx = format!(
                "{label} realistic={realistic} skip_compromised={skip_compromised} \
                 skip_unnecessary={skip_unnecessary} round={round}"
            );
            assert_eq!(surface, expected, "{ctx}: surface");
            let expected_added: NodeSet = expected.difference(&before).copied().collect();
            let expected_removed: NodeSet = before.difference(&expected).copied().collect();
            assert_eq!(added, expected_added, "{ctx}: added");
            assert_eq!(removed, expected_removed, "{ctx}: removed");
        }
    }
}

#[test]
fn random_updates_match_full_rebuild() {
    let scenarios = scenarios();
    for (path, _) in &scenarios {
        for seed in 0..25 {
            random_update_case(path, seed);
        }
    }
}
