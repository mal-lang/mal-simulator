//! Reproducibility (R1): the same seed and inputs must give identical
//! results - RNG draws, step outcomes, logs, effect order.
//!
//! Each case builds the scenario from scratch twice IN THE SAME PROCESS
//! (so std `RandomState`s inside mal-toolbox's graph differ between the
//! two builds), runs it with seeded `RandomAgent` policies and compares a
//! transcript of everything observable per step. Everything is rendered
//! by node *name*, since slotmap keys of two separately generated graphs
//! need not coincide.

use std::collections::HashMap;
use std::path::PathBuf;

use malsim_core::agent_state::AgentState;
use malsim_core::dyna_simulator::DynaSimulator;
use malsim_core::event_logger::LogEntry;
use malsim_core::graph_state::TtcMode;
use malsim_core::model_effects::{AssetOp, AssocOp, ModEffectOp};
use malsim_core::rewards::agent_reward;
use malsim_core::run_simulation::decision_agents;
use malsim_core::scenario::{load_scenario_dict, Scenario};
use malsim_core::simulator::{SimState, Simulator, StepOutcome};
use maltoolbox_attackgraph::AttackGraphNodeId;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde_json::{json, Value};

fn testdata(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/testdata")
        .join(path)
}

/// The scenario file `name` with its `agents` replaced (if given).
fn build(name: &str, agents: Option<Value>) -> Scenario {
    let mut dict = load_scenario_dict(testdata("scenarios").join(name)).unwrap();
    if let Some(agents) = agents {
        dict.insert("agents".to_string(), agents);
    }
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
}

struct Names<'a>(&'a Scenario);

impl Names<'_> {
    fn node(&self, id: AttackGraphNodeId) -> String {
        let graph = self.0.attack_graph.borrow();
        if graph.nodes.contains_key(id) {
            graph.full_name_of(id, Some(&self.0.model.borrow()))
        } else {
            "<removed>".to_string()
        }
    }

    fn sorted(&self, ids: impl IntoIterator<Item = AttackGraphNodeId>) -> Vec<String> {
        let mut v: Vec<String> = ids.into_iter().map(|id| self.node(id)).collect();
        v.sort();
        v
    }

    fn ordered(&self, ids: impl IntoIterator<Item = AttackGraphNodeId>) -> Vec<String> {
        ids.into_iter().map(|id| self.node(id)).collect()
    }

    fn log(&self, l: &LogEntry) -> String {
        let mut context: Vec<(String, String)> = l
            .context
            .iter()
            .map(|(k, &v)| (k.clone(), self.node(v)))
            .collect();
        context.sort();
        format!(
            "t={} det=({},{}) trig={} ctx={:?} fp={}",
            l.timestep,
            self.node(l.detector_id.0),
            l.detector_id.1,
            self.node(l.trigger),
            context,
            l.false_positive
        )
    }

    fn mod_op(&self, op: &ModEffectOp) -> String {
        match op {
            ModEffectOp::Asset(AssetOp::Added { asset }) => format!("+asset {asset:?}"),
            ModEffectOp::Asset(AssetOp::Removed { asset, .. }) => format!("-asset {asset:?}"),
            ModEffectOp::Assoc(AssocOp::Added {
                left,
                field_name,
                right,
            }) => format!("+assoc {left:?} {field_name} {right:?}"),
            ModEffectOp::Assoc(AssocOp::Removed {
                left,
                field_name,
                right,
            }) => format!("-assoc {left:?} {field_name} {right:?}"),
        }
    }
}

/// Everything observable about the state after a reset or a step.
fn snapshot(
    scenario: &Scenario,
    state: &SimState,
    outcome: &StepOutcome,
    step: u64,
    out: &mut Vec<String>,
) {
    let names = Names(scenario);
    out.push(format!(
        "enabled_defenses {:?}",
        names.sorted(state.enabled_defenses.iter().copied())
    ));
    out.push(format!(
        "step_enabled_defenses {:?}",
        names.sorted(outcome.step_enabled_defenses.iter().copied())
    ));
    for (name, a) in &state.attackers {
        let o = &outcome.attackers[name];
        out.push(format!("[{name}] iteration {}", a.iteration));
        out.push(format!(
            "[{name}] step_performed {:?}",
            names.sorted(o.step_performed_nodes.iter().copied())
        ));
        out.push(format!(
            "[{name}] step_attempted {:?}",
            names.sorted(o.step_attempted_nodes.iter().copied())
        ));
        out.push(format!(
            "[{name}] performed {:?}",
            names.sorted(a.performed_nodes.iter().copied())
        ));
        out.push(format!(
            "[{name}] attempted {:?}",
            names.sorted(a.attempted_nodes.iter().copied())
        ));
        out.push(format!(
            "[{name}] action_surface {:?}",
            names.sorted(a.action_surface.iter().copied())
        ));
        let mut attempts: Vec<(String, u64)> = a
            .num_attempts
            .iter()
            .map(|(&n, &c)| (names.node(n), c))
            .collect();
        attempts.sort();
        out.push(format!("[{name}] num_attempts {attempts:?}"));
        let mut ttcs: Vec<(String, u64)> = a
            .ttc_values
            .iter()
            .map(|(&n, v)| (names.node(n), v.to_bits()))
            .collect();
        ttcs.sort();
        out.push(format!("[{name}] ttc_values {ttcs:?}"));
        out.push(format!(
            "[{name}] impossible {:?}",
            names.sorted(a.impossible_steps.iter().copied())
        ));
        history(&names, a, name, out);
    }
    for (name, d) in &state.defenders {
        let o = &outcome.defenders[name];
        out.push(format!("[{name}] iteration {}", d.iteration));
        out.push(format!(
            "[{name}] step_observed {:?}",
            names.sorted(o.step_observed_nodes.iter().copied())
        ));
        let logs: Vec<String> = o.step_logs.iter().map(|l| names.log(l)).collect();
        out.push(format!("[{name}] step_logs {logs:#?}"));
        let logs: Vec<String> = d.logs.iter().map(|l| names.log(l)).collect();
        out.push(format!("[{name}] logs {logs:#?}"));
        out.push(format!(
            "[{name}] observed {:?}",
            names.sorted(d.observed_nodes.iter().copied())
        ));
        out.push(format!(
            "[{name}] compromised {:?}",
            names.sorted(d.compromised_nodes.iter().copied())
        ));
        out.push(format!(
            "[{name}] performed {:?}",
            names.sorted(d.performed_nodes.iter().copied())
        ));
        out.push(format!(
            "[{name}] action_surface {:?}",
            names.sorted(d.action_surface.iter().copied())
        ));
    }

    // Rewards, each with an RNG seeded from the step.
    let graph = scenario.attack_graph.borrow();
    let model = scenario.model.borrow();
    for settings in &scenario.agent_settings {
        let agent_state = AgentState::new(settings, &graph, &model, state, outcome).unwrap();
        let mut rng = StdRng::seed_from_u64(step);
        out.push(format!(
            "[{}] reward {:?}",
            settings.name(),
            agent_reward(&agent_state, &mut rng).map(f64::to_bits)
        ));
    }
}

fn history(
    names: &Names<'_>,
    a: &malsim_core::simulator::AttackerRuntime,
    name: &str,
    out: &mut Vec<String>,
) {
    out.push(format!(
        "[{name}] performed_history {:?}",
        names.ordered(a.performed_history.iter().copied())
    ));
}

/// Runs `steps` steps (or until done) of a fresh instance of the scenario.
fn run(
    make: &dyn Fn(u64) -> Scenario,
    dyna: bool,
    multi: bool,
    seed: u64,
    steps: u64,
) -> Vec<String> {
    let scenario = make(seed);
    let mut driver = if dyna {
        Driver::Dyna(DynaSimulator::new(
            scenario.attack_graph.clone(),
            scenario.model.clone(),
        ))
    } else {
        Driver::Static(Simulator::new(scenario.attack_graph.clone()))
    };
    if let Driver::Dyna(sim) = &mut driver {
        sim.restore_model().unwrap();
    }
    let agents = scenario
        .flatten_agents(&mut StdRng::seed_from_u64(seed))
        .unwrap();
    match &mut driver {
        Driver::Static(sim) => sim.reset(&scenario.sim_settings, agents, seed).map(|_| ()),
        Driver::Dyna(sim) => sim.reset(&scenario.sim_settings, agents, seed).map(|_| ()),
    }
    .unwrap();
    let mut policies = decision_agents(&scenario).unwrap();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);

    let mut out = Vec::new();
    let mut outcome = StepOutcome::from_reset(driver.state());
    out.push("== reset".to_string());
    snapshot(&scenario, driver.state(), &outcome, 0, &mut out);

    for step in 1..=steps {
        if driver.state().done() {
            break;
        }
        let mut actions: HashMap<String, Vec<AttackGraphNodeId>> = HashMap::new();
        {
            let graph = scenario.attack_graph.borrow();
            let model = scenario.model.borrow();
            for settings in &scenario.agent_settings {
                let agent_state =
                    AgentState::new(settings, &graph, &model, driver.state(), &outcome).unwrap();
                if multi {
                    // Several actions per agent and step, so that the
                    // simulator handles multi-element sets.
                    let chosen: Vec<AttackGraphNodeId> = agent_state
                        .sorted_action_surface()
                        .into_iter()
                        .filter(|_| rng.random_bool(0.5))
                        .collect();
                    if !chosen.is_empty() {
                        actions.insert(settings.name().to_string(), chosen);
                    }
                    continue;
                }
                let Some(policy) = policies.get_mut(settings.name()) else {
                    continue;
                };
                if let Some(action) = policy.get_next_action(&agent_state).unwrap() {
                    actions.insert(settings.name().to_string(), vec![action]);
                }
            }
        }
        let names = Names(&scenario);
        out.push(format!("== step {step}"));
        let mut acts: Vec<(String, Vec<String>)> = actions
            .iter()
            .map(|(k, v)| (k.clone(), names.ordered(v.iter().copied())))
            .collect();
        acts.sort();
        out.push(format!("actions {acts:?}"));

        let result = match &mut driver {
            Driver::Static(sim) => sim.step(&actions).map(|o| (o, Vec::new())),
            Driver::Dyna(sim) => sim.step(&actions).map(|o| {
                let record: Vec<String> = o
                    .step_modification_record
                    .iter()
                    .map(|op| names.mod_op(op))
                    .collect();
                (o.step, record)
            }),
        };
        match result {
            Ok((o, record)) => {
                if dyna {
                    out.push(format!("modification_record {record:#?}"));
                }
                outcome = o;
            }
            // Several actions in one step can include a node that an
            // earlier action's model effect removed; that is an error the
            // simulator reports (deterministically).
            Err(e) if multi && dyna => {
                out.push(format!("step error: {e}").replace(|c: char| c.is_ascii_digit(), "#"));
                break;
            }
            Err(e) => panic!("step {step}: {e} (actions {acts:?})"),
        }
        snapshot(&scenario, driver.state(), &outcome, step, &mut out);
        if dyna {
            let graph = scenario.attack_graph.borrow();
            let model = scenario.model.borrow();
            let mut all: Vec<String> = graph
                .nodes
                .keys()
                .map(|id| graph.full_name_of(id, Some(&model)))
                .collect();
            all.sort();
            out.push(format!("graph_nodes {all:?}"));
        }
    }
    out
}

/// Runs the case twice for each of several seeds and asserts the
/// transcripts are identical. Returns the total number of steps taken.
fn assert_reproducible(label: &str, make: &dyn Fn(u64) -> Scenario, dyna: bool, steps: u64) {
    for multi in [false, true] {
        for seed in 0..SEEDS {
            assert_one(label, make, dyna, multi, seed, steps);
        }
    }
}

fn assert_one(
    label: &str,
    make: &dyn Fn(u64) -> Scenario,
    dyna: bool,
    multi: bool,
    seed: u64,
    steps: u64,
) {
    {
        let first = run(make, dyna, multi, seed, steps);
        let second = run(make, dyna, multi, seed, steps);
        if std::env::var("REPRO_DUMP").is_ok() {
            eprintln!(
                "##### {label} multi {multi} seed {seed}\n{}",
                first.join("\n")
            );
        }
        for (i, (a, b)) in first.iter().zip(&second).enumerate() {
            assert_eq!(
                a, b,
                "{label} multi {multi} seed {seed}: transcripts diverge at line {i}"
            );
        }
        assert_eq!(
            first.len(),
            second.len(),
            "{label} multi {multi} seed {seed}: lengths"
        );
    }
}

const SEEDS: u64 = 6;

fn random_agent(seed: u64, wait_prob: f64) -> Value {
    json!({"policy": "RandomAgent", "config": {"seed": seed, "wait_prob": wait_prob}})
}

fn with(mut base: Value, extra: Value) -> Value {
    base.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    base
}

/// ttc_lang: TTC distributions (incl. attacker overrides), two attackers
/// and a defender with false positive/negative rates.
fn ttc_scenario(mode: TtcMode, seed: u64) -> Scenario {
    let agents = json!({
        "AttackerA": with(random_agent(seed, 0.0), json!({
            "type": "attacker",
            "entry_points": ["ComputerA:easyAccess", "Net1:easyAccess"],
            "ttc_overrides": {"by_asset_type": {"Computer": {"easyConnect": "HardAndUncertain"}}},
            "rewards": {"by_asset_type": {"Computer": {"easyConnect": 3.0}}},
            "reward_mode": "SAMPLE_TTC",
        })),
        "AttackerB": with(random_agent(seed + 100, 0.2), json!({
            "type": "attacker",
            "entry_points": ["Net1:easyAccess"],
            "reward_mode": "SAMPLE_TTC",
        })),
        "Defender": with(random_agent(seed + 200, 0.5), json!({
            "type": "defender",
            "false_positive_rates": {"by_asset_type": {
                "Computer": {"easyAccess": 0.5, "easyConnect": 0.4},
                "Network": {"easyAccess": 0.3},
                "Software": {"easyScan": 0.5},
            }},
            "false_negative_rates": {"by_asset_type": {
                "Computer": {"easyAccess": 0.5, "easyConnect": 0.3},
                "Network": {"easyAccess": 0.4},
            }},
            "rewards": {"by_asset_type": {"Computer": {"easyAccess": 2.0}}},
        })),
    });
    let mut scenario = build("ttc_lang_scenario.yml", Some(agents));
    scenario.sim_settings.ttc_mode = mode;
    scenario.sim_settings.seed = Some(seed);
    scenario
}

#[test]
fn ttc_per_step_sample_is_reproducible() {
    assert_reproducible(
        "ttc per-step",
        &|seed| ttc_scenario(TtcMode::PerStepSample, seed),
        false,
        60,
    );
}

#[test]
fn ttc_pre_sample_is_reproducible() {
    assert_reproducible(
        "ttc pre-sample",
        &|seed| ttc_scenario(TtcMode::PreSample, seed),
        false,
        60,
    );
}

#[test]
fn ttc_effort_based_is_reproducible() {
    assert_reproducible(
        "ttc effort",
        &|seed| ttc_scenario(TtcMode::EffortBasedPerStepSample, seed),
        false,
        60,
    );
}

/// coreLang with false alerts, several attackers and Bernoullis.
#[test]
fn corelang_false_alerts_are_reproducible() {
    let make = |seed: u64| {
        let agents = json!({
            "A1": with(random_agent(seed, 0.0), json!({
                "type": "attacker", "entry_points": ["OS App:fullAccess"],
            })),
            "A2": with(random_agent(seed + 100, 0.1), json!({
                "type": "attacker", "entry_points": ["OS App:fullAccess", "Program 1:fullAccess"],
            })),
            "D": with(random_agent(seed + 200, 0.6), json!({
                "type": "defender",
                "false_positive_rates": {"by_asset_name": {
                    "OS App": {"fullAccess": 0.5, "attemptRead": 0.5, "read": 0.3},
                    "Program 1": {"fullAccess": 0.5, "attemptRead": 0.5},
                }},
                "false_negative_rates": {"by_asset_name": {
                    "OS App": {"fullAccess": 0.5, "read": 0.5},
                    "Program 1": {"fullAccess": 0.5},
                }},
            })),
        });
        let dict = json!({
            "lang_file": testdata("langs/org.mal-lang.coreLang-1.0.0.mar").to_string_lossy(),
            "model_file": testdata("models/simple_test_model.yml").to_string_lossy(),
            "agents": agents,
        });
        let mut scenario = Scenario::from_dict(dict.as_object().unwrap()).unwrap();
        scenario.sim_settings.ttc_mode = TtcMode::PerStepSample;
        scenario.sim_settings.seed = Some(seed);
        scenario
    };
    assert_reproducible("corelang", &make, false, 120);
}

/// Detectors and event logging (with false positives) plus model effects.
#[test]
fn detector_scenario_is_reproducible() {
    let make = |seed: u64| {
        let agents = json!({
            "Attacker": with(random_agent(seed, 0.0), json!({
                "type": "attacker", "entry_points": ["Computer:0:physicalAccess"],
            })),
            "Defender": with(random_agent(seed + 200, 0.3), json!({"type": "defender"})),
        });
        let mut scenario = build("detector_lang_scenario.yml", Some(agents));
        scenario.sim_settings.seed = Some(seed);
        scenario
    };
    assert_reproducible("detector (static)", &make, false, 40);
    assert_reproducible("detector (dyna)", &make, true, 40);
}

/// `scenario_file` with every agent replaced by a seeded `RandomAgent`
/// policy.
fn random_policies(scenario_file: &str, seed: u64) -> Scenario {
    let mut dict = load_scenario_dict(testdata("scenarios").join(scenario_file)).unwrap();
    if let Some(Value::Object(agents)) = dict.get_mut("agents") {
        for (i, agent) in agents.values_mut().enumerate() {
            let agent = agent.as_object_mut().unwrap();
            agent.remove("agent_class");
            agent.insert("policy".to_string(), json!("RandomAgent"));
            agent.insert(
                "config".to_string(),
                json!({"seed": seed * 10 + i as u64, "wait_prob": 0.1}),
            );
        }
    }
    let mut scenario = Scenario::from_dict(&dict).unwrap();
    scenario.sim_settings.seed = Some(seed);
    scenario
}

/// Effects of `actions_effects_scenario.yml`.
#[test]
fn actions_effects_is_reproducible() {
    let make = |seed: u64| {
        let mut scenario = random_policies("actions_effects_scenario.yml", seed);
        scenario.sim_settings.ttc_mode = TtcMode::PerStepSample;
        scenario
    };
    assert_reproducible("actions_effects", &make, true, 60);
}

/// Model effects that remove many assets/associations at once.
#[test]
fn remove_many_assoc_is_reproducible() {
    let make = |seed: u64| random_policies("dynamic_remove_many_assoc_scenario.yml", seed);
    assert_reproducible("remove_many_assoc", &make, true, 80);
}

/// Every dynamal example scenario, whose model effects add and remove
/// assets and associations in all sorts of ways.
#[test]
fn dynamal_examples_are_reproducible() {
    let root = testdata("scenarios/dynamal_example_scenarios");
    let mut files: Vec<String> = Vec::new();
    for dir in ["basic", "intermediate", "advanced"] {
        for entry in std::fs::read_dir(root.join(dir)).unwrap() {
            let path = entry.unwrap().path();
            files.push(format!(
                "dynamal_example_scenarios/{dir}/{}",
                path.file_name().unwrap().to_string_lossy()
            ));
        }
    }
    files.sort();
    assert!(files.len() > 20);
    for file in &files {
        let make = |seed: u64| random_policies(file, seed);
        assert_reproducible(file, &make, true, 40);
    }
}
