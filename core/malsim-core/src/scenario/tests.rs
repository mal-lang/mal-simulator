//! Scenario-level tests (Phase C4), ported from the scenario-building
//! cases of `tests/test_scenario.py`. Dict-level `extends`/validation
//! cases live in `loading.rs`; per-node rule semantics in
//! `node_property_rule.rs`/`flatten.rs`.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::SeedableRng;

use super::*;
use crate::settings::{RewardMode, TtcMode};

fn scenario_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/testdata/scenarios")
        .join(name)
}

fn load(name: &str) -> Scenario {
    Scenario::load_from_file(scenario_path(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn attacker<'a>(scenario: &'a Scenario, name: &str) -> &'a AttackerSettings<AttackGraphNodeId> {
    scenario
        .attacker_settings()
        .find(|a| a.name == name)
        .expect("attacker exists")
}

fn defender<'a>(scenario: &'a Scenario, name: &str) -> &'a DefenderSettings {
    scenario
        .defender_settings()
        .find(|d| d.name == name)
        .expect("defender exists")
}

fn node_id(scenario: &Scenario, full_name: &str) -> AttackGraphNodeId {
    scenario
        .attack_graph
        .borrow()
        .get_node_by_full_name(full_name)
        .unwrap()
}

/// Full names of the nodes a rule's `per_node` maps, with their values.
fn per_node_by_name<T: RuleValue>(
    scenario: &Scenario,
    rule: &NodePropertyRule<T>,
) -> BTreeMap<String, T> {
    let graph = scenario.attack_graph.borrow();
    let model = scenario.model.borrow();
    rule.per_node(&graph, &model)
        .into_iter()
        .map(|(id, v)| (graph.full_name_of(id, Some(&model)), v))
        .collect()
}

fn valued(rule: &NodePropertyRule<f64>, asset: &str, step: &str) -> f64 {
    match &rule.by_asset_name.as_ref().unwrap()[asset] {
        StepValues::Valued(values) => values[step],
        StepValues::Listed(_) => panic!("expected mapping form"),
    }
}

#[test]
fn load_scenario() {
    // Port of `test_load_scenario`.
    let scenario = load("simple_scenario.yml");

    let rewards = defender(&scenario, "Defender1").rewards.as_ref().unwrap();
    assert_eq!(valued(rewards, "OS App", "notPresent"), 2.0);
    assert_eq!(valued(rewards, "OS App", "supplyChainAuditing"), 7.0);
    assert_eq!(valued(rewards, "Program 1", "notPresent"), 3.0);
    assert_eq!(valued(rewards, "Program 1", "supplyChainAuditing"), 7.0);
    assert_eq!(
        valued(rewards, "SoftwareVulnerability:4", "notPresent"),
        4.0
    );
    assert_eq!(valued(rewards, "Data:5", "notPresent"), 1.0);
    assert_eq!(valued(rewards, "Credentials:6", "notPhishable"), 7.0);
    assert_eq!(valued(rewards, "Identity:11", "notPresent"), 3.5);

    let attacker1 = attacker(&scenario, "Attacker1");
    let entry_point = node_id(&scenario, "OS App:fullAccess");
    assert_eq!(
        attacker1.entry_points,
        EntryPoints::Single(BTreeSet::from([entry_point]))
    );

    assert_eq!(attacker1.policy.as_deref(), Some("BreadthFirstAttacker"));
    assert_eq!(
        defender(&scenario, "Defender1").policy.as_deref(),
        Some("PassiveAgent")
    );
    assert!(scenario
        .model_file
        .as_ref()
        .unwrap()
        .ends_with("models/simple_test_model.yml"));
    assert!(scenario
        .lang_file
        .ends_with("langs/org.mal-lang.coreLang-1.0.0.mar"));
}

#[test]
fn extend_scenario() {
    // Port of `test_extend_scenario`.
    let scenario = load("traininglang_scenario_extended.yml");
    let rewards = defender(&scenario, "Defender1").rewards.as_ref().unwrap();
    let per_node = per_node_by_name(&scenario, rewards);
    assert_eq!(per_node.len(), 7);
    assert!(per_node.values().all(|r| *r == 1.0));
    assert_eq!(scenario.agent_settings.len(), 2);
}

#[test]
fn extend_scenario_deeper() {
    // Port of `test_extend_scenario_deeper`.
    let scenario = load("sub/traininglang_scenario_extended_again.yml");
    let rewards = defender(&scenario, "Defender1").rewards.as_ref().unwrap();
    let per_node = per_node_by_name(&scenario, rewards);
    assert_eq!(per_node.len(), 7);
    assert!(per_node.values().all(|r| *r == 1.0));
    // Attacker1 is set to null in the final scenario.
    assert_eq!(scenario.agent_settings.len(), 1);
}

#[test]
fn extend_scenario_override_lang_model() {
    // Port of `test_extend_scenario_override_lang_model`.
    let scenario = load("sub/traininglang_scenario_override_lang_model.yml");
    let rewards = attacker(&scenario, "Attacker1").rewards.as_ref().unwrap();
    let graph = scenario.attack_graph.borrow();
    let model = scenario.model.borrow();
    let reward = |full_name: &str| {
        rewards
            .value(
                &graph.nodes[graph.get_node_by_full_name(full_name).unwrap()],
                &model,
            )
            .unwrap()
    };
    assert_eq!(reward("Host:0:notPresent"), 2.0);
    assert_eq!(reward("Host:0:access"), 4.0);
    assert_eq!(reward("Host:1:notPresent"), 7.0);
    assert_eq!(reward("Host:1:access"), 5.0);
    assert_eq!(reward("Data:2:notPresent"), 8.0);
    assert_eq!(reward("Data:2:read"), 5.0);
    assert_eq!(reward("Data:2:modify"), 10.0);
    drop((graph, model));
    assert_eq!(scenario.agent_settings.len(), 2);
}

#[test]
fn load_scenario_no_defender_agent() {
    // Port of `test_load_scenario_no_defender_agent` (policy kept as a
    // name: no agent is instantiated in Rust, §2.6).
    let scenario = load("no_defender_agent_scenario.yml");
    assert_eq!(scenario.defender_settings().count(), 0);
    assert_eq!(
        attacker(&scenario, "attacker1").policy.as_deref(),
        Some("BreadthFirstAttacker")
    );
}

#[test]
fn load_scenario_unknown_policy_is_carried_opaquely() {
    // Python's `test_load_scenario_agent_class_error` expects a
    // `LookupError` here; the Rust loader has no policy registry and
    // keeps the names as given (`PORTING_NOTES.md` §12, C3).
    let scenario = load("wrong_agent_classes_scenario.yml");
    assert_eq!(
        attacker(&scenario, "Attacker1").policy.as_deref(),
        Some("BananaAttacker")
    );
    assert_eq!(
        defender(&scenario, "Defender1").policy.as_deref(),
        Some("FishAttacker")
    );
}

#[test]
fn load_scenario_observability_given() {
    // Port of `test_load_scenario_observability_given`.
    let scenario = load("simple_filtered_observability_scenario.yml");
    let observable = defender(&scenario, "Defender1")
        .observable_steps
        .as_ref()
        .unwrap();
    let per_node = per_node_by_name(&scenario, observable);
    assert!(per_node.values().all(|v| *v));

    let graph = scenario.attack_graph.borrow();
    let model = scenario.model.borrow();
    let expected: BTreeSet<String> = graph
        .nodes
        .values()
        .filter_map(|node| {
            let asset = model.get_asset_by_id(node.model_asset?)?;
            let matches = (asset.asset_type == "Application"
                && (node.name == "fullAccess" || node.name == "supplyChainAuditing"))
                || (asset.name == "Identity:8" && node.name == "assume");
            matches.then(|| format!("{}:{}", asset.name, node.name))
        })
        .collect();
    assert!(!expected.is_empty());
    assert_eq!(per_node.keys().cloned().collect::<BTreeSet<_>>(), expected);
}

#[test]
fn load_scenario_observability_not_given() {
    // Port of `test_load_scenario_observability_not_given`.
    let scenario = load("simple_scenario.yml");
    assert!(defender(&scenario, "Defender1").observable_steps.is_none());
}

#[test]
fn load_scenario_false_positive_negative_rate() {
    // Port of `test_load_scenario_false_positive_negative_rate`.
    let scenario = load("traininglang_fp_fn_scenario.yml");
    let defender = defender(&scenario, "defender");
    let fpr = per_node_by_name(&scenario, defender.false_positive_rates.as_ref().unwrap());
    let fnr = per_node_by_name(&scenario, defender.false_negative_rates.as_ref().unwrap());

    assert_eq!(
        fpr,
        BTreeMap::from([
            ("Host:0:access".to_string(), 0.2),
            ("Host:1:access".to_string(), 0.3)
        ])
    );
    assert_eq!(
        fnr,
        BTreeMap::from([
            ("Host:0:access".to_string(), 0.4),
            ("Host:1:access".to_string(), 0.5),
            ("User:3:compromise".to_string(), 1.0),
        ])
    );
}

#[test]
fn scenario_advanced_agent_settings() {
    // Port of `test_scenario_advanced_agent_settings` (minus the
    // `to_dict` round trip, which the Rust loader doesn't implement).
    let scenario = load("traininglang_scenario_advanced_agent_settings.yml");
    assert!(scenario
        .lang_file
        .ends_with("langs/org.mal-lang.trainingLang-1.0.0.mar"));
    assert!(scenario
        .model_file
        .as_ref()
        .unwrap()
        .ends_with("models/traininglang_model.yml"));

    let attacker = attacker(&scenario, "Attacker1");
    let rewards = attacker.rewards.as_ref().unwrap();
    assert_eq!(valued(rewards, "Host:0", "access"), 4.0);
    assert_eq!(valued(rewards, "Host:1", "access"), 5.0);
    assert_eq!(valued(rewards, "Data:2", "read"), 5.0);
    assert_eq!(valued(rewards, "Data:2", "modify"), 10.0);
    assert_eq!(valued(rewards, "Host:0", "authenticate"), 1000.0);

    assert_eq!(
        attacker.entry_points,
        EntryPoints::Single(BTreeSet::from([
            node_id(&scenario, "User:3:phishing"),
            node_id(&scenario, "Host:0:connect"),
        ]))
    );
    assert_eq!(attacker.policy.as_deref(), Some("BreadthFirstAttacker"));
    assert_eq!(
        attacker.actionable_steps.as_ref().unwrap().by_asset_type,
        Some(BTreeMap::from([
            (
                "Host".to_string(),
                StepValues::Listed(vec!["authenticate".into(), "connect".into()])
            ),
            (
                "User".to_string(),
                StepValues::Listed(vec!["compromise".into()])
            ),
        ]))
    );

    let defender = defender(&scenario, "Defender1");
    assert_eq!(
        defender.actionable_steps.as_ref().unwrap().by_asset_type,
        Some(BTreeMap::from([(
            "Host".to_string(),
            StepValues::Listed(vec!["notPresent".into()])
        )]))
    );
    assert!(defender.observable_steps.is_some());
    let fnr = defender.false_negative_rates.as_ref().unwrap();
    let fpr = defender.false_positive_rates.as_ref().unwrap();
    let by_type = |rule: &NodePropertyRule<f64>, asset: &str, step: &str| match &rule
        .by_asset_type
        .as_ref()
        .unwrap()[asset]
    {
        StepValues::Valued(values) => values[step],
        StepValues::Listed(_) => panic!("expected mapping form"),
    };
    assert_eq!(by_type(fnr, "Host", "access"), 0.5);
    assert_eq!(by_type(fpr, "Host", "connect"), 0.5);
    assert_eq!(
        valued(defender.rewards.as_ref().unwrap(), "Host:0", "notPresent"),
        100.0
    );
}

#[test]
fn inline_model_and_ttc_overrides() {
    let scenario = load("ttc_lang_scenario_override_ttcs.yml");
    assert!(scenario.model_file.is_none());
    let bad = attacker(&scenario, "BadAttacker");
    let per_node = per_node_by_name(&scenario, bad.ttc_dists.as_ref().unwrap());
    // `by_asset_type: Computer: easyConnect` - one node per Computer asset.
    assert_eq!(
        per_node.keys().cloned().collect::<BTreeSet<_>>(),
        ["ComputerA", "ComputerB", "ComputerC", "ComputerD"]
            .iter()
            .map(|c| format!("{c}:easyConnect"))
            .collect()
    );
    assert!(attacker(&scenario, "GoodAttacker").ttc_dists.is_none());
}

#[test]
fn sim_settings_are_parsed() {
    let mut dict = load_scenario_dict(scenario_path("simple_scenario.yml")).unwrap();
    dict.insert(
        "sim_settings".into(),
        serde_json::json!({"ttc_mode": "EXPECTED_VALUE", "seed": 3, "attack_surface": {"skip_unnecessary": true}}),
    );
    let scenario = Scenario::from_dict(&dict).unwrap();
    assert_eq!(scenario.sim_settings.ttc_mode, TtcMode::ExpectedValue);
    assert_eq!(scenario.sim_settings.seed, Some(3));
    assert!(scenario.sim_settings.attack_surface.skip_unnecessary);

    let default = load("simple_scenario.yml");
    assert_eq!(default.sim_settings, MalSimulatorSettings::default());
    assert_eq!(
        attacker(&default, "Attacker1").reward_mode,
        RewardMode::Cumulative
    );
}

#[test]
fn unknown_entry_point_is_an_error() {
    let mut dict = load_scenario_dict(scenario_path("simple_scenario.yml")).unwrap();
    dict["agents"]["Attacker1"]["entry_points"] = serde_json::json!(["Nope:fullAccess"]);
    assert!(matches!(
        Scenario::from_dict(&dict),
        Err(ScenarioError::AgentSettings(
            AgentSettingsError::UnknownNode { .. }
        ))
    ));
}

#[test]
fn git_lang_url_is_reported() {
    assert!(matches!(
        Scenario::load_from_file(scenario_path("simple_scenario_git_url.yml")),
        Err(ScenarioError::Language { .. })
    ));
}

#[test]
fn flatten_agents_orders_attackers_first_and_resolves_ids() {
    let scenario = load("simple_scenario.yml");
    let flat = scenario
        .flatten_agents(&mut StdRng::seed_from_u64(0))
        .unwrap();
    let names: Vec<&str> = flat.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, vec!["Attacker1", "Defender1"]);
    match &flat[0].1 {
        FlatAgentSettings::Attacker(a) => {
            assert_eq!(
                a.entry_points,
                HashSet::from([node_id(&scenario, "OS App:fullAccess")])
            );
        }
        FlatAgentSettings::Defender(_) => panic!("expected attacker first"),
    }
    assert!(matches!(flat[1].1, FlatAgentSettings::Defender(_)));
}
