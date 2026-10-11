//! Shared test-only fixtures for building real `AttackGraphNode`s with a
//! specific `step_type` - needed by `graph_utils.rs`/`necessity.rs` tests
//! since `AttackStepId` (unlike `AttackGraphNodeId`) can only be minted by
//! a real `maltoolbox_language::graph::LanguageGraph`. See
//! `PORTING_NOTES.md` §10's A3/A4 entries for why this dependency (test-
//! only, `maltoolbox-language` in `[dev-dependencies]`) was added at A4.
//!
//! Mirrors `tests/conftest.py::dummy_lang_graph` - compiles the same
//! `tests/testdata/langs/dummy_lang.mal` fixture the Python test suite
//! uses, so Rust and Python tests exercise the same language.

use std::path::PathBuf;
use std::rc::Rc;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_model::Model;

fn dummy_lang_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/testdata/langs/dummy_lang.mal")
}

fn wiper_lang_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/testdata/langs/wiperLang.mal")
}

fn wiper_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/testdata/models/wiper_model.yml")
}

/// Port of `tests/conftest.py::dummy_lang_graph`.
pub(crate) fn dummy_lang_graph() -> maltoolbox_language::graph::LanguageGraph {
    maltoolbox_language::from_mal_spec(dummy_lang_path()).expect("compile dummy_lang.mal")
}

/// An empty `AttackGraph` over the dummy language, mirroring
/// `AttackGraph(dummy_lang_graph)` in the Python tests.
pub(crate) fn dummy_graph() -> AttackGraph {
    AttackGraph::empty(Rc::new(dummy_lang_graph()))
}

/// Adds a node of the given `DummyAsset` attack step (by name, e.g.
/// `"DummyAndAttackStep"`) to `graph`, mirroring
/// `attack_graph.add_node(dummy_attack_steps[step_name])` in the Python
/// tests.
pub(crate) fn add_dummy_node(graph: &mut AttackGraph, step_name: &str) -> AttackGraphNodeId {
    let asset_id = graph
        .lang_graph
        .asset_id("DummyAsset")
        .expect("DummyAsset exists in dummy_lang.mal");
    let step_id = graph.lang_graph.assets[asset_id].attack_steps[step_name];
    graph
        .add_node(step_id, None, None, None, None, None, None)
        .expect("add dummy node")
}

/// Port of `tests/conftest.py::wiperLang_lang_graph`. Used by Phase B1/B2
/// tests (`PORTING_NOTES.md` §6) - wiperLang has model effects
/// (`additive_model_effects`/`subtractive_model_effects`), which
/// `dummy_lang.mal` deliberately doesn't declare.
pub(crate) fn wiper_lang_graph() -> maltoolbox_language::graph::LanguageGraph {
    maltoolbox_language::from_mal_spec(wiper_lang_path()).expect("compile wiperLang.mal")
}

/// Port of `tests/conftest.py::wiperLang_model` + `wiperLang_attack_graph`:
/// loads the same `tests/testdata/models/wiper_model.yml` fixture the
/// Python test suite uses (Internet/C2Server/InfectedDevice/InfectedData/
/// VulnerableDevice, no Wiper/Malware asset yet - those get added at
/// runtime by `InfectedDevice:infect`'s model effect) and builds the full
/// attack graph from it.
pub(crate) fn wiper_attack_graph() -> (AttackGraph, Model) {
    let lang_graph = Rc::new(wiper_lang_graph());
    let model = maltoolbox_model::load_from_file(wiper_model_path(), lang_graph)
        .expect("load wiper_model.yml");
    let graph = AttackGraph::from_model(&model).expect("build wiperLang attack graph");
    (graph, model)
}

/// An empty `AttackGraph` plus an assetless `Model`, both sharing one
/// `Rc<LanguageGraph>` - used by Phase B2 tests that need a real `&mut
/// Model` to pass through `execute_model_effects`/`partially_regenerate_graph`
/// but don't need any actual model effects to fire (`dummy_lang.mal`
/// declares none), unlike the wiperLang fixtures above.
pub(crate) fn dummy_graph_and_model() -> (AttackGraph, Model) {
    let lang_graph = Rc::new(dummy_lang_graph());
    let graph = AttackGraph::empty(lang_graph.clone());
    let model = Model::new("dummy_test_model", lang_graph);
    (graph, model)
}
