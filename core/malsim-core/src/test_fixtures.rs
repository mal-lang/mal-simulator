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

fn dummy_lang_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/testdata/langs/dummy_lang.mal")
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
