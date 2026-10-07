//! Rust port of `python/malsim/mal_simulator/event_logger.py` - see
//! `PORTING_NOTES.md` §5 Phase A6.
//!
//! Per A6's own scope note, `LogEntry`'s Python `detector: Detector`/
//! `trigger: AttackGraphNode` fields become id-based here: `trigger` is an
//! `AttackGraphNodeId`, and a detector is identified by a `DetectorId` -
//! `(AttackGraphNodeId, String)`, the id of the node it's attached to plus
//! its label key in that node's `detectors: HashMap<String, Detector>` map
//! (mal-toolbox's `Detector` type has no id of its own - confirmed by
//! reading `maltoolbox_attackgraph::generate::create_detectors`, which
//! always sets `Detector.node` to the very node whose `detectors` map it's
//! inserted into, so `(node_id, label)` is a sound, resolvable identity).
//! `context: HashMap<String, AttackGraphNodeId>` is the id-based
//! equivalent of Python's `dict[str, AttackGraphNode]`. Resolving a
//! `LogEntry` back to the Python `LogEntry` dataclass (ids -> real
//! `Detector`/`AttackGraphNode` objects) is Python's job when crossing the
//! FFI boundary (§3), not this module's - see `PORTING_NOTES.md` §10.

use std::collections::HashMap;
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId, Detector};
use rand::{Rng, RngExt};

/// Identifies a `Detector` within the graph - see module docs for why this
/// (rather than some id native to `Detector` itself) is the right shape.
pub type DetectorId = (AttackGraphNodeId, String);

/// Id-based port of the `LogEntry` Python dataclass - see module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct LogEntry {
    pub timestep: u64,
    pub detector_id: DetectorId,
    pub trigger: AttackGraphNodeId,
    pub context: HashMap<String, AttackGraphNodeId>,
    pub false_positive: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventLoggerError {
    /// Mirrors Python's bare `assert attack_step.model_asset is not None`
    /// in `collect_logs`.
    MissingModelAsset(AttackGraphNodeId),
    /// Mirrors Python's unhandled `StopIteration` out of `get_context`'s
    /// `next(... if node in previous_compromised_nodes)` when none of a
    /// context label's potential nodes were previously compromised.
    NoCompromisedContextNode(AttackGraphNodeId, String),
    /// Mirrors Python's unhandled `StopIteration` out of
    /// `get_random_context`'s `next(node for node in
    /// potential_context_nodes)` when a context label's potential-node set
    /// is empty.
    EmptyContextCandidates(AttackGraphNodeId, String),
}

impl fmt::Display for EventLoggerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventLoggerError::MissingModelAsset(id) => {
                write!(f, "attack step {id:?} has no model asset")
            }
            EventLoggerError::NoCompromisedContextNode(id, label) => write!(
                f,
                "detector on {id:?} has no previously-compromised candidate for context label \"{label}\""
            ),
            EventLoggerError::EmptyContextCandidates(id, label) => write!(
                f,
                "detector on {id:?} has no candidate nodes at all for context label \"{label}\""
            ),
        }
    }
}

impl std::error::Error for EventLoggerError {}

/// A rate is "truthy" in Python exactly when it's `Some` and non-zero
/// (`if detector.fprate:`/`if detector.tprate:`/`not detector.tprate` -
/// `None` and `0.0` are both falsy, matching every other numeric Python
/// truthiness check, e.g. a negative rate is still truthy).
fn rate_is_truthy(rate: Option<f64>) -> bool {
    rate.is_some_and(|r| r != 0.0)
}

/// Flattened `(node_id, label, detector)` view over every detector in the
/// graph - the graph-independent-of-Python equivalent of Python's
/// `attack_graph.detectors` list (itself seeded, in the PyO3 layer, from
/// exactly this same per-node walk - see `PORTING_NOTES.md` §10).
fn all_detectors(
    graph: &AttackGraph,
) -> impl Iterator<Item = (AttackGraphNodeId, &String, &Detector)> {
    graph.nodes.iter().flat_map(|(node_id, node)| {
        node.detectors
            .iter()
            .map(move |(label, detector)| (node_id, label, detector))
    })
}

/// Port of `get_context`.
fn get_context(
    detector: &Detector,
    previous_compromised_nodes: &std::collections::HashSet<AttackGraphNodeId>,
) -> Result<HashMap<String, AttackGraphNodeId>, EventLoggerError> {
    let mut context_nodes = HashMap::new();
    for (label, potential_context_nodes) in &detector.potential_context {
        let node_id = potential_context_nodes
            .iter()
            .copied()
            .find(|id| previous_compromised_nodes.contains(id))
            .ok_or_else(|| {
                EventLoggerError::NoCompromisedContextNode(detector.node, label.clone())
            })?;
        context_nodes.insert(label.clone(), node_id);
    }
    Ok(context_nodes)
}

/// Port of `get_random_context`.
fn get_random_context(
    detector: &Detector,
) -> Result<HashMap<String, AttackGraphNodeId>, EventLoggerError> {
    let mut context_nodes = HashMap::new();
    for (label, potential_context_nodes) in &detector.potential_context {
        let node_id = potential_context_nodes
            .iter()
            .copied()
            .next()
            .ok_or_else(|| {
                EventLoggerError::EmptyContextCandidates(detector.node, label.clone())
            })?;
        context_nodes.insert(label.clone(), node_id);
    }
    Ok(context_nodes)
}

/// Port of `collect_false_positives`. `get_random_context` is only called
/// (and can only error) for a detector that actually fires, mirroring
/// Python calling it inside the `if` block, not before.
pub fn collect_false_positives(
    iteration: u64,
    graph: &AttackGraph,
    rng: &mut impl Rng,
) -> Result<Vec<LogEntry>, EventLoggerError> {
    let mut logs = Vec::new();
    for (node_id, label, detector) in all_detectors(graph) {
        let fires =
            rate_is_truthy(detector.fprate) && detector.fprate.unwrap() >= rng.random::<f64>();
        if fires {
            logs.push(LogEntry {
                timestep: iteration,
                detector_id: (node_id, label.clone()),
                trigger: detector.node,
                context: get_random_context(detector)?,
                false_positive: true,
            });
        }
    }
    Ok(logs)
}

/// Port of `collect_logs`. `get_context` is called unconditionally for
/// every detector, before the true-positive roll, mirroring Python's
/// `labeled_steps = get_context(...)` being evaluated before the
/// `if not detector.tprate or ...:` check.
pub fn collect_logs(
    iteration: u64,
    graph: &AttackGraph,
    step_compromised_nodes: impl IntoIterator<Item = AttackGraphNodeId>,
    previous_compromised_nodes: &std::collections::HashSet<AttackGraphNodeId>,
    rng: &mut impl Rng,
) -> Result<Vec<LogEntry>, EventLoggerError> {
    let mut logs = Vec::new();
    for attack_step_id in step_compromised_nodes {
        let node = &graph.nodes[attack_step_id];
        if node.model_asset.is_none() {
            return Err(EventLoggerError::MissingModelAsset(attack_step_id));
        }
        for (label, detector) in &node.detectors {
            let context = get_context(detector, previous_compromised_nodes)?;
            let fires =
                !rate_is_truthy(detector.tprate) || detector.tprate.unwrap() >= rng.random::<f64>();
            if fires {
                logs.push(LogEntry {
                    timestep: iteration,
                    detector_id: (attack_step_id, label.clone()),
                    trigger: attack_step_id,
                    context,
                    false_positive: false,
                });
            }
        }
    }
    Ok(logs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashSet;

    fn add_detector(
        graph: &mut AttackGraph,
        node_id: AttackGraphNodeId,
        label: &str,
        tprate: Option<f64>,
        fprate: Option<f64>,
        potential_context: HashMap<String, HashSet<AttackGraphNodeId>>,
    ) {
        graph.nodes[node_id].detectors.insert(
            label.to_string(),
            Detector {
                name: Some(label.to_string()),
                node: node_id,
                potential_context,
                tprate,
                fprate,
            },
        );
    }

    #[test]
    fn collect_logs_missing_model_asset_errors() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        // `add_dummy_node` never sets `model_asset` - mirrors Python's
        // `assert attack_step.model_asset is not None`.
        let mut rng = StdRng::seed_from_u64(1);
        let result = collect_logs(0, &graph, [step], &HashSet::new(), &mut rng);
        assert_eq!(result, Err(EventLoggerError::MissingModelAsset(step)));
    }

    #[test]
    fn collect_logs_no_tprate_always_true_positive_no_context() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[step].model_asset = Some(1);
        add_detector(&mut graph, step, "d", None, None, HashMap::new());

        let mut rng = StdRng::seed_from_u64(2);
        let logs = collect_logs(5, &graph, [step], &HashSet::new(), &mut rng).unwrap();

        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].timestep, 5);
        assert_eq!(logs[0].trigger, step);
        assert_eq!(logs[0].detector_id, (step, "d".to_string()));
        assert!(logs[0].context.is_empty());
        assert!(!logs[0].false_positive);
    }

    #[test]
    fn collect_logs_tprate_one_always_true_positive() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[step].model_asset = Some(1);
        add_detector(&mut graph, step, "d", Some(1.0), None, HashMap::new());

        let mut rng = StdRng::seed_from_u64(3);
        let logs = collect_logs(0, &graph, [step], &HashSet::new(), &mut rng).unwrap();
        assert_eq!(logs.len(), 1);
    }

    #[test]
    fn collect_logs_tprate_negative_is_truthy_but_never_fires() {
        // Negative rates are still "truthy" in Python (only None/0.0 are
        // falsy) but can never satisfy `rate >= roll` since `roll` is
        // always >= 0.0 - mirrors `tests/test_event_logger.py::
        // test_logger_attacks_false_negative`'s `tprate=-1.0` case.
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[step].model_asset = Some(1);
        add_detector(&mut graph, step, "d", Some(-1.0), None, HashMap::new());

        let mut rng = StdRng::seed_from_u64(4);
        let logs = collect_logs(0, &graph, [step], &HashSet::new(), &mut rng).unwrap();
        assert!(logs.is_empty());
    }

    #[test]
    fn collect_logs_fills_context_from_previously_compromised_candidate() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let candidate = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let other_candidate = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[step].model_asset = Some(1);
        let potential: HashMap<_, _> = [(
            "computer".to_string(),
            [candidate, other_candidate].into_iter().collect(),
        )]
        .into_iter()
        .collect();
        add_detector(&mut graph, step, "d", None, None, potential);

        let previous: HashSet<_> = [candidate].into_iter().collect();
        let mut rng = StdRng::seed_from_u64(5);
        let logs = collect_logs(0, &graph, [step], &previous, &mut rng).unwrap();

        assert_eq!(logs.len(), 1);
        assert_eq!(
            logs[0].context,
            [("computer".to_string(), candidate)].into_iter().collect()
        );
    }

    #[test]
    fn collect_logs_context_errors_when_no_candidate_was_compromised() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let candidate = add_dummy_node(&mut graph, "DummyOrAttackStep");
        graph.nodes[step].model_asset = Some(1);
        let potential: HashMap<_, _> =
            [("computer".to_string(), [candidate].into_iter().collect())]
                .into_iter()
                .collect();
        add_detector(&mut graph, step, "d", None, None, potential);

        let mut rng = StdRng::seed_from_u64(6);
        let result = collect_logs(0, &graph, [step], &HashSet::new(), &mut rng);
        assert_eq!(
            result,
            Err(EventLoggerError::NoCompromisedContextNode(
                step,
                "computer".to_string()
            ))
        );
    }

    #[test]
    fn collect_false_positives_fprate_zero_never_fires() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        add_detector(&mut graph, step, "d", None, Some(0.0), HashMap::new());

        let mut rng = StdRng::seed_from_u64(7);
        let logs = collect_false_positives(0, &graph, &mut rng).unwrap();
        assert!(logs.is_empty());
    }

    #[test]
    fn collect_false_positives_fprate_one_always_fires() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        add_detector(&mut graph, step, "d", None, Some(1.0), HashMap::new());

        let mut rng = StdRng::seed_from_u64(8);
        let logs = collect_false_positives(3, &graph, &mut rng).unwrap();

        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].timestep, 3);
        assert_eq!(logs[0].trigger, step);
        assert_eq!(logs[0].detector_id, (step, "d".to_string()));
        assert!(logs[0].false_positive);
    }

    #[test]
    fn collect_false_positives_uses_random_context_candidate() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let candidate = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let potential: HashMap<_, _> =
            [("computer".to_string(), [candidate].into_iter().collect())]
                .into_iter()
                .collect();
        add_detector(&mut graph, step, "d", None, Some(1.0), potential);

        let mut rng = StdRng::seed_from_u64(9);
        let logs = collect_false_positives(0, &graph, &mut rng).unwrap();

        assert_eq!(logs.len(), 1);
        assert_eq!(
            logs[0].context,
            [("computer".to_string(), candidate)].into_iter().collect()
        );
    }

    #[test]
    fn collect_false_positives_empty_context_candidates_errors() {
        let mut graph = dummy_graph();
        let step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let potential: HashMap<String, HashSet<AttackGraphNodeId>> =
            [("computer".to_string(), HashSet::new())]
                .into_iter()
                .collect();
        add_detector(&mut graph, step, "d", None, Some(1.0), potential);

        let mut rng = StdRng::seed_from_u64(10);
        let result = collect_false_positives(0, &graph, &mut rng);
        assert_eq!(
            result,
            Err(EventLoggerError::EmptyContextCandidates(
                step,
                "computer".to_string()
            ))
        );
    }
}
