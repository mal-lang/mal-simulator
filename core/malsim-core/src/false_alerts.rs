//! Rust port of `python/malsim/mal_simulator/false_alerts.py` - see
//! `PORTING_NOTES.md` §5 Phase A6.
//!
//! Per §2.4, `NodePropertyRule[float]` (`false_positive_rates`/
//! `false_negative_rates`) stays Python-only; Python flattens it once per
//! `reset()` into a plain `HashMap<AttackGraphNodeId, f64>` before crossing
//! into this hot path - the same pattern A5 established for
//! `NodePropertyRule[bool]` (`attack_surface.rs`'s `actionable_steps`).
//! `None` means "no rule configured at all" (mirroring
//! `if false_positive_rates_rule:`/`if false_negative_rate_rule:` being
//! false - zero draws, empty result, not even a 0.0-rate check per node);
//! `Some(map)` holds exactly the already-flattened per-node rates, and a
//! node missing from the map still defaults to `0.0`, mirroring
//! `NodePropertyRule.value(node, 0.0)`'s own default argument.

use std::collections::{HashMap, HashSet};

use maltoolbox_attackgraph::AttackGraphNodeId;
use rand::{Rng, RngExt};

/// Port of `node_false_negative_rate`.
pub fn node_false_negative_rate(
    false_negative_rates: Option<&HashMap<AttackGraphNodeId, f64>>,
    node_id: AttackGraphNodeId,
) -> f64 {
    false_negative_rates
        .and_then(|rates| rates.get(&node_id))
        .copied()
        .unwrap_or(0.0)
}

/// Port of `generate_false_negatives`.
pub fn generate_false_negatives(
    false_negative_rates: Option<&HashMap<AttackGraphNodeId, f64>>,
    observed_nodes: &HashSet<AttackGraphNodeId>,
    rng: &mut impl Rng,
) -> HashSet<AttackGraphNodeId> {
    let Some(rates) = false_negative_rates else {
        return HashSet::new();
    };
    observed_nodes
        .iter()
        .copied()
        .filter(|&id| rng.random::<f64>() < node_false_negative_rate(Some(rates), id))
        .collect()
}

/// Port of `node_false_positive_rate`.
pub fn node_false_positive_rate(
    false_positive_rates: Option<&HashMap<AttackGraphNodeId, f64>>,
    node_id: AttackGraphNodeId,
) -> f64 {
    false_positive_rates
        .and_then(|rates| rates.get(&node_id))
        .copied()
        .unwrap_or(0.0)
}

/// Port of `generate_false_positives`. Takes the attack-step ids directly
/// (`attack_graph.attack_steps` on the Python side, e.g.
/// `AttackGraph::attack_steps` in Rust) rather than the whole graph, since
/// that's the only thing this function reads off it.
pub fn generate_false_positives(
    false_positive_rates: Option<&HashMap<AttackGraphNodeId, f64>>,
    attack_step_ids: impl Iterator<Item = AttackGraphNodeId>,
    rng: &mut impl Rng,
) -> HashSet<AttackGraphNodeId> {
    let Some(rates) = false_positive_rates else {
        return HashSet::new();
    };
    attack_step_ids
        .filter(|&id| rng.random::<f64>() < node_false_positive_rate(Some(rates), id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashMap as Map;

    /// Mints `n` distinct, real `AttackGraphNodeId`s. These functions are
    /// graph-independent (plain id-keyed maps/sets), but `AttackGraphNodeId`
    /// is a slotmap generational key - unlike `AttackStepId` it needs no
    /// specific `step_type`, but it still can't be fabricated from a raw
    /// integer, so a throwaway graph mints real ones instead (mirroring
    /// `graph_utils.rs`/`necessity.rs`'s own test fixtures).
    fn ids(n: usize) -> Vec<AttackGraphNodeId> {
        let mut graph = dummy_graph();
        (0..n)
            .map(|_| add_dummy_node(&mut graph, "DummyOrAttackStep"))
            .collect()
    }

    #[test]
    fn node_false_negative_rate_defaults_to_zero_without_rule() {
        let [n1] = ids(1)[..] else { unreachable!() };
        assert_eq!(node_false_negative_rate(None, n1), 0.0);
    }

    #[test]
    fn node_false_negative_rate_defaults_to_zero_for_unmapped_node() {
        let [n1, n2] = ids(2)[..] else { unreachable!() };
        let rates: Map<_, _> = [(n1, 0.5)].into_iter().collect();
        assert_eq!(node_false_negative_rate(Some(&rates), n2), 0.0);
    }

    #[test]
    fn node_false_negative_rate_returns_mapped_value() {
        let [n1] = ids(1)[..] else { unreachable!() };
        let rates: Map<_, _> = [(n1, 0.5)].into_iter().collect();
        assert_eq!(node_false_negative_rate(Some(&rates), n1), 0.5);
    }

    #[test]
    fn node_false_positive_rate_defaults_to_zero_without_rule() {
        let [n1] = ids(1)[..] else { unreachable!() };
        assert_eq!(node_false_positive_rate(None, n1), 0.0);
    }

    #[test]
    fn node_false_positive_rate_returns_mapped_value() {
        let [n1] = ids(1)[..] else { unreachable!() };
        let rates: Map<_, _> = [(n1, 0.5)].into_iter().collect();
        assert_eq!(node_false_positive_rate(Some(&rates), n1), 0.5);
    }

    #[test]
    fn generate_false_negatives_empty_without_rule() {
        let observed: HashSet<_> = ids(2).into_iter().collect();
        let mut rng = StdRng::seed_from_u64(1);
        let result = generate_false_negatives(None, &observed, &mut rng);
        assert!(result.is_empty());
    }

    #[test]
    fn generate_false_negatives_rate_one_always_fires() {
        let observed: HashSet<_> = ids(3).into_iter().collect();
        let rates: Map<_, _> = observed.iter().map(|&n| (n, 1.0)).collect();
        let mut rng = StdRng::seed_from_u64(2);
        let result = generate_false_negatives(Some(&rates), &observed, &mut rng);
        assert_eq!(result, observed);
    }

    #[test]
    fn generate_false_negatives_rate_zero_never_fires() {
        let observed: HashSet<_> = ids(3).into_iter().collect();
        let rates: Map<_, _> = observed.iter().map(|&n| (n, 0.0)).collect();
        let mut rng = StdRng::seed_from_u64(3);
        let result = generate_false_negatives(Some(&rates), &observed, &mut rng);
        assert!(result.is_empty());
    }

    #[test]
    fn generate_false_negatives_only_considers_observed_nodes() {
        let [n1, n2] = ids(2)[..] else { unreachable!() };
        let observed: HashSet<_> = [n1].into_iter().collect();
        // n2 has rate 1.0 but isn't in `observed_nodes` - must never appear
        // in the result, mirroring Python's comprehension iterating
        // `observed_nodes`, never the rate map's own keys.
        let rates: Map<_, _> = [(n1, 1.0), (n2, 1.0)].into_iter().collect();
        let mut rng = StdRng::seed_from_u64(4);
        let result = generate_false_negatives(Some(&rates), &observed, &mut rng);
        assert_eq!(result, observed);
    }

    #[test]
    fn generate_false_positives_empty_without_rule() {
        let steps = ids(2).into_iter();
        let mut rng = StdRng::seed_from_u64(5);
        let result = generate_false_positives(None, steps, &mut rng);
        assert!(result.is_empty());
    }

    #[test]
    fn generate_false_positives_rate_one_always_fires() {
        let steps = ids(3);
        let rates: Map<_, _> = steps.iter().map(|&n| (n, 1.0)).collect();
        let mut rng = StdRng::seed_from_u64(6);
        let result = generate_false_positives(Some(&rates), steps.iter().copied(), &mut rng);
        assert_eq!(result, steps.into_iter().collect::<HashSet<_>>());
    }

    #[test]
    fn generate_false_positives_rate_zero_never_fires() {
        let steps = ids(3);
        let rates: Map<_, _> = steps.iter().map(|&n| (n, 0.0)).collect();
        let mut rng = StdRng::seed_from_u64(7);
        let result = generate_false_positives(Some(&rates), steps.into_iter(), &mut rng);
        assert!(result.is_empty());
    }

    #[test]
    fn generate_false_positives_unmapped_node_defaults_to_zero_rate() {
        let [n1] = ids(1)[..] else { unreachable!() };
        // No entry for n1 in the rate map at all -> defaults to 0.0, never
        // fires, mirroring `node_false_positive_rate`'s default.
        let rates: Map<AttackGraphNodeId, f64> = Map::new();
        let mut rng = StdRng::seed_from_u64(8);
        let result = generate_false_positives(Some(&rates), std::iter::once(n1), &mut rng);
        assert!(result.is_empty());
    }
}
