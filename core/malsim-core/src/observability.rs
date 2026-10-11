//! Rust port of `python/malsim/mal_simulator/observability.py` - see
//! `PORTING_NOTES.md` §5 Phase A6.
//!
//! `node_is_observable` takes the already-flattened `observable_steps` id
//! set instead of a `NodePropertyRule[bool]`, same shape as A5's
//! `node_is_actionable_flat` (`attack_surface.rs`/`defense_surface.rs`):
//! `None` means "no rule configured" (every node observable, mirroring
//! `observed_nodes`'s own `if observable_steps_rule: ... else:
//! compromised_nodes` branch - no filtering at all), `Some(set)` means
//! exactly the already-true-valued ids are observable (mirroring a
//! flattened `NodePropertyRule.per_node()` result). A third private copy
//! of the same four-line idiom, for the same reason A5 kept two
//! independent copies rather than sharing one in `graph_utils.rs`: the
//! real `node_is_observable` stays Python per §2.4, operating on the real
//! `NodePropertyRule` directly, so a same-shaped-but-different Rust
//! function of the same name here would be confusing to a reader
//! grepping for its ported counterpart and not finding one.

use std::collections::{HashMap, HashSet};

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use rand::Rng;

use crate::false_alerts::{generate_false_negatives, generate_false_positives};

/// Port of `node_is_observable`, operating on the already-flattened id-set
/// instead of a `NodePropertyRule` - see module docs.
fn node_is_observable_flat(
    observable_steps: Option<&HashSet<AttackGraphNodeId>>,
    node_id: AttackGraphNodeId,
) -> bool {
    match observable_steps {
        Some(steps) => steps.contains(&node_id),
        None => true,
    }
}

/// Port of `observed_nodes`. Takes `graph` (rather than the whole
/// `MalSimulatorState` the Python function reads `sim_state.attack_graph`
/// off) since that's the only field this function actually needs -
/// `generate_false_positives` only reads `attack_graph.attack_steps`.
pub fn observed_nodes(
    observable_steps: Option<&HashSet<AttackGraphNodeId>>,
    false_positive_rates: Option<&HashMap<AttackGraphNodeId, f64>>,
    false_negative_rates: Option<&HashMap<AttackGraphNodeId, f64>>,
    graph: &AttackGraph,
    compromised_nodes: &HashSet<AttackGraphNodeId>,
    rng: &mut impl Rng,
) -> HashSet<AttackGraphNodeId> {
    let observable_steps_set: HashSet<AttackGraphNodeId> = if observable_steps.is_some() {
        compromised_nodes
            .iter()
            .copied()
            .filter(|&id| node_is_observable_flat(observable_steps, id))
            .collect()
    } else {
        compromised_nodes.clone()
    };

    let false_negatives = generate_false_negatives(false_negative_rates, compromised_nodes, rng);
    let false_positives = generate_false_positives(
        false_positive_rates,
        graph.attack_steps.iter().copied(),
        rng,
    );

    observable_steps_set
        .difference(&false_negatives)
        .copied()
        .collect::<HashSet<_>>()
        .union(&false_positives)
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{add_dummy_node, dummy_graph};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn node_is_observable_flat_true_for_everyone_without_rule() {
        let mut graph = dummy_graph();
        let n = add_dummy_node(&mut graph, "DummyOrAttackStep");
        assert!(node_is_observable_flat(None, n));
    }

    #[test]
    fn node_is_observable_flat_respects_flattened_set() {
        let mut graph = dummy_graph();
        let observable = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let not_observable = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let set: HashSet<_> = [observable].into_iter().collect();
        assert!(node_is_observable_flat(Some(&set), observable));
        assert!(!node_is_observable_flat(Some(&set), not_observable));
    }

    #[test]
    fn observed_nodes_without_any_rule_sees_every_compromised_node() {
        let mut graph = dummy_graph();
        let n1 = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let n2 = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let compromised: HashSet<_> = [n1, n2].into_iter().collect();
        let mut rng = StdRng::seed_from_u64(1);

        let result = observed_nodes(None, None, None, &graph, &compromised, &mut rng);
        assert_eq!(result, compromised);
    }

    #[test]
    fn observed_nodes_filters_by_observable_steps() {
        let mut graph = dummy_graph();
        let observable = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let not_observable = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let compromised: HashSet<_> = [observable, not_observable].into_iter().collect();
        let observable_set: HashSet<_> = [observable].into_iter().collect();
        let mut rng = StdRng::seed_from_u64(2);

        let result = observed_nodes(
            Some(&observable_set),
            None,
            None,
            &graph,
            &compromised,
            &mut rng,
        );
        assert_eq!(result, [observable].into_iter().collect());
    }

    #[test]
    fn observed_nodes_false_negative_rate_one_hides_compromised_node() {
        let mut graph = dummy_graph();
        let n = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let compromised: HashSet<_> = [n].into_iter().collect();
        let fn_rates: HashMap<_, _> = [(n, 1.0)].into_iter().collect();
        let mut rng = StdRng::seed_from_u64(3);

        let result = observed_nodes(None, None, Some(&fn_rates), &graph, &compromised, &mut rng);
        assert!(result.is_empty());
    }

    #[test]
    fn observed_nodes_false_positive_rate_one_adds_uncompromised_attack_step() {
        let mut graph = dummy_graph();
        let compromised_node = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let other_attack_step = add_dummy_node(&mut graph, "DummyOrAttackStep");
        let compromised: HashSet<_> = [compromised_node].into_iter().collect();
        let fp_rates: HashMap<_, _> = graph.attack_steps.iter().map(|&id| (id, 1.0)).collect();
        let mut rng = StdRng::seed_from_u64(4);

        let result = observed_nodes(None, Some(&fp_rates), None, &graph, &compromised, &mut rng);
        // Every attack step (compromised or not) becomes a false positive
        // at rate 1.0, unioned with the (unfiltered) observable set.
        assert!(result.contains(&compromised_node));
        assert!(result.contains(&other_attack_step));
    }
}
