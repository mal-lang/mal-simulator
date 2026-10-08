//! Rust port of `python/malsim/dyna_mal_simulator/graph_state.py` - see
//! `PORTING_NOTES.md` §6 Phase B2.
//!
//! Folds newly-created nodes (from
//! [`crate::model_effects::execute_model_effects`]'s
//! `partially_regenerate_graph` call) into a [`GraphState`] - a thin
//! composition of Phase A3's per-node TTC/pre-enabled-defense/impossible-
//! step helpers plus Phase A3's `calculate_necessity`, restricted to the
//! new nodes rather than the whole graph.

use std::collections::{HashMap, HashSet};

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use rand::Rng;

use crate::graph_state::{
    attack_step_ttc_value, is_impossible_for_dist, is_pre_enabled_for_dist, resolve_ttc_dist,
    GraphState, GraphStateError, TtcMode,
};
use crate::necessity::calculate_necessity;

/// Port of `add_new_nodes_to_graph_state`. Takes the three relevant
/// `MalSimulatorSettings` fields directly rather than the whole settings
/// struct, same pattern A3's `compute_initial_graph_state` established.
pub fn add_new_nodes_to_graph_state(
    graph: &AttackGraph,
    graph_state: &GraphState,
    ttc_mode: TtcMode,
    run_defense_step_bernoullis: bool,
    run_attack_step_bernoullis: bool,
    new_nodes: &HashSet<AttackGraphNodeId>,
    rng: &mut impl Rng,
) -> Result<(GraphState, HashSet<AttackGraphNodeId>), GraphStateError> {
    let new_attack_steps: HashSet<AttackGraphNodeId> = new_nodes
        .iter()
        .copied()
        .filter(|&id| matches!(graph.nodes[id].step_type.as_str(), "and" | "or"))
        .collect();
    let new_defense_steps: Vec<AttackGraphNodeId> = new_nodes
        .iter()
        .copied()
        .filter(|&id| graph.nodes[id].step_type.as_str() == "defense")
        .collect();

    let mut new_ttc_values = HashMap::new();
    for &node_id in &new_attack_steps {
        let node = &graph.nodes[node_id];
        if let Some(value) = attack_step_ttc_value(node, None, ttc_mode, rng)? {
            new_ttc_values.insert(node_id, value);
        }
    }

    let mut new_steps_enabled_defenses = HashSet::new();
    for &node_id in &new_defense_steps {
        let node = &graph.nodes[node_id];
        let ttc_dist = resolve_ttc_dist(node, None)?;
        if is_pre_enabled_for_dist(&ttc_dist, run_defense_step_bernoullis, rng) {
            new_steps_enabled_defenses.insert(node_id);
        }
    }

    let enabled_defenses: HashSet<AttackGraphNodeId> = graph_state
        .pre_enabled_defenses
        .union(&new_steps_enabled_defenses)
        .copied()
        .collect();

    let new_impossible_attack_steps = if run_attack_step_bernoullis {
        let mut impossible = HashSet::new();
        for &node_id in &new_attack_steps {
            let node = &graph.nodes[node_id];
            if is_impossible_for_dist(&resolve_ttc_dist(node, None)?, rng) {
                impossible.insert(node_id);
            }
        }
        impossible
    } else {
        HashSet::new()
    };

    let necessity_per_node = calculate_necessity(graph, &enabled_defenses)?;

    let mut ttc_values = graph_state.ttc_values.clone();
    ttc_values.extend(new_ttc_values);

    let new_graph_state = GraphState {
        ttc_values,
        pre_enabled_defenses: enabled_defenses,
        impossible_attack_steps: graph_state
            .impossible_attack_steps
            .union(&new_impossible_attack_steps)
            .copied()
            .collect(),
        necessity_per_node,
    };
    Ok((new_graph_state, new_steps_enabled_defenses))
}

/// Shared by `dyna_attacker_step.rs`/`dyna_defender_step.rs`: folds the
/// node ids `execute_model_effects` just created into `graph_state`/
/// `enabled_defenses` in place - mirrors Python's `execute_model_effects`
/// unconditionally calling `add_new_nodes_to_graph_state` and reassigning
/// `sim_state`, even when `new_nodes` is empty (cheap, and keeps this
/// call unconditional rather than adding a special case that isn't in the
/// Python source).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_new_nodes_into_graph_state(
    graph: &AttackGraph,
    graph_state: &mut GraphState,
    enabled_defenses: &mut HashSet<AttackGraphNodeId>,
    new_nodes: &HashSet<AttackGraphNodeId>,
    ttc_mode: TtcMode,
    run_defense_step_bernoullis: bool,
    run_attack_step_bernoullis: bool,
    rng: &mut impl Rng,
) -> Result<(), GraphStateError> {
    let (new_graph_state, new_enabled) = add_new_nodes_to_graph_state(
        graph,
        graph_state,
        ttc_mode,
        run_defense_step_bernoullis,
        run_attack_step_bernoullis,
        new_nodes,
        rng,
    )?;
    *graph_state = new_graph_state;
    enabled_defenses.extend(new_enabled);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_state::compute_initial_graph_state;
    use crate::test_fixtures::wiper_attack_graph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(3)
    }

    #[test]
    fn add_new_nodes_to_graph_state_covers_newly_regenerated_nodes() {
        let (mut graph, mut model) = wiper_attack_graph();
        let mut r = rng();
        let initial_graph_state =
            compute_initial_graph_state(&graph, TtcMode::ExpectedValue, false, false, &mut r)
                .unwrap();

        let next_id = model.next_id;
        let wiper_id = model
            .add_asset(
                "Wiper",
                Some("Wiper".to_string()),
                Some(next_id),
                None,
                None,
                false,
            )
            .unwrap();
        let infected_device = model.get_asset_by_name("InfectedDevice").unwrap().id;
        model
            .add_associated_assets(wiper_id, "victim", HashSet::from([infected_device]))
            .unwrap();

        let new_assets = HashSet::from([wiper_id]);
        let new_associations = HashSet::from([(wiper_id, "victim".to_string(), infected_device)]);
        let new_nodes = graph
            .partially_regenerate_graph(
                &model,
                &new_assets,
                &new_associations,
                &HashMap::new(),
                &HashSet::new(),
            )
            .unwrap();
        assert!(
            !new_nodes.is_empty(),
            "adding the Wiper asset should create new attack steps"
        );

        let (new_state, new_enabled) = add_new_nodes_to_graph_state(
            &graph,
            &initial_graph_state,
            TtcMode::ExpectedValue,
            false,
            false,
            &new_nodes,
            &mut r,
        )
        .unwrap();

        assert!(new_enabled.is_empty(), "wiperLang has no defense steps");
        for &node_id in &new_nodes {
            assert!(
                new_state.necessity_per_node.contains_key(&node_id),
                "new node {node_id:?} missing from folded necessity map"
            );
        }
        // Pre-existing ttc_values/pre_enabled_defenses survive the fold.
        for (&node_id, &value) in &initial_graph_state.ttc_values {
            assert_eq!(new_state.ttc_values.get(&node_id), Some(&value));
        }
        assert_eq!(
            new_state.pre_enabled_defenses,
            initial_graph_state.pre_enabled_defenses
        );
    }
}
