//! Rust port of `python/malsim/dyna_mal_simulator/defender_step.py` - see
//! `PORTING_NOTES.md` §6 Phase B2.
//!
//! Same wrapping shape as `dyna_attacker_step.rs`, simpler: no TTC
//! attempt/effect-chain logic, just "enable the defense, then run its
//! model effects."

use std::collections::HashSet;
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_model::Model;
use rand::Rng;

use crate::dyna_graph_state::fold_new_nodes_into_graph_state;
use crate::graph_state::{GraphState, GraphStateError, TtcMode};
use crate::graph_utils::{node_is_live, GraphUtilsError};
use crate::model_effects::{execute_model_effects, ModEffectOp, ModelEffectsError};

#[derive(Debug)]
pub enum DynaDefenderStepError {
    GraphUtils(GraphUtilsError),
    GraphState(GraphStateError),
    ModelEffects(Box<ModelEffectsError>),
    NodeNotInGraph(AttackGraphNodeId),
}

impl From<GraphUtilsError> for DynaDefenderStepError {
    fn from(e: GraphUtilsError) -> Self {
        DynaDefenderStepError::GraphUtils(e)
    }
}

impl From<GraphStateError> for DynaDefenderStepError {
    fn from(e: GraphStateError) -> Self {
        DynaDefenderStepError::GraphState(e)
    }
}

impl From<ModelEffectsError> for DynaDefenderStepError {
    fn from(e: ModelEffectsError) -> Self {
        DynaDefenderStepError::ModelEffects(Box::new(e))
    }
}

impl fmt::Display for DynaDefenderStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynaDefenderStepError::GraphUtils(e) => write!(f, "{e}"),
            DynaDefenderStepError::GraphState(e) => write!(f, "{e}"),
            DynaDefenderStepError::ModelEffects(e) => write!(f, "{e}"),
            DynaDefenderStepError::NodeNotInGraph(id) => write!(
                f,
                "tried to step a node ({id:?}) that is not part of this simulator's attack graph"
            ),
        }
    }
}

impl std::error::Error for DynaDefenderStepError {}

/// Port of `dyna_defender_step`. A node outside `action_surface` is
/// silently skipped (Python logs a warning), not an error - only the
/// graph-membership check is a hard failure, same as A7's `defender_step`.
#[allow(clippy::too_many_arguments)]
pub fn dyna_defender_step(
    graph: &mut AttackGraph,
    model: &mut Model,
    rng: &mut impl Rng,
    ttc_mode: TtcMode,
    run_defense_step_bernoullis: bool,
    run_attack_step_bernoullis: bool,
    nodes: &[AttackGraphNodeId],
    action_surface: &HashSet<AttackGraphNodeId>,
    graph_state: &mut GraphState,
    enabled_defenses: &mut HashSet<AttackGraphNodeId>,
) -> Result<(Vec<AttackGraphNodeId>, Vec<ModEffectOp>), DynaDefenderStepError> {
    let mut enabled = Vec::new();
    let mut modification_record = Vec::new();

    for &node_id in nodes {
        if !node_is_live(graph, node_id) {
            return Err(DynaDefenderStepError::NodeNotInGraph(node_id));
        }

        if !action_surface.contains(&node_id) {
            continue;
        }

        enabled.push(node_id);
        let (ops, new_nodes) = execute_model_effects(graph, model, node_id, rng)?;
        modification_record.extend(ops);
        fold_new_nodes_into_graph_state(
            graph,
            graph_state,
            enabled_defenses,
            &new_nodes,
            ttc_mode,
            run_defense_step_bernoullis,
            run_attack_step_bernoullis,
            rng,
        )?;
    }

    Ok((enabled, modification_record))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::graph_state::compute_initial_graph_state;
    use crate::test_fixtures::{add_dummy_node, dummy_graph_and_model};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(1)
    }

    fn empty_graph_state() -> GraphState {
        GraphState {
            ttc_values: HashMap::new(),
            pre_enabled_defenses: HashSet::new(),
            impossible_attack_steps: HashSet::new(),
            necessity_per_node: HashMap::new(),
        }
    }

    #[test]
    fn dyna_defender_step_fails_node_not_in_graph() {
        let (mut graph, mut model) = dummy_graph_and_model();
        let node = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        graph.remove_node(node).unwrap();
        let action_surface: HashSet<_> = [node].into_iter().collect();
        let mut r = rng();
        let mut graph_state = empty_graph_state();
        let mut enabled_defenses = HashSet::new();

        let result = dyna_defender_step(
            &mut graph,
            &mut model,
            &mut r,
            TtcMode::Disabled,
            false,
            false,
            &[node],
            &action_surface,
            &mut graph_state,
            &mut enabled_defenses,
        );
        assert!(matches!(
            result,
            Err(DynaDefenderStepError::NodeNotInGraph(_))
        ));
    }

    #[test]
    fn dyna_defender_step_skips_node_outside_action_surface() {
        let (mut graph, mut model) = dummy_graph_and_model();
        let node = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let empty_set: HashSet<AttackGraphNodeId> = HashSet::new();
        let mut r = rng();
        let mut graph_state =
            compute_initial_graph_state(&graph, TtcMode::Disabled, false, false, &mut r).unwrap();
        let mut enabled_defenses = HashSet::new();

        let (enabled, record) = dyna_defender_step(
            &mut graph,
            &mut model,
            &mut r,
            TtcMode::Disabled,
            false,
            false,
            &[node],
            &empty_set,
            &mut graph_state,
            &mut enabled_defenses,
        )
        .unwrap();
        assert!(enabled.is_empty());
        assert!(record.is_empty());
    }

    #[test]
    fn dyna_defender_step_enables_defense_with_no_model_effects() {
        let (mut graph, mut model) = dummy_graph_and_model();
        let node = add_dummy_node(&mut graph, "DummyDefenseAttackStep");
        let action_surface: HashSet<_> = [node].into_iter().collect();
        let mut r = rng();
        let mut graph_state =
            compute_initial_graph_state(&graph, TtcMode::Disabled, false, false, &mut r).unwrap();
        let mut enabled_defenses = HashSet::new();

        let (enabled, record) = dyna_defender_step(
            &mut graph,
            &mut model,
            &mut r,
            TtcMode::Disabled,
            false,
            false,
            &[node],
            &action_surface,
            &mut graph_state,
            &mut enabled_defenses,
        )
        .unwrap();
        assert_eq!(enabled, vec![node]);
        assert!(
            record.is_empty(),
            "dummy_lang declares no model effects on this step"
        );
    }
}
