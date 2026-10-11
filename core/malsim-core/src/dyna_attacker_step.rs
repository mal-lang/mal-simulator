//! Rust port of `python/malsim/dyna_mal_simulator/attacker_step.py` - see
//! `PORTING_NOTES.md` §6 Phase B2.
//!
//! Wraps Phase A7's `attacker_step::attempt_attacker_step`/
//! `attacker_step_effects` with Phase B1's `model_effects::
//! execute_model_effects`, called on every successful compromise and every
//! resulting effect node - mirrors Python's `dyna_attacker_step`
//! reassigning `sim_state` after each model effect, which the *next* node
//! in the same batch (and the effect-chain computation for the *current*
//! node) reads back. `graph_state`/`enabled_defenses` are threaded through
//! as `&mut` rather than returned copies for exactly that reason.
//!
//! **No separate `dyna_attempt_attacker_step` here.** Python's version is
//! identical to A7's `attempt_attacker_step` except for resolving
//! `agent.num_attempts.get(node, 0)` (a node created mid-simulation may
//! not be in the dict yet) instead of `agent.num_attempts[node]` - but
//! A7's Rust `attacker_step` already resolves its `num_attempts_before`
//! argument the same defensive way (`num_attempts.get(&node_id).copied()
//! .unwrap_or(0)`, for the same "new nodes aren't seeded yet" reason), so
//! `attacker_step::attempt_attacker_step` is reused directly below with
//! no behavioral gap.

use std::collections::{HashMap, HashSet};
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_model::Model;
use rand::Rng;

use crate::attacker_step::{attacker_step_effects, attempt_attacker_step, AttackerStepError};
use crate::dyna_graph_state::fold_new_nodes_into_graph_state;
use crate::graph_state::{GraphState, GraphStateError, TtcMode};
use crate::graph_utils::{node_is_live, node_is_traversable, GraphUtilsError};
use crate::model_effects::{execute_model_effects, ModEffectOp, ModelEffectsError};
use crate::ttc::TtcDist;

#[derive(Debug)]
pub enum DynaAttackerStepError {
    GraphUtils(GraphUtilsError),
    Attacker(AttackerStepError),
    GraphState(GraphStateError),
    ModelEffects(Box<ModelEffectsError>),
    /// Mirrors Python's `assert node == sim_state.attack_graph.nodes[node.id]`,
    /// translated to "this node id no longer exists in the graph", same
    /// as A7's `AttackerStepError::NodeNotInGraph`.
    NodeNotInGraph(AttackGraphNodeId),
}

impl From<GraphUtilsError> for DynaAttackerStepError {
    fn from(e: GraphUtilsError) -> Self {
        DynaAttackerStepError::GraphUtils(e)
    }
}

impl From<AttackerStepError> for DynaAttackerStepError {
    fn from(e: AttackerStepError) -> Self {
        DynaAttackerStepError::Attacker(e)
    }
}

impl From<GraphStateError> for DynaAttackerStepError {
    fn from(e: GraphStateError) -> Self {
        DynaAttackerStepError::GraphState(e)
    }
}

impl From<ModelEffectsError> for DynaAttackerStepError {
    fn from(e: ModelEffectsError) -> Self {
        DynaAttackerStepError::ModelEffects(Box::new(e))
    }
}

impl fmt::Display for DynaAttackerStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynaAttackerStepError::GraphUtils(e) => write!(f, "{e}"),
            DynaAttackerStepError::Attacker(e) => write!(f, "{e}"),
            DynaAttackerStepError::GraphState(e) => write!(f, "{e}"),
            DynaAttackerStepError::ModelEffects(e) => write!(f, "{e}"),
            DynaAttackerStepError::NodeNotInGraph(id) => write!(
                f,
                "tried to step a node ({id:?}) that is not part of this simulator's attack graph"
            ),
        }
    }
}

impl std::error::Error for DynaAttackerStepError {}

/// (successful compromises, attempted-but-failed compromises, modification
/// record from every model effect executed along the way).
pub type DynaAttackerStepOutcome = (
    Vec<AttackGraphNodeId>,
    Vec<AttackGraphNodeId>,
    Vec<ModEffectOp>,
);

/// Port of `dyna_attacker_step`. `performed_nodes` is read-only here, same
/// contract as A7's `attacker_step` - it does not grow mid-batch even as
/// nodes in `nodes` succeed, matching Python's `agent.performed_nodes`
/// (an immutable per-step snapshot on the frozen `AttackerState`).
#[allow(clippy::too_many_arguments)]
pub fn dyna_attacker_step(
    graph: &mut AttackGraph,
    model: &mut Model,
    rng: &mut impl Rng,
    ttc_mode: TtcMode,
    run_defense_step_bernoullis: bool,
    run_attack_step_bernoullis: bool,
    nodes: &[AttackGraphNodeId],
    entry_points: &HashSet<AttackGraphNodeId>,
    action_surface: &HashSet<AttackGraphNodeId>,
    performed_nodes: &HashSet<AttackGraphNodeId>,
    num_attempts: &HashMap<AttackGraphNodeId, u64>,
    ttc_dist_overrides: Option<&HashMap<AttackGraphNodeId, TtcDist>>,
    ttc_value_overrides: Option<&HashMap<AttackGraphNodeId, f64>>,
    graph_state: &mut GraphState,
    enabled_defenses: &mut HashSet<AttackGraphNodeId>,
) -> Result<DynaAttackerStepOutcome, DynaAttackerStepError> {
    let mut successful_compromises = Vec::new();
    let mut attempted_compromises = Vec::new();
    let mut modification_record = Vec::new();

    for &node_id in nodes {
        if !node_is_live(graph, node_id) {
            return Err(DynaAttackerStepError::NodeNotInGraph(node_id));
        }

        let can_compromise = if entry_points.contains(&node_id) {
            true
        } else {
            action_surface.contains(&node_id)
                && node_is_traversable(
                    graph,
                    node_id,
                    performed_nodes,
                    &graph_state.impossible_attack_steps,
                    enabled_defenses,
                    &graph_state.necessity_per_node,
                )?
        };

        if !can_compromise {
            continue;
        }

        let num_attempts_before = num_attempts.get(&node_id).copied().unwrap_or(0);
        let ttc_override = ttc_dist_overrides.and_then(|m| m.get(&node_id));

        let succeeded = attempt_attacker_step(
            graph,
            rng,
            ttc_mode,
            node_id,
            num_attempts_before,
            ttc_override,
            ttc_value_overrides,
            &graph_state.ttc_values,
        )?;

        if !succeeded {
            attempted_compromises.push(node_id);
            continue;
        }

        successful_compromises.push(node_id);
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

        let effects = attacker_step_effects(
            graph,
            node_id,
            performed_nodes,
            &graph_state.impossible_attack_steps,
            enabled_defenses,
            &graph_state.necessity_per_node,
        )?;
        for effect_node in effects {
            successful_compromises.push(effect_node);
            // `effects` was computed once, above, against the graph as it
            // stood right after `node_id`'s own model effects ran - but an
            // *earlier* `effect_node` in this same list can itself remove
            // the asset a *later* one belongs to (the same self-removal
            // shape `PORTING_NOTES.md` §0 B5 traces for `node_id` itself,
            // one level deeper). `effect_node` was genuinely satisfied at
            // the moment `attacker_step_effects` found it - that's already
            // recorded above - there's just no live node left to run a
            // model effect for.
            if !node_is_live(graph, effect_node) {
                continue;
            }
            let (ops, new_nodes) = execute_model_effects(graph, model, effect_node, rng)?;
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
    }

    Ok((
        successful_compromises,
        attempted_compromises,
        modification_record,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_state::compute_initial_graph_state;
    use crate::model_effects::AssetOp;
    use crate::test_fixtures::wiper_attack_graph;
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
    fn dyna_attacker_step_fails_node_not_in_graph() {
        let (mut graph, mut model) = wiper_attack_graph();
        let infect_id = graph.full_name_to_node["InfectedDevice:infect"];
        graph.remove_node(infect_id).unwrap();
        let mut r = rng();
        let mut graph_state = empty_graph_state();
        let mut enabled_defenses = HashSet::new();
        let empty_set: HashSet<AttackGraphNodeId> = HashSet::new();
        let empty_map_u64: HashMap<AttackGraphNodeId, u64> = HashMap::new();

        let result = dyna_attacker_step(
            &mut graph,
            &mut model,
            &mut r,
            TtcMode::Disabled,
            false,
            false,
            &[infect_id],
            &empty_set,
            &empty_set,
            &empty_set,
            &empty_map_u64,
            None,
            None,
            &mut graph_state,
            &mut enabled_defenses,
        );
        assert!(matches!(
            result,
            Err(DynaAttackerStepError::NodeNotInGraph(_))
        ));
    }

    #[test]
    fn dyna_attacker_step_skips_node_outside_action_surface_and_not_entry_point() {
        let (mut graph, mut model) = wiper_attack_graph();
        let infect_id = graph.full_name_to_node["InfectedDevice:infect"];
        let mut r = rng();
        let mut graph_state =
            compute_initial_graph_state(&graph, TtcMode::Disabled, false, false, &mut r).unwrap();
        let mut enabled_defenses = HashSet::new();
        let empty_set: HashSet<AttackGraphNodeId> = HashSet::new();
        let empty_map_u64: HashMap<AttackGraphNodeId, u64> = HashMap::new();

        let (successful, attempted, record) = dyna_attacker_step(
            &mut graph,
            &mut model,
            &mut r,
            TtcMode::Disabled,
            false,
            false,
            &[infect_id],
            &empty_set,
            &empty_set,
            &empty_set,
            &empty_map_u64,
            None,
            None,
            &mut graph_state,
            &mut enabled_defenses,
        )
        .unwrap();
        assert!(successful.is_empty());
        assert!(attempted.is_empty());
        assert!(record.is_empty());
    }

    #[test]
    fn dyna_attacker_step_entry_point_compromise_runs_model_effects_and_folds_graph_state() {
        let (mut graph, mut model) = wiper_attack_graph();
        let mut r = rng();
        let mut graph_state =
            compute_initial_graph_state(&graph, TtcMode::Disabled, false, false, &mut r).unwrap();
        let mut enabled_defenses: HashSet<AttackGraphNodeId> = HashSet::new();

        let infect_id = graph.full_name_to_node["InfectedDevice:infect"];
        let entry_points: HashSet<_> = [infect_id].into_iter().collect();
        let empty_set: HashSet<AttackGraphNodeId> = HashSet::new();
        let empty_map_u64: HashMap<AttackGraphNodeId, u64> = HashMap::new();

        let (successful, attempted, record) = dyna_attacker_step(
            &mut graph,
            &mut model,
            &mut r,
            TtcMode::Disabled,
            false,
            false,
            &[infect_id],
            &entry_points,
            &empty_set,
            &empty_set,
            &empty_map_u64,
            None,
            None,
            &mut graph_state,
            &mut enabled_defenses,
        )
        .unwrap();

        assert_eq!(successful, vec![infect_id]);
        assert!(attempted.is_empty());
        assert!(
            record
                .iter()
                .any(|op| matches!(op, ModEffectOp::Asset(AssetOp::Added { .. }))),
            "infect's model effect should have created the Wiper asset"
        );
        assert!(model.get_asset_by_name("Wiper-7").is_some());

        // The newly-regenerated Wiper-7:* nodes are folded into the
        // returned graph_state (every node in the graph has a necessity
        // entry, including ones that didn't exist before this step).
        let wiper_activate_id = graph.full_name_to_node["Wiper-7:activate"];
        assert!(graph_state
            .necessity_per_node
            .contains_key(&wiper_activate_id));
    }
}
