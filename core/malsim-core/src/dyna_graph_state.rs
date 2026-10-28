//! Rust port of `python/malsim/dyna_mal_simulator/graph_state.py` - see
//! `PORTING_NOTES.md` §6 Phase B2.
//!
//! Folds newly-created nodes (from
//! [`crate::model_effects::execute_model_effects`]'s
//! `partially_regenerate_graph` call) into a [`GraphState`] - a thin
//! composition of Phase A3's per-node TTC/pre-enabled-defense/impossible-
//! step helpers plus Phase A3's `calculate_necessity`, restricted to the
//! new nodes rather than the whole graph.

use crate::types::NodeMap;
use crate::types::NodeSet;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_language::graph::attack_step::AttackStepType;
use rand::Rng;

use crate::graph_state::{
    attack_step_ttc_value, is_impossible_for_dist, is_pre_enabled_for_dist, resolve_ttc_dist,
    GraphState, GraphStateError, TtcMode,
};
use crate::graph_utils::sort_by_node_id;
use crate::model_effects::ModEffectOp;
use crate::necessity::calculate_necessity;

/// What one step's model-effect folds changed in `graph_state`, so the step can be undone.
#[derive(Debug, Clone, Default)]
pub struct GraphStateDelta {
    /// Every node the step's folds created.
    pub(crate) new_nodes: NodeSet,
    /// For each `necessity_per_node` key the folds changed, added or dropped: its value
    /// before the step (`None` = it was absent). The first value seen wins.
    pub(crate) necessity_prior: NodeMap<Option<bool>>,
}

impl GraphStateDelta {
    /// Appends a later delta from the same step; earlier prior values win.
    pub(crate) fn merge(&mut self, later: GraphStateDelta) {
        self.new_nodes.extend(later.new_nodes);
        for (id, prior) in later.necessity_prior {
            self.necessity_prior.entry(id).or_insert(prior);
        }
    }
}

/// Port of `add_new_nodes_to_graph_state`, in place: extends
/// `graph_state`'s `ttc_values`/`pre_enabled_defenses`/
/// `impossible_attack_steps` with the new nodes' and recomputes
/// `necessity_per_node` against `pre_enabled_defenses ∪ new enabled`.
/// Takes the three relevant `MalSimulatorSettings` fields directly rather
/// than the whole settings struct, same pattern A3's
/// `compute_initial_graph_state` established.
/// RNG draws happen in the same order as the Python original (TTC values,
/// then defense Bernoullis, then attack-step Bernoullis). `graph_state` is
/// only mutated once every fallible RNG/TTC step has succeeded.
/// Records what changed in `delta`. Returns the newly pre-enabled defenses.
#[allow(clippy::too_many_arguments)]
fn extend_graph_state_with_new_nodes(
    graph: &AttackGraph,
    graph_state: &mut GraphState,
    ttc_mode: TtcMode,
    run_defense_step_bernoullis: bool,
    run_attack_step_bernoullis: bool,
    new_nodes: &NodeSet,
    rng: &mut impl Rng,
    delta: &mut GraphStateDelta,
) -> Result<NodeSet, GraphStateError> {
    // RNG draws below follow these Vecs, so they are in node-id order
    // rather than the order of the `new_nodes` set.
    let mut sorted_new_nodes: Vec<AttackGraphNodeId> = new_nodes.iter().copied().collect();
    sort_by_node_id(graph, &mut sorted_new_nodes);
    let new_attack_steps: Vec<AttackGraphNodeId> = sorted_new_nodes
        .iter()
        .copied()
        .filter(|&id| {
            matches!(
                graph.nodes[id].step_type,
                AttackStepType::And | AttackStepType::Or
            )
        })
        .collect();
    let new_defense_steps: Vec<AttackGraphNodeId> = sorted_new_nodes
        .iter()
        .copied()
        .filter(|&id| graph.nodes[id].step_type == AttackStepType::Defense)
        .collect();

    let mut new_ttc_values = NodeMap::default();
    for &node_id in &new_attack_steps {
        let node = &graph.nodes[node_id];
        if let Some(value) = attack_step_ttc_value(node, None, ttc_mode, rng)? {
            new_ttc_values.insert(node_id, value);
        }
    }

    let mut new_steps_enabled_defenses = NodeSet::default();
    for &node_id in &new_defense_steps {
        let node = &graph.nodes[node_id];
        let ttc_dist = resolve_ttc_dist(node, None)?;
        if is_pre_enabled_for_dist(&ttc_dist, run_defense_step_bernoullis, rng) {
            new_steps_enabled_defenses.insert(node_id);
        }
    }

    let mut new_impossible_attack_steps = NodeSet::default();
    if run_attack_step_bernoullis {
        for &node_id in &new_attack_steps {
            let node = &graph.nodes[node_id];
            if is_impossible_for_dist(&resolve_ttc_dist(node, None)?, rng) {
                new_impossible_attack_steps.insert(node_id);
            }
        }
    }

    // Necessity is computed against pre_enabled ∪ new enabled (never the
    // live `enabled_defenses`); the union is only materialised when there
    // is something to add.
    let necessity_per_node = if new_steps_enabled_defenses.is_empty() {
        calculate_necessity(graph, &graph_state.pre_enabled_defenses)?
    } else {
        let union: NodeSet = graph_state
            .pre_enabled_defenses
            .union(&new_steps_enabled_defenses)
            .copied()
            .collect();
        calculate_necessity(graph, &union)?
    };

    // Node ids are versioned slotmap keys, so a regenerated node that reuses
    // a removed node's slot still gets a key no earlier node had. The fold
    // only ever adds keys; undoing it relies on that.
    debug_assert!(
        new_nodes.iter().all(|id| {
            !graph_state.ttc_values.contains_key(id)
                && !graph_state.pre_enabled_defenses.contains(id)
                && !graph_state.impossible_attack_steps.contains(id)
        }),
        "a newly created node's key is already present in graph_state"
    );
    graph_state.ttc_values.extend(new_ttc_values);
    graph_state
        .pre_enabled_defenses
        .extend(new_steps_enabled_defenses.iter().copied());
    graph_state
        .impossible_attack_steps
        .extend(new_impossible_attack_steps);
    let old_necessity = std::mem::replace(&mut graph_state.necessity_per_node, necessity_per_node);
    delta.new_nodes.extend(new_nodes.iter().copied());
    for (id, &v) in old_necessity.iter() {
        if graph_state.necessity_per_node.get(id) != Some(&v) {
            delta.necessity_prior.entry(id).or_insert(Some(v));
        }
    }
    for (id, _) in graph_state.necessity_per_node.iter() {
        if !old_necessity.contains_key(id) {
            delta.necessity_prior.entry(id).or_insert(None);
        }
    }
    Ok(new_steps_enabled_defenses)
}

/// Shared by `dyna_attacker_step.rs`/`dyna_defender_step.rs`: folds the
/// node ids `execute_model_effects` just created into `graph_state`/
/// `enabled_defenses` in place.
///
/// Python's `execute_model_effects` calls `add_new_nodes_to_graph_state`
/// unconditionally; here the fold is skipped when `ops` and `new_nodes`
/// are both empty, which is equivalent: with no recorded ops no
/// asset/association was added or removed, so
/// `partially_regenerate_graph` ran with all-empty inputs and left the
/// graph untouched; an empty `new_nodes` makes every RNG loop in the fold
/// a zero-iteration loop (no draws) and adds nothing to
/// `ttc_values`/`pre_enabled_defenses`/`impossible_attack_steps`/
/// `enabled_defenses`; and `necessity_per_node` would be recomputed over
/// the same graph with the same `pre_enabled_defenses` it was last
/// computed with (`compute_initial_graph_state` and every earlier fold
/// both use `pre_enabled_defenses`), so it comes out identical.
///
/// What the fold changed is recorded in `delta`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_new_nodes_into_graph_state(
    graph: &AttackGraph,
    graph_state: &mut GraphState,
    enabled_defenses: &mut NodeSet,
    ops: &[ModEffectOp],
    new_nodes: &NodeSet,
    ttc_mode: TtcMode,
    run_defense_step_bernoullis: bool,
    run_attack_step_bernoullis: bool,
    rng: &mut impl Rng,
    delta: &mut GraphStateDelta,
) -> Result<(), GraphStateError> {
    if ops.is_empty() && new_nodes.is_empty() {
        return Ok(());
    }
    let new_enabled = extend_graph_state_with_new_nodes(
        graph,
        graph_state,
        ttc_mode,
        run_defense_step_bernoullis,
        run_attack_step_bernoullis,
        new_nodes,
        rng,
        delta,
    )?;
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
    use std::collections::{HashMap, HashSet};

    fn rng() -> StdRng {
        StdRng::seed_from_u64(3)
    }

    #[test]
    fn extend_graph_state_with_new_nodes_covers_newly_regenerated_nodes() {
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
            .unwrap()
            .into_iter()
            .collect::<NodeSet>();
        assert!(
            !new_nodes.is_empty(),
            "adding the Wiper asset should create new attack steps"
        );

        let mut new_state = initial_graph_state.clone();
        let new_enabled = extend_graph_state_with_new_nodes(
            &graph,
            &mut new_state,
            TtcMode::ExpectedValue,
            false,
            false,
            &new_nodes,
            &mut r,
            &mut GraphStateDelta::default(),
        )
        .unwrap();

        assert!(new_enabled.is_empty(), "wiperLang has no defense steps");
        for &node_id in &new_nodes {
            assert!(
                new_state.necessity_per_node.contains_key(node_id),
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

    /// The slot index of a node key (the low 32 bits of its FFI form).
    fn slot_index(id: AttackGraphNodeId) -> u32 {
        use slotmap::Key;
        id.data().as_ffi() as u32
    }

    /// A node created after another node was removed can reuse its slot,
    /// but never its key: slotmap bumps the slot's version. So the keys a
    /// fold adds to `graph_state` are always new ones, which is what lets a
    /// step's fold be undone by removing exactly the step's new nodes.
    #[test]
    fn regenerated_nodes_reuse_slots_but_never_keys() {
        use crate::model_effects::regenerate_graph;

        let (mut graph, mut model) = wiper_attack_graph();
        let mut r = rng();
        let mut state =
            compute_initial_graph_state(&graph, TtcMode::ExpectedValue, false, false, &mut r)
                .unwrap();
        let keys_before: NodeSet = graph.nodes.keys().collect();

        // Remove InfectedData, freeing the slots of its nodes.
        let data_id = model.get_asset_by_name("InfectedData").unwrap().id;
        let device_id = model.get_asset_by_name("InfectedDevice").unwrap().id;
        let snapshot = model.remove_asset(data_id).unwrap();
        regenerate_graph(
            &mut graph,
            &model,
            &HashSet::new(),
            &HashSet::new(),
            &HashMap::from([(data_id, snapshot)]),
            &HashSet::from([(data_id, "node".to_string(), device_id)]),
        )
        .unwrap();
        let removed: NodeSet = keys_before
            .iter()
            .copied()
            .filter(|&id| !graph.nodes.contains_key(id))
            .collect();
        assert!(
            !removed.is_empty(),
            "removing InfectedData removed no nodes"
        );

        // Add a new Data asset; its nodes land in the freed slots.
        let next_id = model.next_id;
        let new_data = model
            .add_asset(
                "Data",
                Some("NewData".to_string()),
                Some(next_id),
                None,
                None,
                false,
            )
            .unwrap();
        model
            .add_associated_assets(new_data, "node", HashSet::from([device_id]))
            .unwrap();
        let new_nodes = regenerate_graph(
            &mut graph,
            &model,
            &HashSet::from([new_data]),
            &HashSet::from([(new_data, "node".to_string(), device_id)]),
            &HashMap::new(),
            &HashSet::new(),
        )
        .unwrap();

        // Positive control: without slot reuse this test proves nothing.
        let removed_slots: HashSet<u32> = removed.iter().map(|&id| slot_index(id)).collect();
        let reused: Vec<AttackGraphNodeId> = new_nodes
            .iter()
            .copied()
            .filter(|&id| removed_slots.contains(&slot_index(id)))
            .collect();
        assert!(!reused.is_empty(), "no new node reused a freed slot");

        // Every new key differs from every key the graph ever held, so none
        // is already in graph_state.
        assert!(new_nodes.is_disjoint(&keys_before));
        for id in &new_nodes {
            assert!(!state.ttc_values.contains_key(id));
            assert!(!state.pre_enabled_defenses.contains(id));
            assert!(!state.impossible_attack_steps.contains(id));
        }

        // Folding keeps the removed nodes' stale entries and adds the new
        // ones next to them (also exercises the fold's debug_assert).
        let ttc_before = state.ttc_values.clone();
        extend_graph_state_with_new_nodes(
            &graph,
            &mut state,
            TtcMode::ExpectedValue,
            false,
            false,
            &new_nodes,
            &mut r,
            &mut GraphStateDelta::default(),
        )
        .unwrap();
        for (id, value) in &ttc_before {
            assert_eq!(state.ttc_values.get(id), Some(value));
        }

        // The necessity map is slot-indexed: a reused slot holds the new
        // node's entry, and the removed node's older key reads as absent.
        // Restoring an old entry there means removing the new key first,
        // since `SecondaryMap::insert` ignores keys older than the slot's.
        for &new_id in &reused {
            assert!(state.necessity_per_node.contains_key(new_id));
            let old_id = *removed
                .iter()
                .find(|&&id| slot_index(id) == slot_index(new_id))
                .unwrap();
            assert!(!state.necessity_per_node.contains_key(old_id));
            let mut necessity = state.necessity_per_node.clone();
            assert_eq!(necessity.insert(old_id, true), None);
            assert!(!necessity.contains_key(old_id), "older key was inserted");
            necessity.remove(new_id);
            necessity.insert(old_id, true);
            assert!(necessity.contains_key(old_id));
        }
    }
}
