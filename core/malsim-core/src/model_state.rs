//! Rust port of `python/malsim/dyna_mal_simulator/model_state.py` - see
//! `PORTING_NOTES.md` §6 Phase B2.
//!
//! Restores a live [`Model`] (and the [`AttackGraph`] derived from it) to
//! a previously-captured snapshot, by computing the minimal asset/
//! association diff and applying it via the same
//! `Model::add_asset`/`remove_asset`/`add_associated_assets`/
//! `remove_associated_assets` + `AttackGraph::partially_regenerate_graph`
//! primitives B1 already uses.
//!
//! **Snapshot type diverges from Python's `dict[str, Any]` (`Model::
//! to_dict()`'s JSON shape), and deliberately isn't ported as-is - see
//! §10.** Python's `reconcile_model_to_snapshot` computes its diff by
//! comparing `frozenset(other.id for other in others)` (asset ids, as
//! Python `int`s) against `frozenset(other_ids)` where `other_ids` is a
//! `dict[str, str]` (`{"<id>": "<name>"}`) - iterating a dict yields its
//! *keys*, so that frozenset holds `str`s, not `int`s. The two frozensets
//! can never intersect, so every reconciliation silently falls back to a
//! full remove-and-re-add of every association on every call - the
//! observable end state is still correct (a full teardown-and-rebuild
//! reaches the same target), just never the "minimal diff" the function's
//! own docstring promises. Rust's [`ModelSnapshot`] stores association
//! targets as `i64` on both sides (no JSON round-trip, no string keys),
//! so this type mismatch cannot exist here - porting the bug faithfully
//! would mean deliberately mistyping one side, which contradicts writing
//! correct Rust. This function therefore implements the *intended*
//! minimal-diff behavior the Python docstring describes, not its actual
//! current behavior.

use std::collections::{HashMap, HashSet};
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
use maltoolbox_model::{AssetSnapshot, Model, ModelError};

#[derive(Debug, Clone)]
pub struct ModelSnapshotAsset {
    pub asset_type: String,
    pub name: String,
    pub associated_assets: HashMap<String, HashSet<i64>>,
}

/// Asset id -> snapshot of that asset's type/name/associations, mirroring
/// `Model.to_dict()['assets']`'s shape minus the JSON string-id detour -
/// see this module's doc comment.
pub type ModelSnapshot = HashMap<i64, ModelSnapshotAsset>;

#[derive(Debug)]
pub enum ModelStateError {
    Model(ModelError),
    Graph(maltoolbox_attackgraph::GraphError),
}

impl From<ModelError> for ModelStateError {
    fn from(e: ModelError) -> Self {
        ModelStateError::Model(e)
    }
}

impl From<maltoolbox_attackgraph::GraphError> for ModelStateError {
    fn from(e: maltoolbox_attackgraph::GraphError) -> Self {
        ModelStateError::Graph(e)
    }
}

impl fmt::Display for ModelStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelStateError::Model(e) => write!(f, "{e}"),
            ModelStateError::Graph(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ModelStateError {}

/// Captures `model`'s current state as a [`ModelSnapshot`] - the Rust
/// equivalent of Python's `attack_graph.model.to_dict()` call at
/// `DynaMalSimulator.__init__` time (see `dyna_mal_simulator/
/// simulator.py`), taken once and restored on every `reset()` via
/// `reset_model_effects`.
pub fn capture_model_snapshot(model: &Model) -> ModelSnapshot {
    model
        .assets
        .iter()
        .map(|(&id, asset)| {
            (
                id,
                ModelSnapshotAsset {
                    asset_type: asset.asset_type.clone(),
                    name: asset.name.clone(),
                    associated_assets: asset.associated_assets.clone(),
                },
            )
        })
        .collect()
}

/// `(new_assets, removed_assets, new_associations, removed_associations)` -
/// the same shape `AttackGraph::partially_regenerate_graph` takes, and B1's
/// `execute_model_effects` builds for the same call.
pub type ModelReconciliation = (
    HashSet<i64>,
    HashMap<i64, AssetSnapshot>,
    HashSet<(i64, String, i64)>,
    HashSet<(i64, String, i64)>,
);

/// Port of `reconcile_model_to_snapshot`. Mutates `model` in place to
/// match `snapshot` exactly and returns a [`ModelReconciliation`].
pub fn reconcile_model_to_snapshot(
    model: &mut Model,
    snapshot: &ModelSnapshot,
) -> Result<ModelReconciliation, ModelError> {
    let current_ids: HashSet<i64> = model.assets.keys().copied().collect();
    let target_ids: HashSet<i64> = snapshot.keys().copied().collect();

    let mut new_assets = HashSet::new();
    let mut removed_assets: HashMap<i64, AssetSnapshot> = HashMap::new();

    for asset_id in current_ids.difference(&target_ids) {
        let snap = model.remove_asset(*asset_id)?;
        removed_assets.insert(*asset_id, snap);
    }

    for &asset_id in target_ids.difference(&current_ids) {
        let asset_snapshot = &snapshot[&asset_id];
        model.add_asset(
            &asset_snapshot.asset_type,
            Some(asset_snapshot.name.clone()),
            Some(asset_id),
            None,
            None,
            false,
        )?;
        new_assets.insert(asset_id);
    }

    let mut new_associations = HashSet::new();
    let mut removed_associations = HashSet::new();

    for (&asset_id, asset_snapshot) in snapshot {
        let current_associations: HashMap<String, HashSet<i64>> = model
            .get_asset_by_id(asset_id)
            .map(|a| a.associated_assets.clone())
            .unwrap_or_default();
        let target_associations = &asset_snapshot.associated_assets;

        let mut field_names: HashSet<&String> = current_associations.keys().collect();
        field_names.extend(target_associations.keys());

        for field_name in field_names {
            let current_for_field = current_associations
                .get(field_name)
                .cloned()
                .unwrap_or_default();
            let target_for_field = target_associations
                .get(field_name)
                .cloned()
                .unwrap_or_default();

            let ids_to_remove: HashSet<i64> = current_for_field
                .difference(&target_for_field)
                .copied()
                .collect();
            if !ids_to_remove.is_empty() {
                model.remove_associated_assets(asset_id, field_name, &ids_to_remove)?;
                for &other in &ids_to_remove {
                    removed_associations.insert((asset_id, field_name.clone(), other));
                }
            }

            let ids_to_add: HashSet<i64> = target_for_field
                .difference(&current_for_field)
                .copied()
                .collect();
            if !ids_to_add.is_empty() {
                model.add_associated_assets(asset_id, field_name, ids_to_add.clone())?;
                for &other in &ids_to_add {
                    new_associations.insert((asset_id, field_name.clone(), other));
                }
            }
        }
    }

    Ok((
        new_assets,
        removed_assets,
        new_associations,
        removed_associations,
    ))
}

/// Port of `reset_model_effects`. Python's version returns `None` - its
/// one caller (`dyna_reset`) always rebuilds `GraphState` from scratch via
/// `compute_initial_graph_state` afterwards rather than folding in new
/// nodes incrementally, so the new-node ids go unused there. This port
/// returns them anyway (same shape as `execute_model_effects`'s return)
/// since discarding them costs the caller nothing and some callers may
/// want them.
pub fn reset_model_effects(
    graph: &mut AttackGraph,
    model: &mut Model,
    snapshot: &ModelSnapshot,
) -> Result<HashSet<AttackGraphNodeId>, ModelStateError> {
    let (new_assets, removed_assets, new_associations, removed_associations) =
        reconcile_model_to_snapshot(model, snapshot)?;
    let new_nodes = graph.partially_regenerate_graph(
        model,
        &new_assets,
        &new_associations,
        &removed_assets,
        &removed_associations,
    )?;
    Ok(new_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_effects::execute_model_effects;
    use crate::test_fixtures::wiper_attack_graph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(5)
    }

    #[test]
    fn reconcile_model_to_snapshot_restores_removed_and_re_adds_associations() {
        let (_graph, mut model) = wiper_attack_graph();
        let snapshot = capture_model_snapshot(&model);

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

        let c2_server = model.get_asset_by_name("C2Server").unwrap().id;
        model
            .remove_associated_assets(infected_device, "sendTo", &HashSet::from([c2_server]))
            .unwrap();

        let (new_assets, removed_assets, new_associations, _removed_associations) =
            reconcile_model_to_snapshot(&mut model, &snapshot).unwrap();

        assert!(new_assets.is_empty());
        assert!(removed_assets.contains_key(&wiper_id));
        // The InfectedDevice<->C2Server link is bidirectional
        // (sendTo/receiveFrom are opposite fieldnames of the same
        // association) - `Model::add_associated_assets` restores both
        // sides as soon as either is re-added, so depending on `snapshot`'s
        // `HashMap` iteration order, *either* `(infected_device, "sendTo",
        // c2_server)` *or* `(c2_server, "receiveFrom", infected_device)`
        // ends up the one explicitly recorded here - whichever asset's
        // turn came first in the reconciliation loop. Assert on the
        // resulting model state instead of which side got the credit.
        assert!(
            new_associations.contains(&(infected_device, "sendTo".to_string(), c2_server))
                || new_associations.contains(&(
                    c2_server,
                    "receiveFrom".to_string(),
                    infected_device
                ))
        );

        assert!(
            model.get_asset_by_id(wiper_id).is_none(),
            "Wiper should be gone after reconciling"
        );
        let vulnerable_device = model.get_asset_by_name("VulnerableDevice").unwrap().id;
        assert_eq!(
            model
                .get_asset_by_id(infected_device)
                .unwrap()
                .associated_assets
                .get("sendTo"),
            Some(&HashSet::from([c2_server, vulnerable_device]))
        );
    }

    #[test]
    fn reset_model_effects_restores_graph_to_pristine_snapshot() {
        let (mut graph, mut model) = wiper_attack_graph();
        let snapshot = capture_model_snapshot(&model);
        let pristine_names: HashSet<String> = graph.full_name_to_node.keys().cloned().collect();

        let infect_id = graph.full_name_to_node["InfectedDevice:infect"];
        let mut r = rng();
        execute_model_effects(&mut graph, &mut model, infect_id, &mut r).unwrap();
        let grown_names: HashSet<String> = graph.full_name_to_node.keys().cloned().collect();
        assert_ne!(
            grown_names, pristine_names,
            "infect's model effect should have grown the graph"
        );

        reset_model_effects(&mut graph, &mut model, &snapshot).unwrap();
        let restored_names: HashSet<String> = graph.full_name_to_node.keys().cloned().collect();
        assert_eq!(restored_names, pristine_names);
    }
}
