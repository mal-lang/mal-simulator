//! Rust port of `python/malsim/dyna_mal_simulator/model_effects.py` - see
//! `PORTING_NOTES.md` §6 Phase B1.
//!
//! Applies a [`LanguageGraphModelEffect`] (mal-toolbox's already-parsed
//! model-effect declaration) to a live [`Model`], mutating it via
//! `Model::add_asset`/`remove_asset`/`add_associated_assets`/
//! `remove_associated_assets` and the attack graph via
//! `AttackGraph::partially_regenerate_graph` - no new graph-mutation
//! primitives invented here, per §6's framing: those already exist
//! upstream in mal-toolbox.
//!
//! **Deliberate split from Python's `execute_model_effects`:** this
//! module's [`execute_model_effects`] stops at `partially_regenerate_graph`
//! and returns the new node ids it created, rather than also folding them
//! into a `GraphState` (Python's version calls dyna `graph_state.py`'s
//! `add_new_nodes_to_graph_state` inline). That composition is Phase B2's
//! job (`PORTING_NOTES.md` §6's own module split - `model_effects.py` vs.
//! dyna `graph_state.py`); B1 shouldn't reach into B2-scoped code, so the
//! caller (B2's `dyna_attacker_step`/`dyna_defender_step`) does that fold
//! itself using this function's returned node ids.

use std::collections::{HashMap, HashSet};
use std::fmt;

use maltoolbox_attackgraph::{AttackGraph, AttackGraphNode, AttackGraphNodeId};
use maltoolbox_language::graph::model_effect::{
    DynTarget, LanguageGraphModelEffect, ModelEffectType,
};
use maltoolbox_model::{AssetSnapshot, Model, ModelError};
use rand::Rng;

use crate::assoc_traversal::{
    apply_quantity_filter, parse_addition, parse_removal, sample_size, traverse_association_chain,
    AdditionTarget, AssocTraversalError,
};

#[derive(Debug)]
pub enum ModelEffectsError {
    AssocTraversal(AssocTraversalError),
    Model(ModelError),
    Graph(maltoolbox_attackgraph::GraphError),
    /// Mirrors the bare `assert node.model_asset` guard in
    /// `_apply_model_effect`/`execute_model_effects`.
    NodeHasNoModelAsset,
    /// Mirrors Python's `raise ValueError(f'Target association traversal
    /// cannot be empty for step {node.full_name}.')`.
    EmptyTargetTraversal(String),
    /// Mirrors Python's two (identically-reached) `raise ValueError(...)`
    /// branches in `add_asset` for a target that resolved to an
    /// *existing* model asset instead of a type to instantiate - both
    /// branches raise regardless of which asset it is, so this one
    /// variant covers both.
    CannotAddExistingAsset(i64),
}

impl From<AssocTraversalError> for ModelEffectsError {
    fn from(e: AssocTraversalError) -> Self {
        ModelEffectsError::AssocTraversal(e)
    }
}

impl From<ModelError> for ModelEffectsError {
    fn from(e: ModelError) -> Self {
        ModelEffectsError::Model(e)
    }
}

impl From<maltoolbox_attackgraph::GraphError> for ModelEffectsError {
    fn from(e: maltoolbox_attackgraph::GraphError) -> Self {
        ModelEffectsError::Graph(e)
    }
}

impl fmt::Display for ModelEffectsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelEffectsError::AssocTraversal(e) => write!(f, "{e}"),
            ModelEffectsError::Model(e) => write!(f, "{e}"),
            ModelEffectsError::Graph(e) => write!(f, "{e}"),
            ModelEffectsError::NodeHasNoModelAsset => {
                write!(f, "attack graph node does not have a model asset")
            }
            ModelEffectsError::EmptyTargetTraversal(full_name) => {
                write!(
                    f,
                    "target association traversal cannot be empty for step {full_name}"
                )
            }
            ModelEffectsError::CannotAddExistingAsset(asset_id) => write!(
                f,
                "cannot add an existing asset ({asset_id}) to the model - it already exists"
            ),
        }
    }
}

impl std::error::Error for ModelEffectsError {}

/// A self-contained reference to a model asset (id + type + name),
/// captured at the moment a [`ModEffectOp`] is recorded rather than
/// resolved later from the id alone - see `PORTING_NOTES.md` §6 Phase
/// B4/B5: a later op in the *same* modification record can remove the
/// very asset an earlier op referenced (directly, e.g. an asset added
/// then removed within one record, or indirectly, e.g. `remove_asset_op`
/// below records an asset's about-to-be-removed associations before
/// removing the asset itself), so by the time a caller across the FFI
/// boundary reads the full record, `model.assets[id]` may no longer
/// resolve for an id this type already described - carrying the
/// snapshot inline avoids needing a live lookup at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetRef {
    pub id: i64,
    pub asset_type: String,
    pub name: String,
}

impl AssetRef {
    pub(crate) fn from_model(model: &Model, id: i64) -> Self {
        let asset = model
            .get_asset_by_id(id)
            .expect("AssetRef::from_model called with an id missing from the live model");
        AssetRef {
            id,
            asset_type: asset.asset_type.clone(),
            name: asset.name.clone(),
        }
    }
}

/// A single modification applied to the model, as a result of executing
/// one model effect. `AssetOp::Removed` additionally carries a full
/// `AssetSnapshot` (not just the `AssetRef` every variant has) to feed
/// `AttackGraph::partially_regenerate_graph` directly - see
/// `execute_model_effects`.
#[derive(Debug, Clone)]
pub enum AssetOp {
    Added {
        asset: AssetRef,
    },
    Removed {
        asset: AssetRef,
        snapshot: Box<AssetSnapshot>,
    },
}

#[derive(Debug, Clone)]
pub enum AssocOp {
    Added {
        left: AssetRef,
        field_name: String,
        right: AssetRef,
    },
    Removed {
        left: AssetRef,
        field_name: String,
        right: AssetRef,
    },
}

#[derive(Debug, Clone)]
pub enum ModEffectOp {
    Asset(AssetOp),
    Assoc(AssocOp),
}

impl ModEffectOp {
    pub fn effect_type(&self) -> ModelEffectType {
        match self {
            ModEffectOp::Asset(AssetOp::Added { .. })
            | ModEffectOp::Assoc(AssocOp::Added { .. }) => ModelEffectType::Additive,
            ModEffectOp::Asset(AssetOp::Removed { .. })
            | ModEffectOp::Assoc(AssocOp::Removed { .. }) => ModelEffectType::Subtractive,
        }
    }
}

/// Port of `target_op`'s dispatch (the `if`/`elif` chain at the bottom of
/// the Python function, not the four closures themselves - see
/// `add_asset_op`/`remove_asset_op`/`add_assoc_op`/`remove_assoc_op`
/// below). `ModelEffectType`/`bool` is exhaustively matched by this 2x2,
/// so Python's trailing `else: raise ValueError(...)` has no Rust
/// equivalent - unreachable by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetKind {
    AddAsset,
    RemoveAsset,
    AddAssoc,
    RemoveAssoc,
}

fn classify(model_effect_type: ModelEffectType, dyn_target: &DynTarget) -> TargetKind {
    match (model_effect_type, dyn_target.assoc_op) {
        (ModelEffectType::Additive, false) => TargetKind::AddAsset,
        (ModelEffectType::Subtractive, false) => TargetKind::RemoveAsset,
        (ModelEffectType::Additive, true) => TargetKind::AddAssoc,
        (ModelEffectType::Subtractive, true) => TargetKind::RemoveAssoc,
    }
}

/// Port of `target_op`'s `add_asset` closure.
fn add_asset_op(
    model: &mut Model,
    node_asset_id: i64,
    base: &HashSet<i64>,
    target: &DynTarget,
    rng: &mut impl Rng,
) -> Result<Vec<ModEffectOp>, ModelEffectsError> {
    let (addition_info, quantity) =
        parse_addition(model, node_asset_id, base, &target.assoc_traversal, rng)?;
    let size = sample_size(quantity, rng, None);
    let mut record = Vec::new();
    for (left_asset_id, field_name, right) in addition_info {
        for _ in 0..size {
            let asset_type = match right {
                AdditionTarget::ExistingAsset(existing_id) => {
                    return Err(ModelEffectsError::CannotAddExistingAsset(existing_id));
                }
                AdditionTarget::NewAssetType(asset_type) => asset_type,
            };
            let type_name = model.lang_graph.asset(asset_type).name.clone();
            let new_name = format!("{type_name}-{}", model.next_id);
            let new_id = model.add_asset(&type_name, Some(new_name), None, None, None, false)?;

            match model.add_associated_assets(left_asset_id, &field_name, HashSet::from([new_id])) {
                Ok(()) => {
                    let new_asset = AssetRef::from_model(model, new_id);
                    let left = AssetRef::from_model(model, left_asset_id);
                    record.push(ModEffectOp::Asset(AssetOp::Added {
                        asset: new_asset.clone(),
                    }));
                    record.push(ModEffectOp::Assoc(AssocOp::Added {
                        left,
                        field_name: field_name.clone(),
                        right: new_asset,
                    }));
                }
                Err(_) => {
                    // Mirrors Python's `logger.error(...)` + undo the
                    // just-created asset + `continue`.
                    model.remove_asset(new_id)?;
                }
            }
        }
    }
    Ok(record)
}

/// Port of `target_op`'s `remove_asset` closure.
fn remove_asset_op(
    model: &mut Model,
    node_asset_id: i64,
    base: &HashSet<i64>,
    target: &DynTarget,
    rng: &mut impl Rng,
) -> Result<Vec<ModEffectOp>, ModelEffectsError> {
    let (removal_info, quantity) =
        parse_removal(model, node_asset_id, base, &target.assoc_traversal, rng)?;
    let removal_info = match quantity {
        Some(q) => apply_quantity_filter(&removal_info, Some(q), rng),
        None => removal_info,
    };

    let mut record = Vec::new();
    let mut removal_assets: HashSet<i64> = HashSet::new();
    for (_left, _field_name, right_id) in &removal_info {
        let right_id = *right_id;
        if let Some(asset) = model.get_asset_by_id(right_id) {
            // Captured here, before `remove_asset` below - by the time a
            // caller across the FFI boundary reads this record, `right_id`
            // itself is gone from the live model (see `AssetRef`'s doc
            // comment), so its `AssetRef` must be built while it's still
            // live, same as the asset's own removal snapshot further down.
            let left_ref = AssetRef::from_model(model, right_id);
            // Snapshot both the dict and its sets: `remove_associated_assets`
            // mutates the live `associated_assets` sets in place.
            let snapshot: Vec<(String, HashSet<i64>)> = asset
                .associated_assets
                .iter()
                .map(|(field_name, assoc_assets)| (field_name.clone(), assoc_assets.clone()))
                .collect();
            for (field_name, assoc_assets) in snapshot {
                if model
                    .remove_associated_assets(right_id, &field_name, &assoc_assets)
                    .is_ok()
                {
                    for assoc_id in assoc_assets {
                        let right_ref = AssetRef::from_model(model, assoc_id);
                        record.push(ModEffectOp::Assoc(AssocOp::Removed {
                            left: left_ref.clone(),
                            field_name: field_name.clone(),
                            right: right_ref,
                        }));
                    }
                }
                // else: mirrors Python's `logger.error(...)` + continue.
            }
        }
        removal_assets.insert(right_id);
    }

    for right_id in removal_assets {
        let asset_ref = AssetRef::from_model(model, right_id);
        let snapshot = model.remove_asset(right_id)?;
        record.push(ModEffectOp::Asset(AssetOp::Removed {
            asset: asset_ref,
            snapshot: Box::new(snapshot),
        }));
    }
    Ok(record)
}

/// Port of `target_op`'s `add_assoc` closure. Traverses from
/// `{node_asset_id}` (not `base`) - `base` is only used below as the
/// candidate pool when the target resolves to a *type* rather than an
/// existing asset, same as Python.
fn add_assoc_op(
    model: &mut Model,
    node_asset_id: i64,
    base: &HashSet<i64>,
    target: &DynTarget,
    rng: &mut impl Rng,
) -> Result<Vec<ModEffectOp>, ModelEffectsError> {
    let instigating = HashSet::from([node_asset_id]);
    let (addition_info, quantity) = parse_addition(
        model,
        node_asset_id,
        &instigating,
        &target.assoc_traversal,
        rng,
    )?;
    let addition_info = match quantity {
        Some(q) => apply_quantity_filter(&addition_info, Some(q), rng),
        None => addition_info,
    };

    let mut record = Vec::new();
    for (left_id, field_name, right) in addition_info {
        match right {
            AdditionTarget::ExistingAsset(right_id) => {
                let already_associated = model
                    .get_asset_by_id(left_id)
                    .and_then(|a| a.associated_assets.get(&field_name))
                    .map(|s| s.contains(&right_id))
                    .unwrap_or(false);
                if already_associated {
                    continue;
                }
                model.add_associated_assets(left_id, &field_name, HashSet::from([right_id]))?;
                record.push(ModEffectOp::Assoc(AssocOp::Added {
                    left: AssetRef::from_model(model, left_id),
                    field_name,
                    right: AssetRef::from_model(model, right_id),
                }));
            }
            AdditionTarget::NewAssetType(_) => {
                for &candidate_id in base {
                    let already_associated = model
                        .get_asset_by_id(left_id)
                        .and_then(|a| a.associated_assets.get(&field_name))
                        .map(|s| s.contains(&candidate_id))
                        .unwrap_or(false);
                    if already_associated {
                        continue;
                    }
                    if model
                        .add_associated_assets(left_id, &field_name, HashSet::from([candidate_id]))
                        .is_ok()
                    {
                        record.push(ModEffectOp::Assoc(AssocOp::Added {
                            left: AssetRef::from_model(model, left_id),
                            field_name: field_name.clone(),
                            right: AssetRef::from_model(model, candidate_id),
                        }));
                    }
                    // else: mirrors Python's `logger.error(...)` + continue.
                }
            }
        }
    }
    Ok(record)
}

/// Port of `target_op`'s `remove_assoc` closure. Traverses from `base`
/// (not `{node_asset_id}`), unlike `add_assoc_op` above - matches Python.
fn remove_assoc_op(
    model: &mut Model,
    node_asset_id: i64,
    base: &HashSet<i64>,
    target: &DynTarget,
    rng: &mut impl Rng,
) -> Result<Vec<ModEffectOp>, ModelEffectsError> {
    let (removal_info, quantity) =
        parse_removal(model, node_asset_id, base, &target.assoc_traversal, rng)?;
    let removal_info = match quantity {
        Some(q) => apply_quantity_filter(&removal_info, Some(q), rng),
        None => removal_info,
    };

    let mut record = Vec::new();
    for (left_id, field_name, right_id) in removal_info {
        let has_field = model
            .get_asset_by_id(left_id)
            .map(|a| a.associated_assets.contains_key(&field_name))
            .unwrap_or(false);
        if has_field {
            let left_ref = AssetRef::from_model(model, left_id);
            let right_ref = AssetRef::from_model(model, right_id);
            model.remove_associated_assets(left_id, &field_name, &HashSet::from([right_id]))?;
            record.push(ModEffectOp::Assoc(AssocOp::Removed {
                left: left_ref,
                field_name,
                right: right_ref,
            }));
        }
        // else: mirrors Python's `logger.error(...)` + continue.
    }
    Ok(record)
}

/// Port of `_apply_model_effect`.
pub fn apply_model_effect(
    model: &mut Model,
    node: &AttackGraphNode,
    model_effect: &LanguageGraphModelEffect,
    rng: &mut impl Rng,
) -> Result<Vec<ModEffectOp>, ModelEffectsError> {
    let node_asset_id = node
        .model_asset
        .ok_or(ModelEffectsError::NodeHasNoModelAsset)?;
    let base_set = traverse_association_chain(
        model,
        &HashSet::from([node_asset_id]),
        &model_effect.base,
        rng,
    )?;

    let mut record = Vec::new();
    for dyn_target in &model_effect.targets {
        if dyn_target.assoc_traversal.is_empty() {
            return Err(ModelEffectsError::EmptyTargetTraversal(node.name.clone()));
        }
        let op_result = match classify(model_effect.model_effect_type, dyn_target) {
            TargetKind::AddAsset => add_asset_op(model, node_asset_id, &base_set, dyn_target, rng)?,
            TargetKind::RemoveAsset => {
                remove_asset_op(model, node_asset_id, &base_set, dyn_target, rng)?
            }
            TargetKind::AddAssoc => add_assoc_op(model, node_asset_id, &base_set, dyn_target, rng)?,
            TargetKind::RemoveAssoc => {
                remove_assoc_op(model, node_asset_id, &base_set, dyn_target, rng)?
            }
        };
        record.extend(op_result);
    }
    Ok(record)
}

/// Port of `execute_model_effects`, minus the `add_new_nodes_to_graph_state`
/// fold - see this module's doc comment for why. Returns the full
/// modification record plus the set of newly-created node ids from
/// `partially_regenerate_graph`, which the caller (Phase B2) folds into
/// its `GraphState`.
pub fn execute_model_effects(
    graph: &mut AttackGraph,
    model: &mut Model,
    action_id: AttackGraphNodeId,
    rng: &mut impl Rng,
) -> Result<(Vec<ModEffectOp>, HashSet<AttackGraphNodeId>), ModelEffectsError> {
    let mut record = Vec::new();
    {
        let node = &graph.nodes[action_id];
        let additive = node.additive_model_effects.as_deref().unwrap_or(&[]);
        for model_effect in additive {
            record.extend(apply_model_effect(model, node, model_effect, rng)?);
        }
        let subtractive = node.subtractive_model_effects.as_deref().unwrap_or(&[]);
        for model_effect in subtractive {
            record.extend(apply_model_effect(model, node, model_effect, rng)?);
        }
    }

    let mut new_assets = HashSet::new();
    let mut removed_assets: HashMap<i64, AssetSnapshot> = HashMap::new();
    let mut new_associations = HashSet::new();
    let mut removed_associations = HashSet::new();
    for op in &record {
        match op {
            ModEffectOp::Asset(AssetOp::Added { asset }) => {
                new_assets.insert(asset.id);
            }
            ModEffectOp::Asset(AssetOp::Removed { asset, snapshot }) => {
                removed_assets.insert(asset.id, (**snapshot).clone());
            }
            ModEffectOp::Assoc(AssocOp::Added {
                left,
                field_name,
                right,
            }) => {
                new_associations.insert((left.id, field_name.clone(), right.id));
            }
            ModEffectOp::Assoc(AssocOp::Removed {
                left,
                field_name,
                right,
            }) => {
                removed_associations.insert((left.id, field_name.clone(), right.id));
            }
        }
    }

    let new_nodes = graph.partially_regenerate_graph(
        model,
        &new_assets,
        &new_associations,
        &removed_assets,
        &removed_associations,
    )?;
    Ok((record, new_nodes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::wiper_attack_graph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    /// Direct port of
    /// `test_dyna_mal_simulator.py::test_apply_model_effect`.
    #[test]
    fn apply_model_effect_infect_then_exfiltrate() {
        let (mut graph, mut model) = wiper_attack_graph();
        let infected_device = model.get_asset_by_name("InfectedDevice").unwrap().id;
        assert!(!model
            .get_asset_by_id(infected_device)
            .unwrap()
            .associated_assets
            .contains_key("malware"));

        let infect_id = graph.full_name_to_node["InfectedDevice:infect"];
        let infect_effect = graph.nodes[infect_id]
            .additive_model_effects
            .as_ref()
            .unwrap()[0]
            .clone();
        let mut r = rng();
        let record =
            apply_model_effect(&mut model, &graph.nodes[infect_id], &infect_effect, &mut r)
                .unwrap();

        assert!(model
            .get_asset_by_id(infected_device)
            .unwrap()
            .associated_assets
            .contains_key("malware"));
        let wiper_id = model.get_asset_by_name("Wiper-7").unwrap().id;
        assert!(record.iter().any(|op| matches!(
            op,
            ModEffectOp::Assoc(AssocOp::Added { left, field_name, right })
                if left.id == infected_device && field_name == "malware" && right.id == wiper_id
        )));
        assert!(model
            .get_asset_by_id(wiper_id)
            .unwrap()
            .associated_assets
            .get("victim")
            .map(|s| s.contains(&infected_device))
            .unwrap_or(false));
        assert!(record.iter().any(
            |op| matches!(op, ModEffectOp::Asset(AssetOp::Added { asset }) if asset.id == wiper_id)
        ));

        let infected_data = model.get_asset_by_name("InfectedData").unwrap().id;
        assert_eq!(
            model
                .get_asset_by_id(infected_data)
                .unwrap()
                .associated_assets
                .get("node"),
            Some(&HashSet::from([infected_device]))
        );

        graph.regenerate_graph(&model).unwrap();
        let exfiltrate_id = graph.full_name_to_node["Wiper-7:exfiltrate"];
        let exfiltrate_effect = graph.nodes[exfiltrate_id]
            .additive_model_effects
            .as_ref()
            .unwrap()[0]
            .clone();
        let record2 = apply_model_effect(
            &mut model,
            &graph.nodes[exfiltrate_id],
            &exfiltrate_effect,
            &mut r,
        )
        .unwrap();
        let c2_server = model.get_asset_by_name("C2Server").unwrap().id;
        assert_eq!(
            model
                .get_asset_by_id(infected_data)
                .unwrap()
                .associated_assets
                .get("node"),
            Some(&HashSet::from([infected_device, c2_server]))
        );
        assert!(record2.iter().any(|op| matches!(
            op,
            ModEffectOp::Assoc(AssocOp::Added { left, field_name, right })
                if left.id == c2_server && field_name == "data" && right.id == infected_data
        )));
    }

    /// Full-name-based structural comparison of two attack graphs, mirroring
    /// `test_dyna_mal_simulator.py::check_graph_equivalence` - used to
    /// confirm `execute_model_effects`' incremental
    /// `partially_regenerate_graph` call produces the exact same graph a
    /// fresh `AttackGraph::from_model` build would, after the same model
    /// mutation. Spiritual counterpart to
    /// `test_apply_model_effect_modification_record_partially_regenerates_graph`,
    /// using wiperLang (hand-built) instead of the
    /// `dynamic_remove_many_assoc` scenario (needs Phase C's not-yet-ported
    /// scenario-YAML loader to build in Rust).
    fn assert_graph_equivalent(incremental: &AttackGraph, fresh: &AttackGraph) {
        let incremental_names: HashSet<&String> = incremental.full_name_to_node.keys().collect();
        let fresh_names: HashSet<&String> = fresh.full_name_to_node.keys().collect();
        assert_eq!(incremental_names, fresh_names, "node full_names differ");

        let edge_names = |g: &AttackGraph, ids: &HashSet<AttackGraphNodeId>| -> HashSet<String> {
            ids.iter().map(|&id| g.nodes[id].name.clone()).collect()
        };
        for (full_name, &incr_id) in &incremental.full_name_to_node {
            let fresh_id = fresh.full_name_to_node[full_name];
            assert_eq!(
                edge_names(incremental, &incremental.nodes[incr_id].children),
                edge_names(fresh, &fresh.nodes[fresh_id].children),
                "different children for {full_name}"
            );
            assert_eq!(
                edge_names(incremental, &incremental.nodes[incr_id].parents),
                edge_names(fresh, &fresh.nodes[fresh_id].parents),
                "different parents for {full_name}"
            );
        }
    }

    /// Exercises the same property
    /// `test_apply_model_effect_modification_record_partially_regenerates_graph`
    /// does - that applying a subtractive model effect's modification
    /// record through `partially_regenerate_graph` leaves the graph
    /// equivalent to a fresh rebuild from the mutated model - via
    /// `Wiper:trigger`'s `R> victim / malware ^ ~data` effect instead of
    /// that test's `dynamic_remove_many_assoc` scenario fixture.
    #[test]
    fn execute_model_effects_removal_matches_fresh_graph_rebuild() {
        let (_, mut model) = wiper_attack_graph();
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

        let mut graph = AttackGraph::from_model(&model).unwrap();
        let trigger_id = graph.full_name_to_node["Wiper:trigger"];
        let mut r = rng();

        let (record, _new_nodes) =
            execute_model_effects(&mut graph, &mut model, trigger_id, &mut r).unwrap();
        assert!(
            record
                .iter()
                .any(|op| matches!(op, ModEffectOp::Asset(AssetOp::Removed { .. }))),
            "Wiper:trigger should remove the Wiper asset itself"
        );

        let fresh_graph = AttackGraph::from_model(&model).unwrap();
        assert_graph_equivalent(&graph, &fresh_graph);
    }
}
