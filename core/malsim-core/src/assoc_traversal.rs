//! Rust port of `python/malsim/dyna_mal_simulator/process_assoc_traversal.py`
//! - see `PORTING_NOTES.md` §6 Phase B1.
//!
//! Evaluates an already-parsed [`AssocTraversalChain`] (mal-toolbox's
//! declaration grammar for a model effect's `base`/`target` expression)
//! against a live [`Model`]'s actual asset-association links - walking
//! `associated_assets` by asset id, sampling quantity filters via `rng`.
//! This is deliberately *not* the same traversal mal-toolbox's own
//! `maltoolbox-language::graph::assoc_traversal` module implements: that
//! module walks the *type*-level `LanguageGraph` (by `AssetId`) to
//! validate a model effect's declaration at language-compile time: this
//! module walks actual model *instances* (by `i64` asset id) at
//! simulation runtime, with RNG-driven quantity filtering the type-level
//! validator never needs. The two intentionally share no code.

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

use maltoolbox_language::graph::ids::AssetId;
use maltoolbox_language::graph::model_effect::{
    AssocSet, AssocTraversal, AssocTraversalElem, GlobAssocTraversal, QuantityFilter, SetOperation,
};
use maltoolbox_model::Model;
use rand::seq::IndexedRandom;
use rand::{Rng, RngExt};

#[derive(Debug, Clone, PartialEq)]
pub enum AssocTraversalError {
    /// Mirrors Python's `raise ValueError('Association traversal chain
    /// did not terminate in an AssocTraversal.')` in
    /// `_resolve_terminal_traversal`.
    EmptyChain,
    /// Mirrors Python's `raise NotImplementedError('Quantity filtering is
    /// not supported set operations in terminal fields.')`.
    QuantityFilterUnsupportedInTerminal,
    /// Mirrors the bare `assert node.model_asset` guard every
    /// `parse_addition`/`parse_removal` caller in `model_effects.py` relies
    /// on before calling them.
    NodeHasNoModelAsset,
    /// Mirrors Python's `raise ValueError(f'{asset.name} does not have a
    /// valid association to the node's model asset {node.model_asset.name},
    /// so cannot traverse to self.')` in `parse_addition`'s terminating
    /// expression for `field_name == 'self'`.
    NoSelfAssociation { asset_id: i64, node_asset_id: i64 },
    /// Mirrors Python's bare `asset.lg_asset.associations[field_name]`
    /// `KeyError` in `parse_addition`'s terminating expression (no
    /// `asset_filter` and no declared association for `field_name`).
    UnknownAssociation { asset_id: i64, field_name: String },
}

impl fmt::Display for AssocTraversalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssocTraversalError::EmptyChain => {
                write!(f, "association traversal chain did not terminate in an AssocTraversal")
            }
            AssocTraversalError::QuantityFilterUnsupportedInTerminal => write!(
                f,
                "quantity filtering is not supported for set operations in terminal fields"
            ),
            AssocTraversalError::NodeHasNoModelAsset => {
                write!(f, "attack graph node does not have a model asset")
            }
            AssocTraversalError::NoSelfAssociation { asset_id, node_asset_id } => write!(
                f,
                "asset {asset_id} does not have a valid association to the node's model asset {node_asset_id}, so cannot traverse to self"
            ),
            AssocTraversalError::UnknownAssociation { asset_id, field_name } => write!(
                f,
                "asset {asset_id} does not have an association for field \"{field_name}\""
            ),
        }
    }
}

impl std::error::Error for AssocTraversalError {}

/// Port of `sample_size`. `max_size: None` mirrors Python's default
/// `max_size=float('inf')`.
///
/// **The `quantity is None` branch returns `1` unconditionally, even when
/// `max_size` is given and is `0`** - ported as-is from Python, which
/// never clips this branch against `max_size` (only the `int`/tuple
/// branches do). Every real call site only reaches this branch with
/// `max_size` left at its default (infinite) - see `model_effects.rs`'s
/// `add_asset_op` - so the mismatch is latent, not reachable today.
pub fn sample_size(
    quantity: Option<QuantityFilter>,
    rng: &mut impl Rng,
    max_size: Option<i64>,
) -> i64 {
    match quantity {
        None => 1,
        Some(QuantityFilter::Exact(q)) => match max_size {
            Some(m) => q.min(m),
            None => q,
        },
        Some(QuantityFilter::Range(lo, hi)) => {
            let high = match max_size {
                Some(m) => hi.min(m),
                None => hi,
            };
            let low = lo.min(high);
            rng.random_range(low..=high)
        }
    }
}

/// Port of `_apply_quantity_filter`. Generic over the sampled element
/// type so it serves both the asset-id sets this module traverses and
/// the `Addition`/`Removal` tuple sets `model_effects.rs` filters.
pub fn apply_quantity_filter<T: Eq + Hash + Clone>(
    objects: &HashSet<T>,
    quantity: Option<QuantityFilter>,
    rng: &mut impl Rng,
) -> HashSet<T> {
    let items: Vec<T> = objects.iter().cloned().collect();
    let size = sample_size(quantity, rng, Some(items.len() as i64)).max(0) as usize;
    items.sample(rng, size).cloned().collect()
}

/// Port of `_assoc_traversal`.
///
/// **`field_name == "self"` returns `instigating_assets` unchanged**,
/// ignoring any `asset_filter`/`quantity_filter` that happen to also be
/// set on the same element - ported as-is from Python's early `return`,
/// which never reaches the rest of the function body in that case.
fn assoc_traversal(
    model: &Model,
    instigating_assets: &HashSet<i64>,
    traversal: &AssocTraversal,
    rng: &mut impl Rng,
) -> HashSet<i64> {
    if traversal.field_name == "self" {
        return instigating_assets.clone();
    }
    let mut next_assets = HashSet::new();
    for &asset_id in instigating_assets {
        let Some(asset) = model.get_asset_by_id(asset_id) else {
            continue;
        };
        // Mirrors Python's `logger.error(...)` + skip for an asset with no
        // such association field - not an error, just excluded.
        let Some(candidates) = asset.associated_assets.get(&traversal.field_name) else {
            continue;
        };
        let mut candidate_set: HashSet<i64> = candidates.clone();
        if let Some(filter_id) = traversal.asset_filter {
            candidate_set.retain(|&id| {
                model
                    .get_asset_by_id(id)
                    .map(|a| a.lg_asset == filter_id)
                    .unwrap_or(false)
            });
        }
        if traversal.quantity_filter.is_some() {
            candidate_set = apply_quantity_filter(&candidate_set, traversal.quantity_filter, rng);
        }
        next_assets.extend(candidate_set);
    }
    next_assets
}

/// Port of `_glob_assoc_traversal` - a transitive-closure fixed-point
/// loop over `glob.pattern`, re-applying the pattern to the growing
/// result set until it stops growing.
fn glob_assoc_traversal(
    model: &Model,
    instigating_assets: &HashSet<i64>,
    glob: &GlobAssocTraversal,
    rng: &mut impl Rng,
) -> Result<HashSet<i64>, AssocTraversalError> {
    let mut next_assets =
        traverse_association_chain(model, instigating_assets, &glob.pattern, rng)?;
    loop {
        let applied = traverse_association_chain(model, &next_assets, &glob.pattern, rng)?;
        let mut union = next_assets.clone();
        union.extend(applied);
        if union.len() == next_assets.len() {
            break;
        }
        next_assets = union;
    }
    if glob.quantity_filter.is_some() {
        next_assets = apply_quantity_filter(&next_assets, glob.quantity_filter, rng);
    }
    Ok(next_assets)
}

/// Port of `_assoc_set_traversal`. `SetOperation`'s Rust enum is
/// exhaustively matched, so Python's `else: raise ValueError('Unknown set
/// operation ...')` has no Rust equivalent - unreachable by construction.
fn assoc_set_traversal(
    model: &Model,
    starting_assets: &HashSet<i64>,
    assoc_set: &AssocSet,
    rng: &mut impl Rng,
) -> Result<HashSet<i64>, AssocTraversalError> {
    let left = traverse_association_chain(model, starting_assets, &assoc_set.left, rng)?;
    let right = traverse_association_chain(model, starting_assets, &assoc_set.right, rng)?;
    let mut candidate_assets: HashSet<i64> = match assoc_set.set_op {
        SetOperation::Union => left.union(&right).copied().collect(),
        SetOperation::Difference => left.difference(&right).copied().collect(),
        SetOperation::Intersection => left.intersection(&right).copied().collect(),
    };
    if assoc_set.quantity_filter.is_some() {
        candidate_assets = apply_quantity_filter(&candidate_assets, assoc_set.quantity_filter, rng);
    }
    Ok(candidate_assets)
}

/// Port of `traverse_association_chain`. `AssocTraversalElem`'s Rust enum
/// is exhaustively matched, so Python's `else: raise ValueError('Unknown
/// association traversal type ...')` has no Rust equivalent.
pub fn traverse_association_chain(
    model: &Model,
    instigating_assets: &HashSet<i64>,
    chain: &[AssocTraversalElem],
    rng: &mut impl Rng,
) -> Result<HashSet<i64>, AssocTraversalError> {
    let mut current_assets = instigating_assets.clone();
    for elem in chain {
        current_assets = match elem {
            AssocTraversalElem::Traversal(t) => assoc_traversal(model, &current_assets, t, rng),
            AssocTraversalElem::Glob(g) => glob_assoc_traversal(model, &current_assets, g, rng)?,
            AssocTraversalElem::Set(s) => assoc_set_traversal(model, &current_assets, s, rng)?,
        };
    }
    Ok(current_assets)
}

/// (anchor asset id, field name used to reach the terminal, right-hand
/// side resolved from [`parse_addition`]/[`parse_removal`]'s `terminate`
/// closures).
type TerminalResolve<T> = (HashSet<T>, Option<QuantityFilter>);

/// Port of `_resolve_terminal_traversal`. Generic over the terminal
/// element type `T` the caller's `terminate` closure produces -
/// `parse_addition`'s `Addition` or `parse_removal`'s `Removal`.
fn resolve_terminal_traversal<T, F>(
    model: &Model,
    instigating_assets: &HashSet<i64>,
    chain: &[AssocTraversalElem],
    rng: &mut impl Rng,
    terminate: &F,
) -> Result<TerminalResolve<T>, AssocTraversalError>
where
    T: Eq + Hash + Clone,
    F: Fn(&HashSet<i64>, &AssocTraversal) -> Result<TerminalResolve<T>, AssocTraversalError>,
{
    let Some((last, init)) = chain.split_last() else {
        return Err(AssocTraversalError::EmptyChain);
    };
    let current_assets = traverse_association_chain(model, instigating_assets, init, rng)?;
    match last {
        AssocTraversalElem::Traversal(t) => terminate(&current_assets, t),
        // `*` terminates in whatever its inner pattern terminates in.
        AssocTraversalElem::Glob(g) => {
            resolve_terminal_traversal(model, &current_assets, &g.pattern, rng, terminate)
        }
        AssocTraversalElem::Set(s) => {
            let (left, left_quantity) =
                resolve_terminal_traversal(model, &current_assets, &s.left, rng, terminate)?;
            let (right, right_quantity) =
                resolve_terminal_traversal(model, &current_assets, &s.right, rng, terminate)?;
            if left_quantity.is_some() || right_quantity.is_some() {
                return Err(AssocTraversalError::QuantityFilterUnsupportedInTerminal);
            }
            let terminal: HashSet<T> = match s.set_op {
                SetOperation::Union => left.union(&right).cloned().collect(),
                SetOperation::Difference => left.difference(&right).cloned().collect(),
                SetOperation::Intersection => left.intersection(&right).cloned().collect(),
            };
            Ok((terminal, None))
        }
    }
}

/// Right-hand side of a parsed addition: either an existing model asset
/// reached by traversal, or the *type* of a not-yet-created asset (when
/// no live association exists yet) - Python's `LanguageGraphAsset |
/// ModelAsset` union, translated to ids on both sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdditionTarget {
    ExistingAsset(i64),
    NewAssetType(AssetId),
}

/// (left asset id, field name, right-hand side).
pub type Addition = (i64, String, AdditionTarget);
/// (left asset id, field name, right asset id).
pub type Removal = (i64, String, i64);

/// Port of `parse_addition`'s inner `_parse_terminating_expr`.
fn parse_addition_terminal(
    model: &Model,
    node_asset_id: i64,
    instigating_assets: &HashSet<i64>,
    traversal: &AssocTraversal,
) -> Result<TerminalResolve<Addition>, AssocTraversalError> {
    let mut additions = HashSet::new();
    for &asset_id in instigating_assets {
        let Some(asset) = model.get_asset_by_id(asset_id) else {
            continue;
        };

        if traversal.field_name == "self" {
            let node_asset = model
                .get_asset_by_id(node_asset_id)
                .expect("node's own model asset must exist in the model");
            if model
                .lang_graph
                .associations_to(asset.lg_asset, node_asset.lg_asset)
                .is_empty()
            {
                return Err(AssocTraversalError::NoSelfAssociation {
                    asset_id,
                    node_asset_id,
                });
            }
            for (field_name, associated) in &asset.associated_assets {
                if associated.contains(&node_asset_id) {
                    additions.insert((
                        asset_id,
                        field_name.clone(),
                        AdditionTarget::ExistingAsset(node_asset_id),
                    ));
                }
            }
        }

        if let Some(filter_id) = traversal.asset_filter {
            additions.insert((
                asset_id,
                traversal.field_name.clone(),
                AdditionTarget::NewAssetType(filter_id),
            ));
        } else {
            let associations = model.lang_graph.associations(asset.lg_asset);
            let assoc = associations.get(&traversal.field_name).ok_or_else(|| {
                AssocTraversalError::UnknownAssociation {
                    asset_id,
                    field_name: traversal.field_name.clone(),
                }
            })?;
            let target_type = assoc.get_field(&traversal.field_name).asset;
            additions.insert((
                asset_id,
                traversal.field_name.clone(),
                AdditionTarget::NewAssetType(target_type),
            ));
        }
    }
    Ok((additions, traversal.quantity_filter))
}

/// Port of `parse_addition`.
pub fn parse_addition(
    model: &Model,
    node_asset_id: i64,
    instigating_assets: &HashSet<i64>,
    chain: &[AssocTraversalElem],
    rng: &mut impl Rng,
) -> Result<TerminalResolve<Addition>, AssocTraversalError> {
    resolve_terminal_traversal(model, instigating_assets, chain, rng, &|assets, t| {
        parse_addition_terminal(model, node_asset_id, assets, t)
    })
}

/// Port of `parse_removal`'s inner `_parse_terminating_expr`. Unlike
/// addition's terminal, removal never errors on a missing association -
/// Python's `.get(field_name, set())` defaults to empty, same as this
/// port's `.get(...).cloned().unwrap_or_default()`.
fn parse_removal_terminal(
    model: &Model,
    node_asset_id: i64,
    instigating_assets: &HashSet<i64>,
    traversal: &AssocTraversal,
) -> Result<TerminalResolve<Removal>, AssocTraversalError> {
    let mut removals = HashSet::new();
    for &asset_id in instigating_assets {
        let Some(asset) = model.get_asset_by_id(asset_id) else {
            continue;
        };

        if traversal.field_name == "self" {
            for (field_name, associated) in &asset.associated_assets {
                if associated.contains(&node_asset_id) {
                    removals.insert((asset_id, field_name.clone(), node_asset_id));
                }
            }
        }

        let mut candidates: HashSet<i64> = asset
            .associated_assets
            .get(&traversal.field_name)
            .cloned()
            .unwrap_or_default();
        if let Some(filter_id) = traversal.asset_filter {
            candidates.retain(|&id| {
                model
                    .get_asset_by_id(id)
                    .map(|a| a.lg_asset == filter_id)
                    .unwrap_or(false)
            });
        }
        for right_id in candidates {
            removals.insert((asset_id, traversal.field_name.clone(), right_id));
        }
    }
    Ok((removals, traversal.quantity_filter))
}

/// Port of `parse_removal`.
pub fn parse_removal(
    model: &Model,
    node_asset_id: i64,
    instigating_assets: &HashSet<i64>,
    chain: &[AssocTraversalElem],
    rng: &mut impl Rng,
) -> Result<TerminalResolve<Removal>, AssocTraversalError> {
    resolve_terminal_traversal(model, instigating_assets, chain, rng, &|assets, t| {
        parse_removal_terminal(model, node_asset_id, assets, t)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::wiper_attack_graph;
    use maltoolbox_attackgraph::{AttackGraph, AttackGraphNodeId};
    use maltoolbox_language::graph::model_effect::AssocTraversalElem;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // --- Direct port of `test_dyna_mal_simulator.py::test_assoc_traversal` ---

    #[test]
    fn wiper_infect_step_base_is_self() {
        let (graph, model) = wiper_attack_graph();
        let infect_id = graph.full_name_to_node["InfectedDevice:infect"];
        let infect = &graph.nodes[infect_id];
        let device_asset = infect.model_asset.expect("infect step has a model asset");
        let effect = &infect
            .additive_model_effects
            .as_ref()
            .expect("infect step has model effects")[0];
        let mut r = rng();

        let terminating = traverse_association_chain(
            &model,
            &HashSet::from([device_asset]),
            &effect.base,
            &mut r,
        )
        .unwrap();
        assert_eq!(
            terminating,
            HashSet::from([device_asset]),
            "infect step has self as base"
        );

        let resolved = traverse_association_chain(
            &model,
            &terminating,
            &effect.targets[0].assoc_traversal,
            &mut r,
        )
        .unwrap();
        assert!(
            resolved.is_empty(),
            "infect step should not have any targets yet"
        );
    }

    #[test]
    fn wiper_infect_step_target_resolves_after_adding_wiper() {
        let (graph, mut model) = wiper_attack_graph();
        let infect_id = graph.full_name_to_node["InfectedDevice:infect"];
        let infect = &graph.nodes[infect_id];
        let device_asset = infect.model_asset.unwrap();
        let effect = infect.additive_model_effects.as_ref().unwrap()[0].clone();
        let mut r = rng();

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

        let resolved = traverse_association_chain(
            &model,
            &HashSet::from([device_asset]),
            &effect.targets[0].assoc_traversal,
            &mut r,
        )
        .unwrap();
        assert_eq!(
            resolved,
            HashSet::from([wiper_id]),
            "infect step should have Wiper as target"
        );
    }

    /// Builds the same `Wiper:exfiltrate` fixture `test_assoc_traversal`
    /// does: the `Wiper:infect` model effect has already fired by hand
    /// (an explicitly-named `Wiper` asset, linked via `malware`), and the
    /// graph has been regenerated so `Wiper:exfiltrate` exists.
    fn wiper_attack_graph_with_wiper_asset() -> (AttackGraph, Model, AttackGraphNodeId) {
        let (mut graph, mut model) = wiper_attack_graph();
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
        graph.regenerate_graph(&model).unwrap();
        let exfiltrate_id = graph.full_name_to_node["Wiper:exfiltrate"];
        (graph, model, exfiltrate_id)
    }

    #[test]
    fn wiper_exfiltrate_step_base_is_infected_data() {
        let (graph, model, exfiltrate_id) = wiper_attack_graph_with_wiper_asset();
        let exfiltrate = &graph.nodes[exfiltrate_id];
        let wiper_asset = exfiltrate
            .model_asset
            .expect("exfiltrate step has a model asset");
        let effect = &exfiltrate.additive_model_effects.as_ref().unwrap()[0];
        let mut r = rng();

        let terminating =
            traverse_association_chain(&model, &HashSet::from([wiper_asset]), &effect.base, &mut r)
                .unwrap();
        let infected_data = model.get_asset_by_name("InfectedData").unwrap().id;
        assert_eq!(
            terminating,
            HashSet::from([infected_data]),
            "exfiltrate step has InfectedData as base"
        );

        // InfectedData isn't associated to C2Server yet, so the target
        // doesn't resolve to anything.
        let resolved = traverse_association_chain(
            &model,
            &terminating,
            &effect.targets[0].assoc_traversal,
            &mut r,
        )
        .unwrap();
        assert!(
            resolved.is_empty(),
            "exfiltrate target shouldn't resolve before the C2Server link exists"
        );
    }

    #[test]
    fn wiper_exfiltrate_step_target_resolves_after_c2_link_from_wiper_asset() {
        let (graph, mut model, exfiltrate_id) = wiper_attack_graph_with_wiper_asset();
        let wiper_asset = graph.nodes[exfiltrate_id].model_asset.unwrap();
        let effect = graph.nodes[exfiltrate_id]
            .additive_model_effects
            .as_ref()
            .unwrap()[0]
            .clone();
        let mut r = rng();

        let infected_data = model.get_asset_by_name("InfectedData").unwrap().id;
        let c2_server = model.get_asset_by_name("C2Server").unwrap().id;
        model
            .add_associated_assets(infected_data, "node", HashSet::from([c2_server]))
            .unwrap();

        // This target is an additive assoc op, so the instigating asset is
        // the asset where the step is defined (the Wiper), not the base.
        let resolved = traverse_association_chain(
            &model,
            &HashSet::from([wiper_asset]),
            &effect.targets[0].assoc_traversal,
            &mut r,
        )
        .unwrap();
        assert_eq!(resolved, HashSet::from([infected_data]));
    }

    // --- sample_size / apply_quantity_filter ---

    #[test]
    fn sample_size_none_quantity_is_always_one() {
        let mut r = rng();
        assert_eq!(sample_size(None, &mut r, Some(0)), 1);
        assert_eq!(sample_size(None, &mut r, None), 1);
    }

    #[test]
    fn sample_size_exact_clamped_to_max_size() {
        let mut r = rng();
        assert_eq!(
            sample_size(Some(QuantityFilter::Exact(5)), &mut r, Some(2)),
            2
        );
        assert_eq!(
            sample_size(Some(QuantityFilter::Exact(1)), &mut r, Some(2)),
            1
        );
    }

    #[test]
    fn sample_size_range_is_within_clamped_bounds() {
        let mut r = rng();
        for _ in 0..50 {
            let size = sample_size(Some(QuantityFilter::Range(1, 10)), &mut r, Some(3));
            assert!((1..=3).contains(&size), "size {size} out of clamped range");
        }
    }

    #[test]
    fn apply_quantity_filter_exact_picks_requested_count() {
        let objects: HashSet<i64> = (0..10).collect();
        let mut r = rng();
        let chosen = apply_quantity_filter(&objects, Some(QuantityFilter::Exact(3)), &mut r);
        assert_eq!(chosen.len(), 3);
        assert!(chosen.is_subset(&objects));
    }

    #[test]
    fn apply_quantity_filter_clamps_to_available_size() {
        let objects: HashSet<i64> = (0..2).collect();
        let mut r = rng();
        let chosen = apply_quantity_filter(&objects, Some(QuantityFilter::Exact(10)), &mut r);
        assert_eq!(chosen, objects);
    }

    // --- glob (transitive) and set-operation traversal, hand-built ---

    #[test]
    fn glob_traversal_reaches_transitive_neighbors() {
        // Device.receiveFrom/sendTo is symmetric, so a glob over "sendTo"
        // starting from one device should reach every device transitively
        // reachable via sendTo links, not just its direct neighbor.
        let (_graph, mut model) = wiper_attack_graph();
        let a = model.get_asset_by_name("InfectedDevice").unwrap().id;
        let b = model.get_asset_by_name("VulnerableDevice").unwrap().id;
        // wiper_model.yml already links InfectedDevice <-> VulnerableDevice
        // via sendTo/receiveFrom; add a third device only reachable through b.
        let c = model
            .add_asset(
                "Device",
                Some("ThirdDevice".to_string()),
                None,
                None,
                None,
                false,
            )
            .unwrap();
        model
            .add_associated_assets(b, "sendTo", HashSet::from([c]))
            .unwrap();

        let glob = AssocTraversalElem::Glob(GlobAssocTraversal {
            pattern: wiper_sendto_chain(),
            quantity_filter: None,
        });
        let mut r = rng();
        let result =
            traverse_association_chain(&model, &HashSet::from([a]), &[glob], &mut r).unwrap();
        assert!(result.contains(&b));
        assert!(
            result.contains(&c),
            "glob should reach c transitively through b"
        );
    }

    fn wiper_sendto_chain() -> Vec<AssocTraversalElem> {
        vec![AssocTraversalElem::Traversal(AssocTraversal {
            field_name: "sendTo".to_string(),
            asset_filter: None,
            quantity_filter: None,
        })]
    }

    #[test]
    fn set_union_combines_both_sides() {
        let (_graph, model) = wiper_attack_graph();
        let a = model.get_asset_by_name("InfectedDevice").unwrap().id;
        let b = model.get_asset_by_name("C2Server").unwrap().id;

        // left: InfectedDevice.sendTo, right: InfectedDevice.receiveFrom -
        // wiper_model.yml makes these disjoint (sendTo={C2Server,
        // VulnerableDevice}, receiveFrom={VulnerableDevice}), so union
        // should just be their combined set.
        let set = AssocSet {
            set_op: SetOperation::Union,
            left: wiper_sendto_chain(),
            right: vec![AssocTraversalElem::Traversal(AssocTraversal {
                field_name: "receiveFrom".to_string(),
                asset_filter: None,
                quantity_filter: None,
            })],
            quantity_filter: None,
        };
        let mut r = rng();
        let result = traverse_association_chain(
            &model,
            &HashSet::from([a]),
            &[AssocTraversalElem::Set(set)],
            &mut r,
        )
        .unwrap();
        assert!(result.contains(&b));
    }

    #[test]
    fn empty_chain_is_an_error_at_the_terminal_resolver() {
        let (graph, model) = wiper_attack_graph();
        let infect_id = graph.full_name_to_node["InfectedDevice:infect"];
        let device_asset = graph.nodes[infect_id].model_asset.unwrap();
        let mut r = rng();

        let result = parse_addition(
            &model,
            device_asset,
            &HashSet::from([device_asset]),
            &[],
            &mut r,
        );
        assert!(matches!(result, Err(AssocTraversalError::EmptyChain)));
    }
}
