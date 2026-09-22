"""Restoring the instance model (and the attack graph derived from it) to a
previous state.
"""

from typing import Any

from maltoolbox.attackgraph import AttackGraph
from maltoolbox.model import Model, ModelAsset


def reconcile_model_to_snapshot(
    model: Model, snapshot: dict[str, Any]
) -> tuple[
    set[ModelAsset],
    set[ModelAsset],
    set[tuple[ModelAsset, str, ModelAsset]],
    set[tuple[ModelAsset, str, ModelAsset]],
]:
    """Mutate `model` in place to match `snapshot` exactly and return the
    (new_assets, removed_assets, new_associations, removed_associations)
    that were applied, suitable as-is for
    AttackGraph.partially_regenerate_graph().
    """
    new_assets: set[ModelAsset] = set()
    removed_assets: set[ModelAsset] = set()

    current_ids = set(model.assets.keys())
    target_ids = set(snapshot['assets'].keys())

    for asset_id in current_ids - target_ids:
        asset = model.assets[asset_id]
        removed_assets.add(asset)
        model.remove_asset(asset)

    for asset_id in target_ids - current_ids:
        asset_dict = snapshot['assets'][asset_id]
        new_asset = model.add_asset(
            asset_type=asset_dict['type'],
            name=asset_dict['name'],
            asset_id=asset_id,
        )
        new_assets.add(new_asset)

    new_associations: set[tuple[ModelAsset, str, ModelAsset]] = set()
    removed_associations: set[tuple[ModelAsset, str, ModelAsset]] = set()

    for asset_id, asset_dict in snapshot['assets'].items():
        asset = model.assets[asset_id]
        current_associations = {
            field_name: frozenset(other.id for other in others)
            for field_name, others in asset.associated_assets.items()
        }
        target_associations = {
            field_name: frozenset(other_ids)
            for field_name, other_ids in asset_dict['associated_assets'].items()
        }
        for field_name in set(current_associations) | set(target_associations):
            current_ids_for_field = current_associations.get(field_name, frozenset())
            target_ids_for_field = target_associations.get(field_name, frozenset())

            ids_to_remove = current_ids_for_field - target_ids_for_field
            if ids_to_remove:
                to_remove = {model.assets[i] for i in ids_to_remove}
                asset.remove_associated_assets(field_name, to_remove)
                removed_associations.update(
                    (asset, field_name, other) for other in to_remove
                )

            ids_to_add = target_ids_for_field - current_ids_for_field
            if ids_to_add:
                to_add = {model.assets[i] for i in ids_to_add}
                asset.add_associated_assets(field_name, to_add)
                new_associations.update((asset, field_name, other) for other in to_add)

    return new_assets, removed_assets, new_associations, removed_associations


def reset_model_effects(
    attack_graph: AttackGraph, model_snapshot: dict[str, Any]
) -> None:
    assert attack_graph.model, 'AttackGraph needs to have the model object available.'
    new_assets, removed_assets, new_associations, removed_associations = (
        reconcile_model_to_snapshot(attack_graph.model, model_snapshot)
    )
    attack_graph.partially_regenerate_graph(
        new_assets=new_assets,
        removed_assets=removed_assets,
        new_associations=new_associations,
        removed_associations=removed_associations,
    )
