"""Functions for processing association traversal of model effects
in the DynaMalSimulator.
"""

import logging
from collections.abc import Callable
from typing import TypeVar

from maltoolbox.language.language_graph_model_effect import (
    AssocTraversal,
    GlobAssocTraversal,
    AssocSet,
)
from maltoolbox.language import LanguageGraphAsset
from maltoolbox.model import ModelAsset
import numpy as np
from maltoolbox.attackgraph import AttackGraphNode
from maltoolbox.language.language_graph_model_effect import AssocTraversalChain
from maltoolbox.language.language_graph_model_effect import (
    SetOperation,
)

logger = logging.getLogger(__name__)

AssetTypeOrInstance = TypeVar('AssetTypeOrInstance', bound=object)

Quantity = int | tuple[int, int] | None

T = TypeVar('T', bound=object)


def sample_size(
    quantity: Quantity, rng: np.random.Generator, max_size: int | float = float('inf')
) -> int:
    if quantity is None:
        size = 1
    elif isinstance(quantity, int):
        size = int(min(quantity, max_size))
    else:
        high = int(min(quantity[1], max_size))
        low = int(min(quantity[0], high))
        size = int(rng.integers(low, high, endpoint=True))
    return size


def _apply_quantity_filter(
    objects: set[T],
    quantity: Quantity,
    rng: np.random.Generator,
) -> set[T]:
    size = sample_size(quantity, rng, len(objects))
    sorted_objects = list(objects)
    indicies = np.arange(len(sorted_objects))
    chosen_indicies: list[int] = rng.choice(indicies, size=size, replace=False).tolist()
    return {sorted_objects[i] for i in chosen_indicies}


def _assoc_traversal(
    instigating_assets: set[ModelAsset],
    assoc_traversal: AssocTraversal,
    rng: np.random.Generator,
) -> set[ModelAsset]:
    if assoc_traversal.field_name == 'self':
        return instigating_assets
    next_assets = set()
    for asset in instigating_assets:
        if assoc_traversal.field_name not in asset.associated_assets:
            logger.error(
                f"Asset {asset.name} doesn't have any "
                f'`{assoc_traversal.field_name}` associated. '
                'Skipping this asset in the association traversal.'
            )
            continue
        candidate_assets = asset.associated_assets[assoc_traversal.field_name]
        if assoc_traversal.asset_filter:
            candidate_assets = {
                asset
                for asset in candidate_assets
                if asset.lg_asset.name == assoc_traversal.asset_filter.name
            }
        if assoc_traversal.quantity_filter:
            candidate_assets = _apply_quantity_filter(
                candidate_assets, assoc_traversal.quantity_filter, rng
            )
        next_assets.update(candidate_assets)
    return next_assets


def _glob_assoc_traversal(
    instigating_assets: set[ModelAsset],
    glob_assoc_traversal: GlobAssocTraversal,
    rng: np.random.Generator,
) -> set[ModelAsset]:
    next_assets = traverse_association_chain(
        instigating_assets, glob_assoc_traversal.pattern, rng
    )
    while True:
        new_assets = traverse_association_chain(
            instigating_assets, glob_assoc_traversal.pattern, rng
        )
        if len(new_assets.difference(next_assets)) == 0:
            break
        next_assets = new_assets
    if glob_assoc_traversal.quantity_filter:
        next_assets = _apply_quantity_filter(
            next_assets, glob_assoc_traversal.quantity_filter, rng
        )
    return next_assets


def _assoc_set_traversal(
    starting_assets: set[ModelAsset], assoc_set: AssocSet, rng: np.random.Generator
) -> set[ModelAsset]:
    candidate_assets = set()
    left = traverse_association_chain(starting_assets, assoc_set.left, rng)
    right = traverse_association_chain(starting_assets, assoc_set.right, rng)
    if assoc_set.set_op == SetOperation.UNION:
        candidate_assets = left | right
    elif assoc_set.set_op == SetOperation.DIFFERENCE:
        candidate_assets = left - right
    elif assoc_set.set_op == SetOperation.INTERSECTION:
        candidate_assets = left & right
    else:
        raise ValueError(
            f'Unknown set operation {assoc_set.set_op} in association set traversal.'
        )
    if assoc_set.quantity_filter:
        candidate_assets = _apply_quantity_filter(
            candidate_assets, assoc_set.quantity_filter, rng
        )
    return candidate_assets


def traverse_association_chain(
    instigating_assets: set[ModelAsset],
    assoc_traversals: AssocTraversalChain,
    rng: np.random.Generator,
) -> set[ModelAsset]:
    """Traverse the association chain starting from the given asset."""

    current_assets = instigating_assets
    for assoc in assoc_traversals:
        if isinstance(assoc, AssocTraversal):
            current_assets = _assoc_traversal(current_assets, assoc, rng)
        elif isinstance(assoc, GlobAssocTraversal):
            current_assets = _glob_assoc_traversal(current_assets, assoc, rng)
        elif isinstance(assoc, AssocSet):
            current_assets = _assoc_set_traversal(current_assets, assoc, rng)
        else:
            raise ValueError(f'Unknown association traversal type: {type(assoc)}')
    return current_assets


def _resolve_terminal_traversal(
    instigating_assets: set[ModelAsset],
    assoc_traversals: AssocTraversalChain,
    rng: np.random.Generator,
    terminate: Callable[
        [set[ModelAsset], AssocTraversal],
        tuple[set[tuple[ModelAsset, str, AssetTypeOrInstance]], Quantity],
    ],
) -> tuple[set[tuple[ModelAsset, str, AssetTypeOrInstance]], Quantity]:
    """Traverse `assoc_traversals` like `traverse_association_chain`, except
    the very last `AssocTraversal` reached (however deep inside nested
    `GlobAssocTraversal`/`AssocSet` structures) is resolved via `terminate`
    instead of being traversed to an asset set.

    Used by `parse_addition` and `parse_removal`, which only differ in what
    `terminate` does with the final `AssocTraversal`.
    """
    if not assoc_traversals:
        raise ValueError(
            'Association traversal chain did not terminate in an AssocTraversal.'
        )
    current_assets = traverse_association_chain(
        instigating_assets, assoc_traversals[:-1], rng
    )
    last = assoc_traversals[-1]
    if isinstance(last, AssocTraversal):
        return terminate(current_assets, last)
    elif isinstance(last, GlobAssocTraversal):
        # `*` is a repeated application of the pattern, so the termination
        # is whatever the pattern itself terminates in.
        return _resolve_terminal_traversal(current_assets, last.pattern, rng, terminate)
    elif isinstance(last, AssocSet):
        left, left_quantity = _resolve_terminal_traversal(
            current_assets, last.left, rng, terminate
        )
        right, right_quantity = _resolve_terminal_traversal(
            current_assets, last.right, rng, terminate
        )
        if left_quantity is not None or right_quantity is not None:
            raise NotImplementedError(
                'Quantity filtering is not supported set operations in terminal fields.'
            )
        terminal: set[tuple[ModelAsset, str, AssetTypeOrInstance]] = set()
        if last.set_op == SetOperation.UNION:
            terminal = left | right
        elif last.set_op == SetOperation.DIFFERENCE:
            terminal = left - right
        elif last.set_op == SetOperation.INTERSECTION:
            terminal = left & right
        else:
            raise ValueError(
                f'Unknown set operation {last.set_op} in association set traversal.'
            )
        return terminal, None
    else:
        raise ValueError(f'Unknown association traversal type: {type(last)}')


def parse_addition(
    node: AttackGraphNode,
    instigating_assets: set[ModelAsset],
    assoc_traversals: AssocTraversalChain,
    rng: np.random.Generator,
) -> tuple[set[tuple[ModelAsset, str, LanguageGraphAsset | ModelAsset]], Quantity]:

    def _parse_terminating_expr(
        node: AttackGraphNode,
        instigating_assets: set[ModelAsset],
        assoc_traversal: AssocTraversal,
    ) -> tuple[set[tuple[ModelAsset, str, LanguageGraphAsset | ModelAsset]], Quantity]:
        additions: set[tuple[ModelAsset, str, LanguageGraphAsset | ModelAsset]] = set()
        for asset in instigating_assets:
            if assoc_traversal.field_name == 'self':
                assert node.model_asset, (
                    f'Node {node.name} does not have a model asset, '
                    'so cannot traverse to self.'
                )

                if len(asset.lg_asset.associations_to(node.model_asset.lg_asset)) == 0:
                    raise ValueError(
                        f'{asset.name} does not have a valid association'
                        f" to the node's model asset {node.model_asset.name},"
                        ' so cannot traverse to self.'
                    )
                for field_name, associated_assets in asset.associated_assets.items():
                    if node.model_asset in associated_assets:
                        additions.add((asset, field_name, node.model_asset))

            if assoc_traversal.asset_filter:
                additions.add(
                    (asset, assoc_traversal.field_name, assoc_traversal.asset_filter)
                )
            else:
                assoc = asset.lg_asset.associations[assoc_traversal.field_name]
                additions.add(
                    (
                        asset,
                        assoc_traversal.field_name,
                        assoc.get_field(assoc_traversal.field_name).asset,
                    )
                )

        if assoc_traversal.quantity_filter is not None:
            pass
            # additions = _apply_quantity_filter(
            #     additions, assoc_traversal.quantity_filter, rng
            # )

        return additions, assoc_traversal.quantity_filter

    return _resolve_terminal_traversal(
        instigating_assets,
        assoc_traversals,
        rng,
        lambda assets, assoc: _parse_terminating_expr(node, assets, assoc),
    )


def parse_removal(
    node: AttackGraphNode,
    instigating_assets: set[ModelAsset],
    assoc_traversals: AssocTraversalChain,
    rng: np.random.Generator,
) -> tuple[set[tuple[ModelAsset, str, ModelAsset]], Quantity]:

    def _parse_terminating_expr(
        node: AttackGraphNode,
        instigating_assets: set[ModelAsset],
        assoc_traversal: AssocTraversal,
    ) -> tuple[set[tuple[ModelAsset, str, ModelAsset]], Quantity]:
        removals: set[tuple[ModelAsset, str, ModelAsset]] = set()
        for asset in instigating_assets:
            if assoc_traversal.field_name == 'self':
                assert node.model_asset, (
                    f'Node {node.name} does not have a model asset, '
                    'so cannot traverse to self.'
                )
                for field_name, associated_assets in asset.associated_assets.items():
                    if node.model_asset in associated_assets:
                        removals.add((asset, field_name, node.model_asset))

            removal_candidates = asset.associated_assets.get(
                assoc_traversal.field_name, set()
            )
            if assoc_traversal.asset_filter:
                removal_candidates = {
                    asset
                    for asset in removal_candidates
                    if asset.lg_asset.name == assoc_traversal.asset_filter.name
                }
            for removal_candidate in removal_candidates:
                removals.add((asset, assoc_traversal.field_name, removal_candidate))
        if assoc_traversal.quantity_filter is not None:
            pass
            # removals = _apply_quantity_filter(
            #     removals, assoc_traversal.quantity_filter, rng
            # )
        return removals, assoc_traversal.quantity_filter

    return _resolve_terminal_traversal(
        instigating_assets,
        assoc_traversals,
        rng,
        lambda assets, assoc: _parse_terminating_expr(node, assets, assoc),
    )
