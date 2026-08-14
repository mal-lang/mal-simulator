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

RightAsset = TypeVar('RightAsset', bound=object)


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
            logger.warning(
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
            candidate_assets = set(
                rng.choice(
                    list(candidate_assets),
                    size=assoc_traversal.quantity_filter,
                    replace=False,
                )
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
        [set[ModelAsset], AssocTraversal], set[tuple[ModelAsset, str, RightAsset]]
    ],
) -> set[tuple[ModelAsset, str, RightAsset]]:
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
        left = _resolve_terminal_traversal(current_assets, last.left, rng, terminate)
        right = _resolve_terminal_traversal(current_assets, last.right, rng, terminate)
        if last.set_op == SetOperation.UNION:
            return left | right
        elif last.set_op == SetOperation.DIFFERENCE:
            return left - right
        elif last.set_op == SetOperation.INTERSECTION:
            return left & right
        else:
            raise ValueError(
                f'Unknown set operation {last.set_op} in association set traversal.'
            )
    else:
        raise ValueError(f'Unknown association traversal type: {type(last)}')


def parse_addition(
    node: AttackGraphNode,
    instigating_assets: set[ModelAsset],
    assoc_traversals: AssocTraversalChain,
    rng: np.random.Generator,
) -> set[tuple[ModelAsset, str, LanguageGraphAsset | ModelAsset]]:

    def _parse_terminating_expr(
        node: AttackGraphNode,
        instigating_assets: set[ModelAsset],
        assoc_traversal: AssocTraversal,
    ) -> set[tuple[ModelAsset, str, LanguageGraphAsset | ModelAsset]]:
        additions: set[tuple[ModelAsset, str, LanguageGraphAsset | ModelAsset]] = set()
        for asset in instigating_assets:
            if assoc_traversal.field_name == 'self':
                assert node.model_asset, (
                    f'Node {node.name} does not have a model asset, '
                    'so cannot traverse to self.'
                )

                if len(asset.lg_asset.associations_to(node.model_asset.lg_asset)) == 0:
                    raise ValueError(
                        f"{asset.name} does not have a valid association"
                        f" to the node's model asset {node.model_asset.name},"
                        " so cannot traverse to self."
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
            # TODO: Do something with the quantity filter?
        return additions

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
) -> set[tuple[ModelAsset, str, ModelAsset]]:

    def _parse_terminating_expr(
        node: AttackGraphNode,
        instigating_assets: set[ModelAsset],
        assoc_traversal: AssocTraversal,
    ) -> set[tuple[ModelAsset, str, ModelAsset]]:
        removals: set[tuple[ModelAsset, str, ModelAsset]] = set()
        for asset in instigating_assets:
            if assoc_traversal.field_name == 'self':
                assert node.model_asset, (
                    f'Node {node.name} does not have a model asset, '
                    'so cannot traverse to self.'
                )
                for (
                    field_name,
                    associated_assets,
                ) in node.model_asset.associated_assets.items():
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
            # TODO: Do something with the quantity filter?
            for removal_candidate in removal_candidates:
                removals.add((asset, assoc_traversal.field_name, removal_candidate))
        return removals

    return _resolve_terminal_traversal(
        instigating_assets,
        assoc_traversals,
        rng,
        lambda assets, assoc: _parse_terminating_expr(node, assets, assoc),
    )
