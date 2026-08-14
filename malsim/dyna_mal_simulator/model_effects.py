"""Functions relating to modifications of the instance model based on the
model effects of attack steps.
"""

import logging
from collections.abc import Callable

import numpy as np
from maltoolbox.attackgraph import AttackGraphNode
from maltoolbox.model import ModelAsset
from maltoolbox.model import Model
from malsim.dyna_mal_simulator.graph_state import update_graph_state
from malsim.dyna_mal_simulator.process_assoc_traversal import (
    parse_addition,
    parse_removal,
    traverse_association_chain,
)
from malsim.dyna_mal_simulator.simulator_state import (
    DynaMalSimulatorState,
    AssetOp,
    AssocOp,
)
from maltoolbox.language.language_graph_model_effect import (
    LanguageGraphModelEffect,
    ModelEffectType,
    DynTarget,
)

logger = logging.getLogger(__name__)


def target_op(
    node: AttackGraphNode, model_effect: LanguageGraphModelEffect, dyn_target: DynTarget
) -> Callable[
    [set[ModelAsset], DynTarget, Model, np.random.Generator], list[AssetOp | AssocOp]
]:

    def add_asset(
        base: set[ModelAsset], target: DynTarget, model: Model, rng: np.random.Generator
    ) -> list[AssetOp | AssocOp]:
        assert node.model_asset, 'AttackGraph needs to have the model object available.'
        modification_record: list[AssetOp | AssocOp] = []
        addition_info = parse_addition(
            node=node,
            instigating_assets=base,
            assoc_traversals=target.assoc_traversal,
            rng=rng,
        )
        for left_asset, field_name, right_asset in addition_info:
            if isinstance(right_asset, ModelAsset):
                if not (right_asset == node.model_asset):
                    raise ValueError(
                        'Cannot add an existing asset'
                        f' ({right_asset.name}) to the model.'
                        ' It already exists in the model!'
                    )
                else:
                    raise ValueError(
                        f'Cannot add `self` ({right_asset.name}) to the model.'
                        ' It already exists in the model!'
                    )
            else:
                new_asset = model.add_asset(
                    asset_type=right_asset.name, name=right_asset.name
                )
                added_asset_op = AssetOp(type=ModelEffectType.ADDITIVE, asset=new_asset)
                try:
                    left_asset.add_associated_assets(field_name, {new_asset})
                    added_assoc_op = AssocOp(
                        type=ModelEffectType.ADDITIVE,
                        assoc=(left_asset, field_name, new_asset),
                    )
                    modification_record.extend([added_asset_op, added_assoc_op])
                except ValueError as exception:
                    logger.error(
                        f'Failed to add a {field_name} asset to '
                        f'{left_asset.name}: {exception}\n'
                        f'Skipping addition of {field_name}.'
                    )
                    model.remove_asset(new_asset)
                    continue
        return modification_record

    def remove_asset(
        base: set[ModelAsset], target: DynTarget, model: Model, rng: np.random.Generator
    ) -> list[AssetOp | AssocOp]:
        modification_record: list[AssetOp | AssocOp] = []
        removal_info = parse_removal(
            node=node,
            instigating_assets=base,
            assoc_traversals=target.assoc_traversal,
            rng=rng,
        )

        removal_assets: set[ModelAsset] = set()
        for left_asset, field_name, right_asset in removal_info:
            try:
                left_asset.remove_associated_assets(field_name, {right_asset})
                modification_record.append(
                    AssocOp(
                        type=ModelEffectType.SUBTRACTIVE,
                        assoc=(left_asset, field_name, right_asset),
                    )
                )
                removal_assets.add(right_asset)
            except ValueError as exception:
                logger.error(
                    f'Failed to remove a {field_name} asset from '
                    f'{left_asset.name}: {exception}\n'
                    f'Skipping removal of {field_name}.'
                )
                continue
        for removal_asset in removal_assets:
            model.remove_asset(removal_asset)
            modification_record.append(
                AssetOp(type=ModelEffectType.SUBTRACTIVE, asset=removal_asset)
            )
        return modification_record

    def add_assoc(
        base: set[ModelAsset], target: DynTarget, model: Model, rng: np.random.Generator
    ) -> list[AssetOp | AssocOp]:
        assert node.model_asset, 'AttackGraph needs to have the model object available.'
        modification_record: list[AssetOp | AssocOp] = []
        addition_info = parse_addition(
            node=node,
            instigating_assets={node.model_asset},
            assoc_traversals=target.assoc_traversal,
            rng=rng,
        )
        for left_asset, field_name, right_asset in addition_info:
            if isinstance(right_asset, ModelAsset):
                # Check if the association already exists
                if right_asset in left_asset.associated_assets.get(field_name, ()):
                    # Skip recording the addition otherwise modification_record
                    # will get out of sync with the model state and reset() will fail
                    continue
                left_asset.add_associated_assets(field_name, {right_asset})
                modification_record.append(
                    AssocOp(
                        type=ModelEffectType.ADDITIVE,
                        assoc=(left_asset, field_name, right_asset),
                    )
                )
            else:
                for assoc_candidate in base:
                    if assoc_candidate in left_asset.associated_assets.get(
                        field_name, ()
                    ):
                        # Already associated, see comment above.
                        continue
                    try:
                        left_asset.add_associated_assets(field_name, {assoc_candidate})
                        modification_record.append(
                            AssocOp(
                                type=ModelEffectType.ADDITIVE,
                                assoc=(
                                    left_asset,
                                    field_name,
                                    assoc_candidate,
                                ),
                            )
                        )
                    except ValueError as exception:
                        logger.error(
                            f'Failed to associate the {assoc_candidate.name} asset to '
                            f'the {left_asset.name} asset: {exception} Skipping '
                            'addition of association.'
                        )
                        continue
        return modification_record

    def remove_assoc(
        base: set[ModelAsset], target: DynTarget, model: Model, rng: np.random.Generator
    ) -> list[AssetOp | AssocOp]:
        modification_record: list[AssetOp | AssocOp] = []
        removal_info = parse_removal(
            node=node,
            instigating_assets=base,
            assoc_traversals=target.assoc_traversal,
            rng=rng,
        )
        for left_asset, field_name, right_asset in removal_info:
            if field_name in left_asset.associated_assets:
                left_asset.remove_associated_assets(field_name, {right_asset})
                modification_record.append(
                    AssocOp(
                        type=ModelEffectType.SUBTRACTIVE,
                        assoc=(
                            left_asset,
                            field_name,
                            right_asset,
                        ),
                    )
                )
            else:
                logger.error(
                    f'Asset {left_asset.name} is not associated to any '
                    f"'{field_name}'. Skipping removal of "
                    f'association to {field_name}.'
                )
                continue
        return modification_record

    if (
        model_effect.model_effect_type == ModelEffectType.ADDITIVE
        and not dyn_target.assoc_op
    ):
        return add_asset
    elif (
        model_effect.model_effect_type == ModelEffectType.SUBTRACTIVE
        and not dyn_target.assoc_op
    ):
        return remove_asset
    elif (
        model_effect.model_effect_type == ModelEffectType.ADDITIVE
        and dyn_target.assoc_op
    ):
        return add_assoc
    elif (
        model_effect.model_effect_type == ModelEffectType.SUBTRACTIVE
        and dyn_target.assoc_op
    ):
        return remove_assoc
    else:
        raise ValueError(
            f'Invalid combination of model effect type '
            f'{model_effect.model_effect_type} and target association operation '
            f'{dyn_target.assoc_op}.'
        )


def _apply_model_effect(
    node: AttackGraphNode,
    model_effect: LanguageGraphModelEffect,
    model: Model,
    rng: np.random.Generator,
) -> list[AssetOp | AssocOp]:
    """Applies a modification to the instance model based on the provided model
    effect.
    """
    assert node.model_asset, 'AttackGraph needs to have the model object available.'
    modification_record = []
    base_set = traverse_association_chain({node.model_asset}, model_effect.base, rng)
    for dyn_target in model_effect.targets:
        # is_edge_addition = (
        #     dyn_target.assoc_op
        #     and model_effect.model_effect_type == ModelEffectType.ADDITIVE
        # )
        # instigating_assets = {node.model_asset} if is_edge_addition else base_set

        if len(dyn_target.assoc_traversal) == 0:
            raise ValueError(
                f'Target association traversal cannot be empty for step '
                f'{node.full_name}.'
            )

        target_op_fn = target_op(node, model_effect, dyn_target)
        op_result = target_op_fn(base_set, dyn_target, model, rng)
        modification_record.extend(op_result)
    return modification_record


def execute_model_effects(
    sim_state: DynaMalSimulatorState, action: AttackGraphNode, rng: np.random.Generator
) -> DynaMalSimulatorState:
    """Execute the model effects of an attack step and return the effects that
    were applied.

    Args:
        sim_state: The current simulator state.
        agent: The attacker agent performing the action.
        action: The attack step node being executed.

    Returns:
        A list of the effects that were applied as a result of the action.
    """
    assert sim_state.attack_graph.model, (
        'AttackGraph needs to have the model object available.'
    )
    modification_record = []
    for model_effect in action.additive_model_effects or []:
        modification_record.extend(
            _apply_model_effect(action, model_effect, sim_state.attack_graph.model, rng)
        )
    for model_effect in action.subtractive_model_effects or []:
        modification_record.extend(
            _apply_model_effect(action, model_effect, sim_state.attack_graph.model, rng)
        )
    logger.debug('Partially regenerating attack graph after applying model effects.')
    new_nodes = sim_state.attack_graph.partially_regenerate_graph(
        new_assets={
            op.asset
            for op in modification_record
            if isinstance(op, AssetOp) and op.type == ModelEffectType.ADDITIVE
        },
        removed_assets={
            op.asset
            for op in modification_record
            if isinstance(op, AssetOp) and op.type == ModelEffectType.SUBTRACTIVE
        },
        new_associations={
            op.assoc
            for op in modification_record
            if isinstance(op, AssocOp) and op.type == ModelEffectType.ADDITIVE
        },
        removed_associations={
            op.assoc
            for op in modification_record
            if isinstance(op, AssocOp) and op.type == ModelEffectType.SUBTRACTIVE
        },
    )
    new_graph_state, new_enabled_defenses = update_graph_state(
        sim_state.graph_state,
        sim_state.settings,
        sim_state.attack_graph,
        new_nodes,
        rng,
    )
    return DynaMalSimulatorState(
        attack_graph=sim_state.attack_graph,
        settings=sim_state.settings,
        graph_state=new_graph_state,
        enabled_defenses=sim_state.enabled_defenses | new_enabled_defenses,
        modification_record=sim_state.modification_record + modification_record,
    )
