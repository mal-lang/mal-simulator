from maltoolbox.language import LanguageGraphAssociation
import numpy as np

from .mal_spaces import (
    Asset,
    Step,
    Association,
    LogicGate,
    MALObsInstance,
)
from .serialization import LangSerializer
from malsim.mal_simulator import MalSimAttackerState, MalSimDefenderState

def attacker_state2graph(
    state: MalSimAttackerState, lang_serializer: LangSerializer, use_logic_gates: bool, see_defense_steps: bool = False
) -> MALObsInstance:
    """Create a MALObsInstance of an attackers observation"""

    # Get visible assets from performed and action surface nodes
    visible_assets_set = (
        {node.model_asset for node in state.performed_nodes if node.model_asset} 
        | {node.model_asset for node in state.action_surface if node.model_asset}
    )
    visible_assets = sorted(visible_assets_set, key=lambda asset: asset.id)
    assets = Asset(
        type=np.array([
            lang_serializer.asset_type[asset.type] for asset in visible_assets
        ]),
        id=np.array([asset.id for asset in visible_assets]),
    )

    # The attacker is assumed to know the language
    # so it can see all attack steps on the visible assets

    # NOTE: This sorting is assumed to be used for the actions in the action space
    visible_steps = sorted({
        node for node in state.sim.attack_graph.nodes.values()
        if node.model_asset in visible_assets and node.type in ("and", "or")
    }, key=lambda step: step.id)
    if see_defense_steps:
        visible_steps += sorted({
            node for node in state.sim.attack_graph.nodes.values()
            if node.model_asset in visible_assets and node.type in ('defense', 'exist', 'notExist')
        }, key=lambda step: step.id)

    if lang_serializer.split_attack_step_types:
        attack_step_type = np.array([
            lang_serializer.attack_step_type[(node.model_asset.type, node.name)]
            for node in visible_steps if node.model_asset
        ])
    else:
        attack_step_type = np.array([
            lang_serializer.attack_step_type[(node.name,)] for node in visible_steps
        ])

    attack_steps = Step(
        type=attack_step_type,
        id=np.array([node.id for node in visible_steps]),
        logic_class=np.array([
            lang_serializer.attack_step_class[node.type] for node in visible_steps
        ]),
        tags=np.array([
            lang_serializer.attack_step_tag[node.tags[0] if len(node.tags) > 0 else None]
            for node in visible_steps
        ]),
        compromised=np.array([
            state.sim.node_is_compromised(node) for node in visible_steps
        ]),
        attempts=np.array([
            state.num_attempts.get(node, 0) for node in visible_steps
        ]),
        action_mask=np.array([
            state.sim.node_is_traversable(state.performed_nodes, node)
            for node in visible_steps
        ]),
    )
    step2asset = {
        (visible_steps.index(node), visible_assets.index(node.model_asset))
        for node in visible_steps if node.model_asset
    }
    step2step = {
        (visible_steps.index(node), visible_steps.index(child))
        for node in visible_steps for child in node.children if child in visible_steps
    }

    if len(visible_assets) > 1:
        associations: list[tuple[LanguageGraphAssociation, int, int]] = []
        assoc2asset = set()
        for asset in visible_assets:
            for fieldname, other_assets in asset.associated_assets.items():
                assoc = asset.lg_asset.associations[fieldname]
                for other_asset in filter(lambda asset: asset in visible_assets, other_assets):
                    asset1, asset2 = sorted([asset, other_asset], key=lambda asset: asset.id)
                    if (assoc, asset1.id, asset2.id) not in associations:
                        associations.append((assoc, asset1.id, asset2.id))
                        assoc_idx = associations.index((assoc, asset1.id, asset2.id))
                        assoc2asset.add((assoc_idx, visible_assets.index(asset1)))
                        assoc2asset.add((assoc_idx, visible_assets.index(asset2)))
        association = Association(
            type=np.array([
                lang_serializer.association_type[(assoc.name,)]
                for (assoc, _, _) in associations
            ]),
        )
        assoc2asset_links = np.array(list(zip(*assoc2asset)))
    else:
        association = None
        assoc2asset_links = None

    if use_logic_gates:
        visible_and_or_steps = [
            node for node in visible_steps if node.type in ('and', 'or')
        ]
        logic_gates = LogicGate(
            type=np.array([
                (0 if node.type == 'and' else 1) for node in visible_and_or_steps
            ])
        )
        step2logic = set()
        logic2step = set()
        for logic_gate_id, node in enumerate(visible_and_or_steps):
            logic2step.add((logic_gate_id, visible_steps.index(node)))
            for child in filter(lambda child: child in visible_steps, node.children):
                step2logic.add((visible_steps.index(child), logic_gate_id))
        logic2step_links = np.array(list(zip(*logic2step)))
        step2logic_links = np.array(list(zip(*step2logic)))
    else:
        logic_gates = None
        logic2step_links = None
        step2logic_links = None

    return MALObsInstance(
        time=np.int64(state.sim.cur_iter),
        assets=assets,
        steps=attack_steps,
        associations=association,
        logic_gates=logic_gates,
        step2asset=np.array(list(zip(*step2asset))),
        step2step=np.array(list(zip(*step2step))),
        assoc2asset=assoc2asset_links,
        logic2step=logic2step_links,
        step2logic=step2logic_links,
    )

def attacker_update_obs(
    obs: MALObsInstance,
    state: MalSimAttackerState,
) -> MALObsInstance:
    """Update the observation of the serialized obs attacker"""
    traversable_node_ids = np.array([node.id for node in state.step_action_surface_additions])
    untraversable_node_ids = np.array([node.id for node in state.step_action_surface_removals])
    attempted_node_ids = np.array([node.id for node in state.step_attempted_nodes])
    compromised_node_ids = np.array([node.id for node in state.performed_nodes])

    if traversable_node_ids.size > 0:
        obs.steps.action_mask[np.where(np.tile(traversable_node_ids, (len(obs.steps.id), 1)) == obs.steps.id[:, np.newaxis])[0]] = True
    if untraversable_node_ids.size > 0:
        obs.steps.action_mask[np.where(np.tile(untraversable_node_ids, (len(obs.steps.id), 1)) == obs.steps.id[:, np.newaxis])[0]] = False
    if obs.steps.attempts is not None and attempted_node_ids.size > 0:
        obs.steps.attempts[np.where(np.tile(attempted_node_ids, (len(obs.steps.id), 1)) == obs.steps.id[:, np.newaxis])[0]] += 1
    if compromised_node_ids.size > 0:
        obs.steps.compromised[np.where(np.tile(compromised_node_ids, (len(obs.steps.id), 1)) == obs.steps.id[:, np.newaxis])[0]] = True
    
    return obs

def defender_state2graph(
    state: MalSimDefenderState, lang_serializer: LangSerializer, use_logic_gates: bool
) -> MALObsInstance:
    """Create a MALObsInstance of a defender's observation"""
    model = state.sim.attack_graph.model
    assert model is not None, "Attack graph must have a model"
    visible_assets = list(sorted(model.assets.values(), key=lambda asset: asset.id))
    assets = Asset(
        type=np.array([
            lang_serializer.asset_type[asset.type] for asset in visible_assets
        ]),
        id=np.array([asset.id for asset in visible_assets]),
    )

    # NOTE: This sorting of the steps is assumed to be used for the actions in the action space
    visible_steps = list(sorted({
        node for node in state.sim.attack_graph.nodes.values()
        if node.model_asset in visible_assets and node.type == "defense"
    }, key=lambda step: step.id))
    visible_steps += list(sorted({
        node for node in state.sim.attack_graph.nodes.values()
        if node.model_asset in visible_assets and node.type != "defense"
    }, key=lambda step: step.id))

    if lang_serializer.split_attack_step_types:
        attack_step_type = np.array([
            lang_serializer.attack_step_type[(node.model_asset.type, node.name)]
            for node in visible_steps if node.model_asset is not None
        ])
    else:
        attack_step_type = np.array([
            lang_serializer.attack_step_type[(node.name,)] for node in visible_steps
        ])

    attack_steps = Step(
        type=attack_step_type,
        id=np.array([node.id for node in visible_steps]),
        logic_class=np.array([
            lang_serializer.attack_step_class[node.type] for node in visible_steps
        ]),
        tags=np.array([
            lang_serializer.attack_step_tag[node.tags[0] if len(node.tags) > 0 else None]
            for node in visible_steps
        ]),
        compromised=np.array([
            (True if node in state.observed_nodes else False) for node in visible_steps
        ]),
        attempts=None,
        action_mask=np.array([True if node in state.action_surface else False for node in visible_steps], dtype=np.bool_),
    )
    step2asset_links = np.array(list(zip(*{
        (visible_steps.index(node), visible_assets.index(node.model_asset))
        for node in visible_steps if node.model_asset
    })))
    step2step_links = np.array(list(zip(*{
        (visible_steps.index(node), visible_steps.index(child))
        for node in visible_steps for child in node.children if child in visible_steps
    })))

    if len(visible_assets) > 1:
        associations: list[tuple[LanguageGraphAssociation, int, int]] = []
        assoc2asset = set()
        for asset in visible_assets:
            for fieldname, other_assets in asset.associated_assets.items():
                assoc = asset.lg_asset.associations[fieldname]
                for other_asset in filter(lambda asset: asset in visible_assets, other_assets):
                    asset1, asset2 = sorted([asset, other_asset], key=lambda asset: asset.id)
                    if (assoc, asset1.id, asset2.id) not in associations:
                        associations.append((assoc, asset1.id, asset2.id))
                        assoc_idx = associations.index((assoc, asset1.id, asset2.id))
                        assoc2asset.add((assoc_idx, visible_assets.index(asset1)))
                        assoc2asset.add((assoc_idx, visible_assets.index(asset2)))
        association = Association(
            type=np.array([
                lang_serializer.association_type[(assoc.name,)]
                for (assoc, _, _) in associations
            ]),
        )
        assoc2asset_links = np.array(list(zip(*assoc2asset)))
    else:
        association = None
        assoc2asset_links = None

    if use_logic_gates:
        visible_and_or_steps = [
            node for node in visible_steps if node.type in ('and', 'or')
        ]
        logic_gates = LogicGate(
            type=np.array([
                (0 if node.type == 'and' else 1) for node in visible_and_or_steps
            ])
        )
        step2logic = set()
        logic2step = set()
        for logic_gate_id, node in enumerate(visible_and_or_steps):
            logic2step.add((logic_gate_id, visible_steps.index(node)))
            for child in filter(lambda child: child in visible_steps, node.children):
                step2logic.add((visible_steps.index(child), logic_gate_id))
        logic2step_links = np.array(list(zip(*logic2step)))
        step2logic_links = np.array(list(zip(*step2logic)))
    else:
        logic_gates = None
        logic2step_links = None
        step2logic_links = None

    return MALObsInstance(
        time=np.int64(state.sim.cur_iter),
        assets=assets,
        steps=attack_steps,
        associations=association,
        logic_gates=logic_gates,
        step2asset=step2asset_links,
        step2step=step2step_links,
        assoc2asset=assoc2asset_links,
        logic2step=logic2step_links,
        step2logic=step2logic_links,
    )

def defender_update_obs(
    obs: MALObsInstance,
    state: MalSimDefenderState,
) -> MALObsInstance:
    """Update the observation of the serialized obs defender"""
    actionable_node_ids = np.array([node.id for node in state.step_action_surface_additions])
    non_actionable_node_ids = np.array([node.id for node in state.step_action_surface_removals])
    if actionable_node_ids.size > 0:
        obs.steps.action_mask[np.where(np.tile(actionable_node_ids, (len(obs.steps.id), 1)) == obs.steps.id[:, np.newaxis])[0]] = True
    if non_actionable_node_ids.size > 0:
        obs.steps.action_mask[np.where(np.tile(non_actionable_node_ids, (len(obs.steps.id), 1)) == obs.steps.id[:, np.newaxis])[0]] = False

    observed_node_ids = np.array([node.id for node in state.observed_nodes])
    if observed_node_ids.size > 0:
        obs.steps.compromised[np.where(np.tile(observed_node_ids, (len(obs.steps.id), 1)) == obs.steps.id[:, np.newaxis])[0]] = True
        obs.steps.compromised[np.where(np.tile(observed_node_ids, (len(obs.steps.id), 1)) != obs.steps.id[:, np.newaxis])[0]] = False
    return obs