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
from malsim.mal_simulator import MalSimAttackerState, MalSimDefenderState, MalSimulator
from maltoolbox.attackgraph import AttackGraphNode
from maltoolbox.model import ModelAsset


def create_full_obs(sim: MalSimulator, serializer: LangSerializer) -> MALObsInstance:
    def get_total_attempts(node: AttackGraphNode) -> int:
        return sum(state.num_attempts.get(node, 0) 
            for state in sim._get_attacker_agents()
        )
    def is_traversable_by_any(node: AttackGraphNode) -> bool:
        return any(sim.node_is_traversable(state.performed_nodes, node) 
            for state in sim._get_attacker_agents()
        )
    # NOTE: Sorting is for defender
    # The attacker changes the step sorting anyways
    sorted_steps = sorted([
        node for node in sim.attack_graph.nodes.values() if node.type == "defense"
    ], key=lambda node: node.id)
    sorted_steps += sorted([
        node for node in sim.attack_graph.nodes.values() if node.type != "defense"
    ], key=lambda node: node.id)
    step_type_keys: list[tuple[str, ...]]
    if serializer.split_attack_step_types:
        step_type_keys = [(node.model_asset.type, node.name) for node in sorted_steps if node.model_asset]
    else:
        step_type_keys = [(node.name,) for node in sorted_steps]
    steps = Step(
        type=np.array([
            serializer.attack_step_type[step_type_key]
            for step_type_key in step_type_keys
        ]),
        id=np.array([node.id for node in sorted_steps]),
        logic_class=np.array([
            serializer.attack_step_class[node.type]
            for node in sorted_steps
        ]),
        tags=np.array([
            serializer.attack_step_tag[node.tags[0] if len(node.tags) > 0 else None]
            for node in sorted_steps
        ]),
        compromised=np.array([
            sim.node_is_compromised(node) for node in sorted_steps
        ]),
        observable=np.array([
            sim.node_is_observable(node) for node in sorted_steps
        ]),
        attempts=np.array([
            get_total_attempts(node) for node in sorted_steps
        ]),
        action_mask=np.array([
            is_traversable_by_any(node) for node in sorted_steps
        ]),
    )
    step2step = {
        (sorted_steps.index(node), sorted_steps.index(child))
        for node in sorted_steps for child in node.children if child in sorted_steps
    }
    
    assert sim.attack_graph.model, "Attack graph needs to have a model attached to it"
    sorted_assets = [
        sim.attack_graph.model.assets[asset_id]
        for asset_id in sorted(sim.attack_graph.model.assets.keys())
    ]
    assets = Asset(
        type=np.array([serializer.asset_type[asset.type] for asset in sorted_assets]),
        id=np.array([asset.id for asset in sorted_assets]),
    )
    step2asset = {(
        sorted_steps.index(node), sorted_assets.index(node.model_asset))
        for node in sorted_steps if node.model_asset in sorted_assets
    }

    associations: list[tuple[LanguageGraphAssociation, ModelAsset, ModelAsset]] = []
    for asset in sorted_assets:
        for fieldname, other_assets in asset.associated_assets.items():
            assoc = asset.lg_asset.associations[fieldname]
            for other_asset in other_assets:
                if (
                    (assoc, asset.id, other_asset.id) not in associations 
                    and (assoc, other_asset.id, asset.id) not in associations
                ):
                    associations.append((assoc, asset, other_asset))
    sorted_associations = sorted(associations, key=lambda x: (x[0].name, x[1].id, x[2].id))
    association = Association(
        type=np.array([
            serializer.association_type[assoc.name,]
            for (assoc, _, _) in sorted_associations
        ])
    )
    assoc2asset: set[tuple[int, int]] = set()
    for i, (assoc, asset1, asset2) in enumerate(sorted_associations):
        assoc2asset.add((i, sorted_assets.index(asset1)))
        assoc2asset.add((i, sorted_assets.index(asset2)))

    sorted_and_or_steps = sorted(
        [node for node in sorted_steps if node.type in ('and', 'or')],
        key=lambda x: x.id
    )
    logic_gates = LogicGate(
        type=np.array([(0 if node.type == 'and' else 1) for node in sorted_and_or_steps]),
        id=np.array([node.id for node in sorted_and_or_steps]),
    )
    step2logic: set[tuple[int, int]] = set()
    logic2step: set[tuple[int, int]] = set()
    for logic_gate_id, node in enumerate(sorted_and_or_steps):
        logic2step.add((logic_gate_id, node.id))
        for child in node.children:
            step2logic.add((child.id, logic_gate_id))
    logic2step_links = np.array(list(zip(*logic2step)))
    step2logic_links = np.array(list(zip(*step2logic)))
        

    return MALObsInstance(
        time=np.int64(sim.cur_iter),
        assets=assets,
        steps=steps,
        associations=association,
        logic_gates=logic_gates,
        step2asset=np.array(list(zip(*step2asset))),
        step2step=np.array(list(zip(*step2step))),
        assoc2asset=np.array(list(zip(*assoc2asset))),
        logic2step=logic2step_links,
        step2logic=step2logic_links,
    )

def full_obs2attacker_obs(full_obs: MALObsInstance, state: MalSimAttackerState, see_defense_steps: bool = False) -> MALObsInstance:
    visible_asset_ids = np.array(list(sorted(
        {node.model_asset.id for node in state.performed_nodes if node.model_asset} 
        | {node.model_asset.id for node in state.action_surface if node.model_asset}
    )))
    old2new_asset_idx = {int(np.where(full_obs.assets.id == asset_id)[0]): new_idx for new_idx, asset_id in enumerate(visible_asset_ids)}
    old_asset_idx = np.array([old for old, _ in sorted(old2new_asset_idx.items(), key=lambda x: x[1])], dtype=np.int64)
    assets = Asset(
        type=full_obs.assets.type[old_asset_idx],
        id=full_obs.assets.id[old_asset_idx],
    )

    visible_steps = sorted(
        {node for node in state.sim.attack_graph.nodes.values()
        if node.model_asset and node.model_asset.id in visible_asset_ids and node.type in ("and", "or")
    }, key=lambda step: step.id)
    if see_defense_steps:
        visible_steps += sorted(
            {node for node in state.sim.attack_graph.nodes.values()
            if node.model_asset and node.model_asset.id in visible_asset_ids and node.type in ('defense', 'exist', 'notExist')
        }, key=lambda step: step.id)
    visible_step_ids = np.array([step.id for step in visible_steps])
    compromised_steps = np.array([step in state.performed_nodes for step in visible_steps])
    observable_steps = np.array([step.type in ("and", "or") for step in visible_steps])
    step_attempts = np.array([state.num_attempts.get(step, 0) for step in visible_steps])
    traversable_steps = np.array([state.sim.node_is_traversable(state.performed_nodes, step) for step in visible_steps])
    old2new_step_idx = {int(np.where(full_obs.steps.id == step_id)[0]): new_idx for new_idx, step_id in enumerate(visible_step_ids)}
    old_step_idx = np.array([old for old, _ in sorted(old2new_step_idx.items(), key=lambda x: x[1])], dtype=np.int64)
    # old_step_idx = np.where(np.tile(visible_step_ids, (len(full_obs.steps.id), 1)) == full_obs.steps.id[:, np.newaxis])[0]
    # new_step_idx = np.arange(len(old_step_idx))
    steps = Step(
        type=full_obs.steps.type[old_step_idx],
        id=full_obs.steps.id[old_step_idx],
        logic_class=full_obs.steps.logic_class[old_step_idx],
        tags=full_obs.steps.tags[old_step_idx],
        compromised=compromised_steps,
        observable=observable_steps,
        attempts=step_attempts,
        action_mask=traversable_steps,
    )


    visible_old_step2step = full_obs.step2step[:, np.isin(full_obs.step2step[0], old_step_idx) & np.isin(full_obs.step2step[1], old_step_idx)]
    new_step2step = np.stack((
        np.array([old2new_step_idx[old_step_idx] for old_step_idx in visible_old_step2step[0]]),
        np.array([old2new_step_idx[old_step_idx] for old_step_idx in visible_old_step2step[1]]),
    ), axis=0)

    visible_old_step2asset = full_obs.step2asset[:, np.isin(full_obs.step2asset[0], old_step_idx) & np.isin(full_obs.step2asset[1], old_asset_idx)]
    new_step2asset = np.stack((
        np.array([old2new_step_idx[old_step_idx] for old_step_idx in visible_old_step2asset[0]]),
        np.array([old2new_asset_idx[old_asset_idx] for old_asset_idx in visible_old_step2asset[1]]),
    ), axis=0)

    if len(visible_asset_ids) > 1 and full_obs.associations is not None and full_obs.assoc2asset is not None:
        # Get all associations connected to visible assets
        visible_assets_old_assoc2asset = full_obs.assoc2asset[:, np.isin(full_obs.assoc2asset[1], old_asset_idx)]
        # Filter out associations that are connected to only one visible asset
        unique_assocs, assoc_link_counts = np.unique(visible_assets_old_assoc2asset[0], return_counts=True)
        old_assoc_idx = unique_assocs[assoc_link_counts > 1]
        old2new_assoc_idx = {old_assoc_idx: new_idx for new_idx, old_assoc_idx in enumerate(old_assoc_idx)}
        assocs = Association(
            type=full_obs.associations.type[old_assoc_idx],
        )

        visible_old_assoc2asset = visible_assets_old_assoc2asset[:, np.isin(visible_assets_old_assoc2asset[0], old_assoc_idx)]
        new_assoc2asset = np.stack((
            np.array([old2new_assoc_idx[old_assoc_idx] for old_assoc_idx in visible_old_assoc2asset[0]]),
            np.array([old2new_asset_idx[old_asset_idx] for old_asset_idx in visible_old_assoc2asset[1]]),
        ), axis=0)
    else:
        assocs = None
        new_assoc2asset = None


    if full_obs.logic_gates is not None and full_obs.step2logic is not None and full_obs.logic2step is not None:
        # Logic gates have the same ID as the steps they are associated with
        old2new_logic_idx = {
            int(np.where(full_obs.logic_gates.id == logic_id)[0]): new_idx 
            for new_idx, logic_id in enumerate(visible_step_ids) 
            if np.isin(logic_id, full_obs.logic_gates.id)
        }
        old_logic_idx = np.array([old for old, _ in sorted(old2new_logic_idx.items(), key=lambda x: x[1])], dtype=np.int64)
        logic_gates = LogicGate(
            id=full_obs.logic_gates.id[old_logic_idx],
            type=full_obs.logic_gates.type[old_logic_idx]
        )

        visible_old_step2logic = full_obs.step2logic[:, np.isin(full_obs.step2logic[0], old_step_idx) & np.isin(full_obs.step2logic[1], old_logic_idx)]
        new_step2logic = np.stack((
            np.array([old2new_step_idx[old_step_idx] for old_step_idx in visible_old_step2logic[0]]),
            np.array([old2new_logic_idx[old_logic_idx] for old_logic_idx in visible_old_step2logic[1]]),
        ), axis=0)

        visible_old_logic2step = full_obs.logic2step[:, np.isin(full_obs.logic2step[0], old_logic_idx) & np.isin(full_obs.logic2step[1], old_step_idx)]
        new_logic2step = np.stack((
            np.array([old2new_logic_idx[old_logic_idx] for old_logic_idx in visible_old_logic2step[0]]),
            np.array([old2new_step_idx[old_step_idx] for old_step_idx in visible_old_logic2step[1]]),
        ), axis=0)
    else:
        logic_gates = None
        new_logic2step = None
        new_step2logic = None

    return MALObsInstance(
        time=full_obs.time,
        assets=assets,
        steps=steps,
        associations=assocs,
        logic_gates=logic_gates,
        step2asset=new_step2asset,
        step2step=new_step2step,
        assoc2asset=new_assoc2asset,
        logic2step=new_logic2step,
        step2logic=new_step2logic,
    )

def full_obs2defender_obs(full_obs: MALObsInstance, state: MalSimDefenderState) -> MALObsInstance:
    action_step_ids = np.array([node.id for node in state.action_surface])
    actionable = np.isin(full_obs.steps.id, action_step_ids)
    observed_step_ids = np.array([node.id for node in state.observed_nodes])
    observed = np.isin(full_obs.steps.id, observed_step_ids)

    return MALObsInstance(
        time=full_obs.time,
        assets=full_obs.assets,
        steps=Step(
            type=full_obs.steps.type,
            id=full_obs.steps.id,
            logic_class=full_obs.steps.logic_class,
            tags=full_obs.steps.tags,
            compromised=observed,
            observable=full_obs.steps.observable,
            attempts=None,
            action_mask=actionable,
        ),
        associations=full_obs.associations,
        logic_gates=full_obs.logic_gates,
        step2asset=full_obs.step2asset,
        step2step=full_obs.step2step,
        assoc2asset=full_obs.assoc2asset,
        logic2step=full_obs.logic2step,
        step2logic=full_obs.step2logic,
    )


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