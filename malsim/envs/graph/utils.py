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
    """Create a full MALObsInstance.
    This observation can be updated for individual agents."""

    def get_total_attempts(node: AttackGraphNode) -> int:
        return sum(
            state.num_attempts.get(node, 0) for state in sim._get_attacker_agents()
        )

    def is_traversable_by_any(node: AttackGraphNode) -> bool:
        return any(
            sim.node_is_traversable(state.performed_nodes, node)
            for state in sim._get_attacker_agents()
        )

    # NOTE: Sorting is for defender
    # The attacker changes the step sorting anyways
    sorted_steps = sorted(
        [node for node in sim.attack_graph.nodes.values() if node.type == 'defense'],
        key=lambda node: node.id,
    )
    sorted_steps += sorted(
        [node for node in sim.attack_graph.nodes.values() if node.type != 'defense'],
        key=lambda node: node.id,
    )
    step_type_keys: list[tuple[str, ...]]
    if serializer.split_step_types:
        step_type_keys = [
            (node.model_asset.type, node.name)
            for node in sorted_steps
            if node.model_asset
        ]
    else:
        step_type_keys = [(node.name,) for node in sorted_steps]
    steps = Step(
        type=np.array(
            [serializer.step_type[step_type_key] for step_type_key in step_type_keys]
        ),
        id=np.array([node.id for node in sorted_steps]),
        logic_class=np.array(
            [serializer.step_class[node.type] for node in sorted_steps]
        ),
        tags=np.array(
            [
                serializer.step_tag[node.tags[0] if len(node.tags) > 0 else None]
                for node in sorted_steps
            ]
        ),
        compromised=np.array([sim.node_is_compromised(node) for node in sorted_steps]),
        observable=np.array([sim.node_is_observable(node) for node in sorted_steps]),
        attempts=np.array([get_total_attempts(node) for node in sorted_steps]),
        action_mask=np.array([is_traversable_by_any(node) for node in sorted_steps]),
    )
    step2step = {
        (sorted_steps.index(node), sorted_steps.index(child))
        for node in sorted_steps
        for child in node.children
        if child in sorted_steps
    }

    assert sim.attack_graph.model, 'Attack graph needs to have a model attached to it'
    sorted_assets = [
        sim.attack_graph.model.assets[asset_id]
        for asset_id in sorted(sim.attack_graph.model.assets.keys())
    ]
    step2asset = {
        (sorted_steps.index(node), sorted_assets.index(node.model_asset))
        for node in sorted_steps
        if node.model_asset in sorted_assets
    }
    step2asset_links = np.array(list(zip(*step2asset)), dtype=np.int64)
    # Compute action_mask for each asset: True if any connected step
    # has action_mask == True
    asset_action_mask = np.zeros(len(sorted_assets), dtype=np.bool_)
    # For each asset, check if any connected step has action_mask == True
    for asset_idx in range(len(sorted_assets)):
        connected_step_indices = step2asset_links[0, step2asset_links[1] == asset_idx]
        if len(connected_step_indices) > 0:
            asset_action_mask[asset_idx] = steps.action_mask[
                connected_step_indices
            ].any()
    assets = Asset(
        type=np.array([serializer.asset_type[asset.type] for asset in sorted_assets]),
        id=np.array([asset.id for asset in sorted_assets]),
        action_mask=asset_action_mask,
    )

    associations: list[tuple[LanguageGraphAssociation, ModelAsset, ModelAsset]] = []
    for asset in sorted_assets:
        for fieldname, other_assets in asset.associated_assets.items():
            assoc = asset.lg_asset.associations[fieldname]
            for other_asset in other_assets:
                if (assoc, asset.id, other_asset.id) not in associations and (
                    assoc,
                    other_asset.id,
                    asset.id,
                ) not in associations:
                    associations.append((assoc, asset, other_asset))
    sorted_associations = sorted(
        associations, key=lambda x: (x[0].name, x[1].id, x[2].id)
    )
    association = Association(
        type=np.array(
            [
                serializer.association_type[assoc.name,]
                for (assoc, _, _) in sorted_associations
            ]
        )
    )
    assoc2asset: set[tuple[int, int]] = set()
    for i, (assoc, asset1, asset2) in enumerate(sorted_associations):
        assoc2asset.add((i, sorted_assets.index(asset1)))
        assoc2asset.add((i, sorted_assets.index(asset2)))

    asset2asset: set[tuple[int, int]] = set()
    for assoc, asset1, asset2 in sorted_associations:
        asset1_idx = sorted_assets.index(asset1)
        asset2_idx = sorted_assets.index(asset2)
        # NOTE: Unidirectional links between assets
        # Just duplicate the transpose if you want bidirectional links
        asset2asset.add((asset1_idx, asset2_idx))

    sorted_and_or_steps = sorted(
        [node for node in sorted_steps if node.type in ('and', 'or')],
        key=lambda x: x.id,
    )
    logic_gates = LogicGate(
        type=np.array(
            [(0 if node.type == 'and' else 1) for node in sorted_and_or_steps],
            dtype=np.int64,
        ),
        id=np.array([node.id for node in sorted_and_or_steps], dtype=np.int64),
    )
    step2logic: set[tuple[int, int]] = set()
    logic2step: set[tuple[int, int]] = set()
    for logic_gate_id, node in enumerate(sorted_and_or_steps):
        logic2step.add((logic_gate_id, node.id))
        for child in node.children:
            step2logic.add((child.id, logic_gate_id))
    if logic2step:
        logic2step_links = np.array(list(zip(*logic2step)), dtype=np.int64)
    else:
        logic2step_links = np.empty((2, 0), dtype=np.int64)
    if step2logic:
        step2logic_links = np.array(list(zip(*step2logic)), dtype=np.int64)
    else:
        step2logic_links = np.empty((2, 0), dtype=np.int64)

    if asset2asset:
        asset2asset_links = np.array(list(zip(*asset2asset)), dtype=np.int64)
    else:
        asset2asset_links = np.empty((2, 0), dtype=np.int64)

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
        asset2asset=asset2asset_links,
    )


def full_obs2attacker_obs(
    full_obs: MALObsInstance,
    state: MalSimAttackerState,
    serializer: LangSerializer,
    see_defense_steps: bool = False,
) -> MALObsInstance:
    """Create an attacker observation from a full observation.

    This observation makes all assets with compromised nodes visible to the attacker.
    All steps that are on a visible asset are also visible to the attacker.
    NOTE: This comes from an assumption that the attacker "knows" the
    language of generalization used to create the full observation.

    Sorts the steps so that all `and`/`or` steps have lower indices than
    `defense`/`exist`/`notExist` steps. Re-indexes the step types so that step types of
    `and`/`or` have lower indices than other step types.

    Args:
        full_obs: The full observation.
        state: The state of the attacker.
        serializer: The language serializer.
        see_defense_steps: Whether to include defense steps in the observation.

    Returns:
        The attacker observation.
    """
    # Create lookup dictionaries for efficient index mapping
    asset_id_to_idx = {asset_id: idx for idx, asset_id in enumerate(full_obs.assets.id)}
    step_id_to_idx = {step_id: idx for idx, step_id in enumerate(full_obs.steps.id)}
    logic_id_to_idx = {
        logic_id: idx for idx, logic_id in enumerate(full_obs.logic_gates.id)
    }
    performed_nodes_set = set(state.performed_nodes)

    # Get all visible asset IDs for model asset of nodes that
    # are performed or on the action surface.
    visible_asset_ids = np.array(
        sorted(
            {node.model_asset.id for node in state.performed_nodes if node.model_asset}
            | {node.model_asset.id for node in state.action_surface if node.model_asset}
        )
    )
    visible_asset_ids_set = set(visible_asset_ids)

    # Map old asset IDs to new asset IDs using lookup
    num_visible_assets = len(visible_asset_ids)
    new2old_asset_idx = np.array(
        [asset_id_to_idx[asset_id] for asset_id in visible_asset_ids], dtype=np.int64
    )
    old2new_asset_idx = {
        old_idx: new_idx for new_idx, old_idx in enumerate(new2old_asset_idx)
    }

    # Get all visible steps for model assets that are visible to the attacker.
    visible_steps = []
    for node in state.sim.attack_graph.nodes.values():
        if (
            node.model_asset
            and node.model_asset.id in visible_asset_ids_set
            and node.type in ('and', 'or')
        ):
            visible_steps.append(node)
    visible_steps.sort(key=lambda step: step.id)

    if see_defense_steps:
        defense_steps = []
        for node in state.sim.attack_graph.nodes.values():
            if (
                node.model_asset
                and node.model_asset.id in visible_asset_ids_set
                and node.type in ('defense', 'exist', 'notExist')
            ):
                defense_steps.append(node)
        defense_steps.sort(key=lambda step: step.id)
        visible_steps.extend(defense_steps)

    # Get step attributes in a single pass where possible
    num_visible_steps = len(visible_steps)
    visible_step_ids = np.empty(num_visible_steps, dtype=np.int64)
    compromised_steps = np.empty(num_visible_steps, dtype=np.bool_)
    observable_steps = np.empty(num_visible_steps, dtype=np.bool_)
    step_attempts = np.empty(num_visible_steps, dtype=np.int64)
    traversable_steps = np.empty(num_visible_steps, dtype=np.bool_)

    for i, step in enumerate(visible_steps):
        visible_step_ids[i] = step.id
        compromised_steps[i] = step in performed_nodes_set
        observable_steps[i] = step.type in ('and', 'or')
        step_attempts[i] = state.num_attempts.get(step, 0)
        traversable_steps[i] = state.sim.node_is_traversable(
            state.performed_nodes, step
        )

    # Map old step indices to new step indices using lookup
    new2old_step_idx = np.array(
        [step_id_to_idx[step_id] for step_id in visible_step_ids], dtype=np.int64
    )
    old2new_step_idx = {
        old_idx: new_idx for new_idx, old_idx in enumerate(new2old_step_idx)
    }

    # Re-index step type according to attacker serialization.
    step_type = full_obs.steps.type[new2old_step_idx]
    step_type_attacker_indexing = serializer.step_type2attacker_step_type[step_type]
    steps = Step(
        type=step_type_attacker_indexing,
        id=full_obs.steps.id[new2old_step_idx],
        logic_class=full_obs.steps.logic_class[new2old_step_idx],
        tags=full_obs.steps.tags[new2old_step_idx],
        compromised=compromised_steps,
        observable=observable_steps,
        attempts=step_attempts,
        action_mask=traversable_steps,
    )

    # Get all step2step links (with old step indices) for visible steps
    visible_old_steps_mask = np.zeros(full_obs.steps.id.shape[0], dtype=bool)
    visible_old_steps_mask[new2old_step_idx] = True
    step2step_mask = (
        visible_old_steps_mask[full_obs.step2step[0]]
        & visible_old_steps_mask[full_obs.step2step[1]]
    )
    visible_old_step2step = full_obs.step2step[:, step2step_mask]
    # Map old step2step links to new step2step links using vectorized lookup
    new_step2step = np.array(
        [
            [
                old2new_step_idx[old_step_idx]
                for old_step_idx in visible_old_step2step[0]
            ],
            [
                old2new_step_idx[old_step_idx]
                for old_step_idx in visible_old_step2step[1]
            ],
        ],
        dtype=np.int64,
    )

    # Get all step2asset links (with old step indices) for visible steps
    visible_old_asset_mask = np.zeros(full_obs.assets.id.shape[0], dtype=bool)
    visible_old_asset_mask[new2old_asset_idx] = True
    step2asset_mask = (
        visible_old_steps_mask[full_obs.step2asset[0]]
        & visible_old_asset_mask[full_obs.step2asset[1]]
    )
    visible_old_step2asset = full_obs.step2asset[:, step2asset_mask]
    # Map old step2asset links to new step2asset links using vectorized lookup
    new_step2asset = np.array(
        [
            [
                old2new_step_idx[old_step_idx]
                for old_step_idx in visible_old_step2asset[0]
            ],
            [
                old2new_asset_idx[old_asset_idx]
                for old_asset_idx in visible_old_step2asset[1]
            ],
        ],
        dtype=np.int64,
    )

    # Compute action_mask for filtered assets: True if any connected
    # visible step has action_mask == True (vectorized)
    asset_action_mask = np.zeros(num_visible_assets, dtype=np.bool_)
    if new_step2asset.shape[1] > 0:
        for asset_idx in range(num_visible_assets):
            connected_step_indices = new_step2asset[0, new_step2asset[1] == asset_idx]
            if len(connected_step_indices) > 0:
                asset_action_mask[asset_idx] = steps.action_mask[
                    connected_step_indices
                ].any()

    assets = Asset(
        type=full_obs.assets.type[new2old_asset_idx],
        id=full_obs.assets.id[new2old_asset_idx],
        action_mask=asset_action_mask,
    )

    if (
        num_visible_assets > 1
        and full_obs.associations is not None
        and full_obs.assoc2asset is not None
    ):
        # Get all associations connected to visible assets
        visible_assets_old_assoc2asset = full_obs.assoc2asset[
            :, visible_old_asset_mask[full_obs.assoc2asset[1]]
        ]
        # Filter out associations that are connected to only one visible asset
        unique_assocs, assoc_link_counts = np.unique(
            visible_assets_old_assoc2asset[0], return_counts=True
        )
        old_assoc_idx = unique_assocs[assoc_link_counts > 1]
        old2new_assoc_idx = {
            old_idx: new_idx for new_idx, old_idx in enumerate(old_assoc_idx)
        }
        assocs = Association(type=full_obs.associations.type[old_assoc_idx])

        visible_old_assoc2asset = visible_assets_old_assoc2asset[
            :, np.isin(visible_assets_old_assoc2asset[0], old_assoc_idx)
        ]
        new_assoc2asset = np.array(
            [
                [
                    old2new_assoc_idx[old_assoc_idx]
                    for old_assoc_idx in visible_old_assoc2asset[0]
                ],
                [
                    old2new_asset_idx[old_asset_idx]
                    for old_asset_idx in visible_old_assoc2asset[1]
                ],
            ],
            dtype=np.int64,
        )
    else:
        assocs = None
        new_assoc2asset = None

    # Logic gates have the same ID as the steps they are associated with
    # Map old logic indices to new logic indices
    visible_logic_ids = np.array(
        [logic_id for logic_id in visible_step_ids if logic_id in logic_id_to_idx]
    )
    if len(visible_logic_ids) > 0:
        new2old_logic_idx = np.array(
            [logic_id_to_idx[logic_id] for logic_id in visible_logic_ids],
            dtype=np.int64,
        )
        old2new_logic_idx = {
            old_idx: new_idx for new_idx, old_idx in enumerate(new2old_logic_idx)
        }
        # TODO: Check if full_obs.logic_gates.type needs to be re-indexed
        # with attacker serialization.
        logic_gates = LogicGate(
            id=full_obs.logic_gates.id[new2old_logic_idx],
            type=full_obs.logic_gates.type[new2old_logic_idx],
        )

        # Get all step2logic links (with old step and logic indices)
        # for visible steps and logic gates
        visible_old_logic_mask = np.zeros(full_obs.logic_gates.id.shape[0], dtype=bool)
        visible_old_logic_mask[new2old_logic_idx] = True
        step2logic_mask = (
            visible_old_steps_mask[full_obs.step2logic[0]]
            & visible_old_logic_mask[full_obs.step2logic[1]]
        )
        visible_old_step2logic = full_obs.step2logic[:, step2logic_mask]
        new_step2logic = np.array(
            [
                [
                    old2new_step_idx[old_step_idx]
                    for old_step_idx in visible_old_step2logic[0]
                ],
                [
                    old2new_logic_idx[old_logic_idx]
                    for old_logic_idx in visible_old_step2logic[1]
                ],
            ],
            dtype=np.int64,
        )

        # Get all logic2step links (with old logic and step indices)
        # for visible steps and logic gates
        logic2step_mask = (
            visible_old_logic_mask[full_obs.logic2step[0]]
            & visible_old_steps_mask[full_obs.logic2step[1]]
        )
        visible_old_logic2step = full_obs.logic2step[:, logic2step_mask]
        new_logic2step = np.array(
            [
                [
                    old2new_logic_idx[old_logic_idx]
                    for old_logic_idx in visible_old_logic2step[0]
                ],
                [
                    old2new_step_idx[old_step_idx]
                    for old_step_idx in visible_old_logic2step[1]
                ],
            ],
            dtype=np.int64,
        )
    else:
        logic_gates = LogicGate(
            id=np.empty(0, dtype=np.int64), type=np.empty(0, dtype=np.int64)
        )
        new_step2logic = np.empty((2, 0), dtype=np.int64)
        new_logic2step = np.empty((2, 0), dtype=np.int64)

    # Filter asset2asset links to only include visible assets
    asset2asset_mask = (
        visible_old_asset_mask[full_obs.asset2asset[0]]
        & visible_old_asset_mask[full_obs.asset2asset[1]]
    )
    visible_old_asset2asset = full_obs.asset2asset[:, asset2asset_mask]
    new_asset2asset = np.array(
        [
            [
                old2new_asset_idx[old_asset_idx]
                for old_asset_idx in visible_old_asset2asset[0]
            ],
            [
                old2new_asset_idx[old_asset_idx]
                for old_asset_idx in visible_old_asset2asset[1]
            ],
        ],
        dtype=np.int64,
    )

    return MALObsInstance(
        time=np.int64(state.sim.cur_iter),
        assets=assets,
        steps=steps,
        associations=assocs,
        logic_gates=logic_gates,
        step2asset=new_step2asset,
        step2step=new_step2step,
        assoc2asset=new_assoc2asset,
        logic2step=new_logic2step,
        step2logic=new_step2logic,
        asset2asset=new_asset2asset,
    )


def full_obs2defender_obs(
    full_obs: MALObsInstance, state: MalSimDefenderState, serializer: LangSerializer
) -> MALObsInstance:
    """Create a defender observation from a full observation.

    This observation makes all assets and steps visible. The defender can only see if
    an attack step has been compromised if the state says that it has been observed.

    Assumes that steps are sorted so that all `defense` steps have lower indices than
    `and`/`or`/`defense`/`exist`/`notExist` steps in the full observation. Re-indexes
    the step types so that step types of `defense` have lower indices than other step
     types.

    Args:
        full_obs: The full observation.
        state: The state of the defender.
        serializer: The language serializer.

    Returns:
        The defender observation.
    """
    action_step_ids = np.array([node.id for node in state.action_surface])
    actionable = np.isin(full_obs.steps.id, action_step_ids)
    observed_step_ids = np.array([node.id for node in state.observed_nodes])
    observed = np.isin(full_obs.steps.id, observed_step_ids)

    return MALObsInstance(
        time=np.int64(state.sim.cur_iter),
        assets=full_obs.assets,
        steps=Step(
            type=serializer.step_type2defender_step_type[full_obs.steps.type],
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
        asset2asset=full_obs.asset2asset,
    )
