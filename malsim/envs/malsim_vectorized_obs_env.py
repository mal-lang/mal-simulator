""""MalSimVectorizedObsEnv:
    - Abide to the ParallelEnv interface
    - Build serialized observations from the MalSimulator state
    - step() assumes that actions are given as AttackGraphNodes
    - Used by AttackerEnv/DefenderEnv to be able to 
"""

from __future__ import annotations

from typing import Any, Optional
import functools
import logging
import sys

import numpy as np
from gymnasium.spaces import MultiDiscrete, Box, Dict
from pettingzoo import ParallelEnv
from maltoolbox.attackgraph import AttackGraphNode

from ..mal_simulator import (
    MalSimulator,
    MalSimAgentState,
    MalSimAttackerState,
    MalSimDefenderState
)


ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)


class MalSimVectorizedObsEnv(ParallelEnv): # type: ignore
    """
    Environment that runs simulation between agents.
    Builds serialized observations.
    Implements the ParallelEnv.
    """

    def __init__(self, sim: MalSimulator):

        self.sim = sim
        self.attack_graph = sim.attack_graph
        assert self.attack_graph.model, (
            "Attack graph in simulator needs to have a model attached to it"
        )

        # List mapping from node/asset index to id/name/type
        self._index_to_id = [n.id for n in self.attack_graph.nodes.values()]
        self._index_to_full_name = (
            [n.full_name for n in self.attack_graph.nodes.values()]
        )
        self._index_to_asset_type = (
            [n.name for n in self.attack_graph.lang_graph.assets.values()]
        )

        unique_step_type_names = {
            n.full_name
            for asset in self.attack_graph.lang_graph.assets.values()
            for n in asset.attack_steps.values()
        }
        self._index_to_step_type_name = list(unique_step_type_names)

        self._index_to_model_asset_id = (
            [int(asset_id) for asset_id in self.attack_graph.model.assets]
        )

        unique_assoc_type_names = {
            assoc.full_name
            for asset in self.attack_graph.lang_graph.assets.values()
            for assoc in asset.associations.values()
        }
        self._index_to_model_assoc_type = list(unique_assoc_type_names)

        # Lookup dicts attribute to index
        self._id_to_index = {
            n: i for i, n in enumerate(self._index_to_id)}
        self._asset_type_to_index = {
            n: i for i, n in enumerate(self._index_to_asset_type)}
        self._step_type_name_to_index = {
            n: i for i, n in enumerate(self._index_to_step_type_name)
        }
        self._model_asset_id_to_index = {
            asset: i for i, asset in enumerate(self._index_to_model_asset_id)
        }
        self._model_assoc_type_to_index = {
            assoc_type: i for i, assoc_type in
            enumerate(self._index_to_model_assoc_type)
        }

        self._blank_observation = self._create_blank_observation()
        self._agent_observations: dict[str, Any] = {}
        self._agent_infos: dict[str, Any] = {}

    @property
    def agents(self) -> list[str]:
        """Required by ParallelEnv"""
        return list(self.sim._alive_agents)

    @property
    def possible_agents(self) -> list[str]:
        """Required by ParallelEnv"""
        return list(self.sim._agent_states.keys())

    def get_agent_state(self, agent_name: str) -> MalSimAgentState:
        return self.sim.agent_states[agent_name]

    def _create_blank_observation(
            self, default_obs_state: int = -1
        ) -> dict[str, Any]:
        """Create the initial observation"""
        # For now, an `object` is an attack step
        num_steps = len(self.sim.attack_graph.nodes)

        observation = {
            # If no observability set for node, assume observable.
            "is_observable": [step.extras.get('observable', 1)
                           for step in self.attack_graph.nodes.values()],
            # Same goes for actionable.
            "is_actionable": [step.extras.get('actionable', 1)
                           for step in self.attack_graph.nodes.values()],
            "observed_state": num_steps * [default_obs_state],
            "remaining_ttc": num_steps * [0],
            "asset_type": [
                self._asset_type_to_index[step.lg_attack_step.asset.name]
                for step in self.attack_graph.nodes.values()],
            "asset_id": [step.model_asset.id
                         for step in self.attack_graph.nodes.values()
                         if step.model_asset],
            "step_name": [
                self._step_type_name_to_index.get(
                    str(step.lg_attack_step.asset.name + ":" + step.name)
                ) for step in self.attack_graph.nodes.values()],
        }

        logger.debug(
            'Create blank observation with %d attack steps.', num_steps)

        # Add attack graph edges to observation
        observation["attack_graph_edges"] = []
        for attack_step in self.attack_graph.nodes.values():
            # For determinism we need to order the children
            ordered_children = list(attack_step.children)
            ordered_children.sort(key=lambda n: n.id)
            for child in ordered_children:
                observation["attack_graph_edges"].append(
                    [
                        self._id_to_index[attack_step.id],
                        self._id_to_index[child.id]
                    ]
                )

        # Add reverse attack graph edges for defense steps (required by some
        # defender agent logic)
        for attack_step in self.attack_graph.nodes.values():
            if attack_step.type == "defense":
                # For determinism we need to order the children
                ordered_children = list(attack_step.children)
                ordered_children.sort(key=lambda n: n.id)
                for child in ordered_children:
                    observation["attack_graph_edges"].append(
                        [
                            self._id_to_index[child.id],
                            self._id_to_index[attack_step.id]
                        ]
                    )

        # Add instance model assets
        observation["model_asset_id"] = []
        observation["model_asset_type"] = []
        observation["model_edges_ids"] = []
        observation["model_edges_type"] = []

        assert self.attack_graph.model, "Graph needs model attached to it"
        for asset in self.attack_graph.model.assets.values():
            observation["model_asset_id"].append(asset.id)
            observation["model_asset_type"].append(
                self._asset_type_to_index[asset.type])

            for fieldname, other_assets in asset.associated_assets.items():
                for other_asset in other_assets:
                    observation["model_edges_ids"].append(
                        [
                            self._model_asset_id_to_index[asset.id],
                            self._model_asset_id_to_index[other_asset.id]
                        ]
                    )

                    lg_assoc = asset.lg_asset.associations[fieldname]
                    observation["model_edges_type"].append(
                        self._model_assoc_type_to_index[lg_assoc.full_name]
                    )

        np_obs = {
            "is_observable": np.array(observation["is_observable"],
                             dtype=np.int8),
            "is_actionable": np.array(observation["is_actionable"],
                             dtype=np.int8),
            "observed_state": np.array(observation["observed_state"],
                              dtype=np.int8),
            "remaining_ttc": np.array(observation["remaining_ttc"],
                             dtype=np.int64),
            "asset_type": np.array(observation["asset_type"], dtype=np.int64),
            "asset_id": np.array(observation["asset_id"], dtype=np.int64),
            "step_name": np.array(observation["step_name"], dtype=np.int64),
            "attack_graph_edges": np.array(observation["attack_graph_edges"],
                                  dtype=np.int64),
            "model_asset_id": np.array(observation["model_asset_id"],
                                dtype=np.int64),
            "model_asset_type": np.array(observation["model_asset_type"],
                                dtype=np.int64),
            "model_edges_ids": np.array(observation["model_edges_ids"],
                                dtype=np.int64),
            "model_edges_type": np.array(observation["model_edges_type"],
                                dtype=np.int64)
        }
        return np_obs

    def create_action_mask(self, agent_state: MalSimAgentState) -> dict[str, Any]:
        """
        Create an action mask for an agent based on its action_surface.

        Parameters:
            agent: The agent for whom the mask is created.

        Returns:
            A dictionary with the action mask for the agent.
        """

        available_actions = [0] * len(self.sim.attack_graph.nodes)
        can_wait = 1 if isinstance(agent_state, MalSimDefenderState) else 0
        can_act = 0

        for node in agent_state.action_surface:

            if isinstance(agent_state, MalSimDefenderState):
                # Defender can act on its whole action surface
                index = self._id_to_index[node.id]
                available_actions[index] = 1
                can_act = 1

            if isinstance(agent_state, MalSimAttackerState):
                # Attacker can only act on nodes that are not compromised

                if node not in agent_state.performed_nodes:
                    index = self._id_to_index[node.id]
                    available_actions[index] = 1
                    can_act = 1

        return {
            'action_mask': (
                np.array([can_wait, can_act], dtype=np.int8),
                np.array(available_actions, dtype=np.int8)
            )
        }

    def _update_agent_infos(self) -> None:
        for agent in self.sim.agent_states.values():
            self._agent_infos[agent.name] = self.create_action_mask(agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: Optional[str] = None) -> MultiDiscrete:
        num_actions = 2  # two actions: wait or use
        # For now, an `object` is an attack step
        num_steps = len(self.sim.attack_graph.nodes)
        return MultiDiscrete([num_actions, num_steps], dtype=np.int64)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_name: Optional[str] = None) -> Dict:
        # For now, an `object` is an attack step
        assert self.attack_graph.model, (
            "Attack graph in simulator needs to have a model attached to it"
        )
        num_assets = len(self.attack_graph.model.assets)
        num_steps = len(self.attack_graph.nodes)
        num_lang_asset_types = len(self.sim.attack_graph.lang_graph.assets)

        unique_step_types = set()
        for asset_type in self.sim.attack_graph.lang_graph.assets.values():
            unique_step_types |= set(asset_type.attack_steps.values())
        num_lang_attack_steps = len(unique_step_types)

        unique_assoc_type_names = set()
        for asset_type in self.sim.attack_graph.lang_graph.assets.values():
            for assoc_type in asset_type.associations.values():
                unique_assoc_type_names.add(
                    assoc_type.full_name
                )
        num_lang_association_types = len(unique_assoc_type_names)

        num_attack_graph_edges = len(
            self._blank_observation["attack_graph_edges"])
        num_model_edges = len(
            self._blank_observation["model_edges_ids"])        
        
        return Dict(
            {
                "is_observable": Box(
                    0, 1, shape=(num_steps,), dtype=np.int8
                ),  #  0 for unobservable, 1 for observable
                "is_actionable": Box(
                    0, 1, shape=(num_steps,), dtype=np.int8
                ),  #  0 for non-actionable, 1 for actionable
                "observed_state": Box(
                    -1, 1, shape=(num_steps,), dtype=np.int8
                ),  # -1 for unknown,
                #  0 for disabled/not compromised,
                #  1 for enabled/compromised
                "remaining_ttc": Box(
                    0, sys.maxsize, shape=(num_steps,), dtype=np.int64
                ),  # remaining TTC
                "asset_type": Box(
                    0,
                    num_lang_asset_types,
                    shape=(num_steps,),
                    dtype=np.int64,
                ),  # asset type
                "asset_id": Box(
                    0, sys.maxsize, shape=(num_steps,), dtype=np.int64
                ),  # asset id
                "step_name": Box(
                    0,
                    num_lang_attack_steps,
                    shape=(num_steps,),
                    dtype=np.int64,
                ),  # attack/defense step name
                "attack_graph_edges": Box(
                    0,
                    num_steps,
                    shape=(num_attack_graph_edges, 2),
                    dtype=np.int64,
                ),  # edges between attack graph steps
                "model_asset_id": Box(
                    0,
                    num_assets,
                    shape=(num_assets,),
                    dtype=np.int64,
                ),  # instance model asset ids
                "model_asset_type": Box(
                    0,
                    num_lang_asset_types,
                    shape=(num_assets,),
                    dtype=np.int64,
                ),  # instance model asset types
                "model_edges_ids": Box(
                    0,
                    num_assets,
                    shape=(num_model_edges, 2),
                    dtype=np.int64,
                ),  # instance model edge ids
                "model_edges_type": Box(
                    0,
                    num_lang_association_types,
                    shape=(num_model_edges, ),
                    dtype=np.int64,
                ),  # instance model edge types
            }
        )

    def index_to_node(self, index: int) -> AttackGraphNode:
        """Get a node from the attack graph by index

        Index is the position of the node in the lookup list.
        First convert index to id and then fetch the node from the
        AttackGraph.

        Raise LookupError if node with given index does not map to a node in
        the attack graph and IndexError if the index is out of range for the
        lookup list.

        Returns:
        Attack graph node matching the id of the index in the lookup list
        """

        if index >= len(self._index_to_id):
            raise IndexError(
                f'Index {index}, is out of range of the '
                f'lookup list which is of length {len(self._index_to_id)}'
            )

        node_id = self._index_to_id[index]
        node = self.sim.attack_graph.nodes[node_id]
        if not node:
            raise LookupError(
                f'Index {index} (id: {node_id}), does not map to a node'
            )
        return node

    def node_to_index(self, node: AttackGraphNode) -> int:
        """Get the index of an attack graph node

        Returns:
        Index of the attack graph node in the lookup list
        """

        assert node, "Node can not be None"
        return self._id_to_index[node.id]

    def serialized_action_to_node(
            self, serialized_action: tuple[int, int]
        ) -> list[AttackGraphNode]:
        """Convert serialized action to malsim action format
        
        (0, None) -> []
        (1, idx) -> [Node with idx]

        Currently supports single action only.
        """
        nodes = []
        act, step_idx = serialized_action
        if act:
            nodes = [self.index_to_node(step_idx)]
        return nodes

    def register_attacker(
            self, attacker_name: str, entry_points: set[AttackGraphNode]
        ) -> None:
        self.sim.register_attacker(attacker_name, entry_points)
        agent = self.sim.agent_states[attacker_name]
        self._init_agent(agent)

    def register_defender(self, defender_name: str) -> None:
        self.sim.register_defender(defender_name)
        agent = self.sim.agent_states[defender_name]
        self._init_agent(agent)

    def _init_agent(self, agent: MalSimAgentState) -> None:
        # Fill dicts with env specific agent obs/infos
        self._agent_observations[agent.name] = \
            self._create_blank_observation()

        self._agent_infos[agent.name] = \
            self.create_action_mask(agent)

    def _update_attacker_obs(
            self,
            compromised_nodes: set[AttackGraphNode],
            disabled_nodes: set[AttackGraphNode],
            attacker_agent: MalSimAttackerState
        ) -> None:
        """Update the observation of the serialized obs attacker"""

        def _enable_node(
                node: AttackGraphNode, agent_observation: dict[str, Any]
            ) -> None:
            """Set enabled node obs state to enabled and
            its children to disabled"""

            # Mark enabled node obs state with 1 (enabled)
            node_index = self._id_to_index[node.id]
            agent_observation['observed_state'][node_index] = 1

            # Mark unknown (-1) children node obs states with 0 (disabled)
            for child_node in node.children:
                child_index = self._id_to_index[child_node.id]
                child_obs = agent_observation['observed_state'][child_index]
                if child_obs == -1:
                    agent_observation['observed_state'][child_index] = 0

        attacker_observation = self._agent_observations[attacker_agent.name]

        for node in compromised_nodes:
            if node in attacker_agent.performed_nodes:
                # Enable node
                logger.debug("Enable %s in attacker obs", node.full_name)
                _enable_node(node, attacker_observation)

        for node in disabled_nodes:
            if (
                node in attacker_agent.performed_nodes
                and node not in attacker_agent.entry_points
            ):
                logger.debug("Disable %s in attacker obs", node.full_name)
                # Mark attacker compromised steps that were
                # disabled by a defense as disabled in obs
                node_idx = self.node_to_index(node)
                attacker_observation['observed_state'][node_idx] = 0

    def _update_defender_obs(
            self,
            compromised_nodes: set[AttackGraphNode],
            disabled_nodes: set[AttackGraphNode],
            defender_agent: MalSimDefenderState
        ) -> None:
        """Update the observation of the defender"""

        defender_observation = self._agent_observations[defender_agent.name]

        for node in compromised_nodes:
            logger.debug("Enable %s in defender obs", node.full_name)
            node_idx = self.node_to_index(node)
            defender_observation['observed_state'][node_idx] = 1

        for node in disabled_nodes:
            is_entrypoint = node.extras.get('entrypoint', False)
            if not is_entrypoint:
                logger.debug("Disable %s in defender obs", node.full_name)
                node_idx = self.node_to_index(node)
                defender_observation['observed_state'][node_idx] = 0

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset simulator and return current
        observation and infos for each agent"""

        if seed is not None:
            self.sim.sim_settings.seed = seed
        self.sim.reset(options=options)
        self.attack_graph = self.sim.attack_graph # new ref
        assert self.attack_graph.model, (
            "Attack graph in simulator needs to have a model attached to it"
        )

        pre_enabled_nodes: set[AttackGraphNode] = set()
        for agent in self.sim.agent_states.values():
            # Reset observation and action mask for agents
            self._agent_observations[agent.name] = \
                self._create_blank_observation()
            self._agent_infos[agent.name] = \
                self.create_action_mask(agent)
            pre_enabled_nodes |= agent.performed_nodes

        self._update_observations(
            pre_enabled_nodes, set()
        )

        # TODO: should we return copies instead so they are not modified externally?
        return self._agent_observations, self._agent_infos

    def _update_observations(
            self,
            compromised_nodes: set[AttackGraphNode],
            disabled_nodes: set[AttackGraphNode]
        ) -> None:
        """Update observations of all agents"""

        if not self.sim.sim_settings.uncompromise_untraversable_steps:
            disabled_nodes = set()

        # TODO: Is this correct? All attackers get the same compromised_nodes?
        logger.debug("Enable:\n\t%s", [n.full_name for n in compromised_nodes])
        logger.debug("Disable:\n\t%s", [n.full_name for n in disabled_nodes])

        for agent in self.sim.agent_states.values():
            if isinstance(agent, MalSimAttackerState):
                self._update_attacker_obs(
                    compromised_nodes, disabled_nodes, agent
                )
            elif isinstance(agent, MalSimDefenderState):
                self._update_defender_obs(
                    compromised_nodes, disabled_nodes, agent
                )

    def step(
            self, actions: dict[str, tuple[int, Optional[int]]]
        ) -> tuple[
            dict[str, dict[str, Any]],
            dict[str, float],
            dict[str, bool],
            dict[str, bool],
            dict[str, dict[str, Any]]
        ]:
        """Perform step with mal simulator and observe in parallel env"""

        malsim_actions: dict[str, list[AttackGraphNode]] = {}
        for agent_name, agent_action in actions.items():
            malsim_actions[agent_name] = []

            if agent_action[0] and agent_action[1] is not None:
                # If agent wants to act, convert index to node
                malsim_actions[agent_name].append(
                    self.index_to_node(agent_action[1])
                )

        states = self.sim.step(malsim_actions)

        all_actioned = set(
            n for state in states.values()
            for n in state.step_performed_nodes
        )
        disabled_nodes = next(iter(states.values())).step_unviable_nodes

        self._update_agent_infos() # Update action masks
        self._update_observations(all_actioned, set(disabled_nodes))

        observations = self._agent_observations
        rewards = {}
        terminations = {}
        truncations = {}
        infos = self._agent_infos

        for agent in self.sim.agent_states.values():
            rewards[agent.name] = self.sim.agent_reward(agent.name)
            terminations[agent.name] = self.sim.agent_is_terminated(agent.name)
            truncations[agent.name] = self.sim.done()

        return observations, rewards, terminations, truncations, infos
