""""VectorizedObsMalSimulator:
    - Abide to the ParallelEnv interface
    - Build serialized observations from the MalSimulator state
    - step() assumes that actions are given as AttackGraphNodes
    - Used by AttackerEnv/DefenderEnv to be able to 
"""

from __future__ import annotations

import logging
import sys

import numpy as np
from gymnasium.spaces import MultiDiscrete, Box, Dict
from pettingzoo import ParallelEnv
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from ..sims.mal_sim_settings import MalSimulatorSettings
from ..sims.mal_simulator import MalSimulator
from ..agents.agent_base import AgentType, MalSimAgent, MalSimAttacker, MalSimDefender
from ..sims.mal_sim_logging_utils import format_full_observation, log_mapping_tables

ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)

class VectorizedObsMalSimulator(MalSimulator, ParallelEnv):
    """
    Environment that runs simulation between agents.
    Builds serialized observations.
    """

    def __init__(
            self,
            attack_graph: AttackGraph,
            prune_unviable_unnecessary: bool = True,
            sim_settings: MalSimulatorSettings = MalSimulatorSettings(),
            max_iter=ITERATIONS_LIMIT
        ):

        super().__init__(
            attack_graph,
            prune_unviable_unnecessary,
            sim_settings,
            max_iter
        )

        # List mapping from node/asset index to id/name/type
        self._index_to_id = [n.id for n in self.attack_graph.nodes]
        self._index_to_full_name = \
            [n.full_name for n in self.attack_graph.nodes]
        self._index_to_asset_type = \
            [n.name for n in self.attack_graph.lang_graph.assets]
        self._index_to_step_name = \
            [n.asset.name + ":" + n.name
             for n in self.attack_graph.lang_graph.attack_steps]
        self._index_to_model_asset_id = \
            [int(asset.id) for asset in self.attack_graph.model.assets]
        self._index_to_model_assoc_type = [
            assoc.name + '_' + \
            assoc.left_field.asset.name + '_' + \
            assoc.right_field.asset.name \
            for assoc in self.attack_graph.lang_graph.associations
        ]

        # Lookup dicts attribute to index
        self._id_to_index = {
            n: i for i, n in enumerate(self._index_to_id)}
        self._asset_type_to_index = {
            n: i for i, n in enumerate(self._index_to_asset_type)}
        self._step_name_to_index = {
            n: i for i, n in enumerate(self._index_to_step_name)
        }
        self._model_asset_id_to_index = {
            asset: i for i, asset in enumerate(self._index_to_model_asset_id)
        }
        self._model_assoc_type_to_index = {
            assoc_type: i for i, assoc_type in \
                enumerate(self._index_to_model_assoc_type)
        }

        if logger.isEnabledFor(logging.DEBUG):
            log_mapping_tables(logger, self)

        self._blank_observation = self._create_blank_observation()
        self.reset()

    def _create_blank_observation(self, default_obs_state=-1):
        """Create the initial observation"""
        # For now, an `object` is an attack step
        num_steps = len(self.attack_graph.nodes)

        observation = {
            # If no observability set for node, assume observable.
            "is_observable": [step.extras.get('observable', 1)
                              for step in self.attack_graph.nodes],
            # Same goes for actionable.
            "is_actionable": [step.extras.get('actionable', 1)
                              for step in self.attack_graph.nodes],
            "observed_state": num_steps * [default_obs_state],
            "remaining_ttc": num_steps * [0],
            "asset_type": [self._asset_type_to_index[step.asset.type]
                           for step in self.attack_graph.nodes],
            "asset_id": [step.asset.id
                         for step in self.attack_graph.nodes],
            "step_name": [self._step_name_to_index[
                          str(step.asset.type + ":" + step.name)]
                          for step in self.attack_graph.nodes],
        }

        logger.debug(
            'Create blank observation with %d attack steps.', num_steps)

        # Add attack graph edges to observation
        observation["attack_graph_edges"] = [
            [self._id_to_index[attack_step.id], self._id_to_index[child.id]]
                for attack_step in self.attack_graph.nodes
                    for child in attack_step.children
        ]

        # Add reverse attack graph edges for defense steps (required by some
        # defender agent logic)
        for attack_step in self.attack_graph.nodes:
            if attack_step.type == "defense":
                for child in attack_step.children:
                    observation["attack_graph_edges"].append(
                        [self._id_to_index[child.id],
                            self._id_to_index[attack_step.id]]
                    )

        # Add instance model assets
        observation["model_asset_id"] = []
        observation["model_asset_type"] = []
        observation["model_edges_ids"] = []
        observation["model_edges_type"] = []

        for asset in self.attack_graph.model.assets:
            observation["model_asset_id"].append(asset.id)
            observation["model_asset_type"].append(
                self._asset_type_to_index[asset.type])

        for assoc in self.attack_graph.model.associations:
            left_field_name, right_field_name = \
                self.attack_graph.model.get_association_field_names(assoc)
            left_field = getattr(assoc, left_field_name)
            right_field = getattr(assoc, right_field_name)
            for left_asset in left_field:
                for right_asset in right_field:
                    observation["model_edges_ids"].append(
                        [
                            self._model_asset_id_to_index[left_asset.id],
                            self._model_asset_id_to_index[right_asset.id]
                        ]
                    )
                    observation["model_edges_type"].append(
                        self._model_assoc_type_to_index[
                            self._get_association_full_name(assoc)])

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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                format_full_observation(self, np_obs)
            )

        return np_obs

    def create_action_mask(self, agent: MalSimAgent):
        """
        Create an action mask for an agent based on its action_surface.

        Parameters:
            agent: The agent for whom the mask is created.

        Returns:
            A dictionary with the action mask for the agent.
        """

        available_actions = [0] * len(self.attack_graph.nodes)
        can_wait = 1 if agent.type == AgentType.DEFENDER else 0
        can_act = 0

        for node in agent.action_surface:

            if agent.type == AgentType.DEFENDER:
                # Defender can act on its whole action surface
                index = self._id_to_index[node.id]
                available_actions[index] = 1
                can_act = 1

            if agent.type == AgentType.ATTACKER:
                # Attacker can only act on nodes that are not compromised
                attacker = \
                    self.attack_graph.get_attacker_by_id(agent.attacker_id)
                if not node.is_compromised_by(attacker):
                    index = self._id_to_index[node.id]
                    available_actions[index] = 1
                    can_act = 1

        return {
            'action_mask': (
                np.array([can_wait, can_act], dtype=np.int8),
                np.array(available_actions, dtype=np.int8)
            )
        }

    def _update_agent_infos(self):
        for agent in self.agents_dict.values():
            agent.info = self.create_action_mask(agent)

    def _get_association_full_name(self, association) -> str:
        """Get association full name

        TODO: Remove this method once the language graph integration is
        complete in the mal-toolbox because the language graph associations
        will use their full names for the name property

        Arguments:
        association     - the association whose full name will be returned

        Return:
        A string containing the association name and the name of each of the
        two asset types for the left and right fields separated by
        underscores.
        """

        assoc_name = association.__class__.__name__
        if '_' in assoc_name:
            # TODO: Not actually a to-do, but just an extra clarification that
            # this is an ugly hack that will work for now until we get the
            # unique association names. Right now some associations already
            # use the asset types as part of their name if there are multiple
            # associations with the same name.
            return assoc_name

        left_field_name, right_field_name = \
            self.attack_graph.model.get_association_field_names(association)
        left_field = getattr(association, left_field_name)
        right_field = getattr(association, right_field_name)
        lang_assoc = self.attack_graph.lang_graph.get_association_by_fields_and_assets(
            left_field_name,
            right_field_name,
            left_field[0].type,
            right_field[0].type
        )
        if lang_assoc is None:
            raise LookupError('Failed to find association for fields '
                '"%s" "%s" and asset types "%s" "%s"!' % (
                    left_field_name,
                    right_field_name,
                    left_field[0].type,
                    right_field[0].type
                )
            )
        assoc_full_name = lang_assoc.name + '_' + \
            lang_assoc.left_field.asset.name + '_' + \
            lang_assoc.right_field.asset.name
        return assoc_full_name

    def action_space(self, agent=None):
        num_actions = 2  # two actions: wait or use
        # For now, an `object` is an attack step
        num_steps = len(self.attack_graph.nodes)
        return MultiDiscrete([num_actions, num_steps], dtype=np.int64)

    def observation_space(self, agent_name: str = None):
        # For now, an `object` is an attack step
        num_assets = len(self.attack_graph.model.assets)
        num_steps = len(self.attack_graph.nodes)
        num_lang_asset_types = len(self.attack_graph.lang_graph.assets)
        num_lang_attack_steps = len(self.attack_graph.lang_graph.attack_steps)
        num_lang_association_types = len(self.attack_graph.lang_graph.associations)
        num_attack_graph_edges = len(self._blank_observation["attack_graph_edges"])
        num_model_edges = len(self._blank_observation["model_edges_ids"])
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
        node = self.attack_graph.get_node_by_id(node_id)
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

    def register_agent(self, agent):
        super().register_agent(agent)

        # Fill in required fields for parallel env
        agent.observation = self._create_blank_observation()
        agent.info = self.create_action_mask(agent)

    def _update_attacker_obs(
            self,
            enabled_nodes,
            disabled_nodes,
            attacker_agent: MalSimAttacker
        ):
        """Update the observation of the serialized obs attacker"""

        def _enable_node(
                node: AttackGraphNode, agent: MalSimAttacker
            ):
            """Set enabled node obs state to enabled and
            its children to disabled"""

            # Mark enabled node obs state with 1 (enabled)
            node_index = self._id_to_index[node.id]
            agent.observation['observed_state'][node_index] = 1

            # Mark unknown (-1) children node obs states with 0 (disabled)
            for child_node in node.children:
                child_index = self._id_to_index[child_node.id]
                child_obs = agent.observation['observed_state'][child_index]
                if child_obs == -1:
                    agent.observation['observed_state'][child_index] = 0

        attacker = self.attack_graph.get_attacker_by_id(
            attacker_agent.attacker_id)

        for node in enabled_nodes:
            if node.is_compromised_by(attacker):
                # Enable node
                logger.debug("Enable %s in attacker obs", node.full_name)
                _enable_node(node, attacker_agent)

        for node in disabled_nodes:
            is_entrypoint = node.extras.get('entrypoint', False)
            if node.is_compromised_by(attacker) and not is_entrypoint:
                logger.debug("Disable %s in attacker obs", node.full_name)
                # Mark attacker compromised steps that were
                # disabled by a defense as disabled in obs
                node_idx = self.node_to_index(node)
                attacker_agent.observation['observed_state'][node_idx] = 0

    def _update_defender_obs(
            self,
            enabled_nodes: list[AttackGraphNode],
            disabled_nodes: list[AttackGraphNode],
            defender_agent: MalSimDefender
        ):
        """Update the observation of the defender"""

        for node in enabled_nodes:
            logger.debug("Enable %s in defender obs", node.full_name)
            node_idx = self.node_to_index(node)
            defender_agent.observation['observed_state'][node_idx] = 1

        for node in disabled_nodes:
            is_entrypoint = node.extras.get('entrypoint', False)
            if not is_entrypoint:
                logger.debug("Disable %s in defender obs", node.full_name)
                node_idx = self.node_to_index(node)
                defender_agent.observation['observed_state'][node_idx] = 0

    def reset(
            self,
            seed: int | None = None,
            options: dict | None = None
        ) -> tuple[dict, dict]:
        """Reset simulator and return current
        observation and infos for each agent"""

        super().reset(seed, options)
        obs = {}
        infos = {}
        for agent in self.agents_dict.values():
            # Reset observation and action mask for agents
            agent.observation = self._create_blank_observation()
            agent.info = self.create_action_mask(agent)
            obs[agent.name] = agent.observation
            infos[agent.name] = agent.info

        # Enable pre-enabled nodes in observation
        attacker_entry_points = [
            n for n in self.attack_graph.nodes if n.is_compromised()]
        pre_enabled_defenses = [
            n for n in self.attack_graph.nodes if n.defense_status == 1.0]

        for node in attacker_entry_points:
            node.extras['entrypoint'] = True

        self._update_observations(
            attacker_entry_points + pre_enabled_defenses, []
        )

        return obs, infos

    def _update_observations(self, enabled_nodes, disabled_nodes):
        """Update observations of all agents"""

        if not self.sim_settings.uncompromise_untraversable_steps:
            disabled_nodes = []

        logger.debug("Enable:\n\t%s", [n.full_name for n in enabled_nodes])
        logger.debug("Disable:\n\t%s", [n.full_name for n in disabled_nodes])

        for agent in self.agents_dict.values():
            if agent.type == AgentType.ATTACKER:
                self._update_attacker_obs(
                    enabled_nodes, disabled_nodes, agent
                )
            elif agent.type == AgentType.DEFENDER:
                self._update_defender_obs(
                    enabled_nodes, disabled_nodes, agent
                )

    def step(self, actions: dict[str, list[AttackGraphNode]]):
        """Perform step with mal simulator and observe in parallel env"""

        enabled_nodes, disabled_nodes = super().step(actions)
        self._update_agent_infos()
        self._update_observations(enabled_nodes, disabled_nodes)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent_name, agent in self.agents_dict.items():
            observations[agent_name] = agent.observation
            rewards[agent_name] = agent.reward
            terminations[agent_name] = agent.terminated
            truncations[agent_name] = agent.truncated
            infos[agent_name] = agent.info

        return observations, rewards, terminations, truncations, infos
