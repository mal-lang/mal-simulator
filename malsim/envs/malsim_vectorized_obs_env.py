""""MalSimVectorizedObsEnv:
    - Abide to the ParallelEnv interface
    - Build serialized observations from the MalSimulator state
    - step() assumes that actions are given as AttackGraphNodes
    - Used by AttackerEnv/DefenderEnv to be able to 
"""

from __future__ import annotations

import functools
import logging
import sys

import numpy as np
from gymnasium.spaces import MultiDiscrete, Box, Dict
from pettingzoo import ParallelEnv
from maltoolbox.attackgraph import AttackGraphNode

from ..mal_simulator import (
    MalSimulator,
    AgentType,
    MalSimAgentStateView,
    MalSimAttackerState,
    MalSimDefenderState
)

from .base_classes import MalSimEnv

ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)

# First the logging methods:

def format_full_observation(sim, observation):
    """
    Return a formatted string of the entire observation. This includes
    sections that will not change over time, these define the structure of
    the attack graph.
    """
    obs_str = '\nAttack Graph Steps\n'

    str_format = "{:<5} {:<80} {:<6} {:<5} {:<5} {:<30} {:<8} {:<}\n"
    header_entry = [
        "Entry", "Name", "Is_Obs", "State",
        "RTTC", "Asset Type(Index)", "Asset Id", "Step"
    ]
    entries = []
    for entry in range(0, len(observation["observed_state"])):
        asset_type_index = observation["asset_type"][entry]
        asset_type_str = sim._index_to_asset_type[asset_type_index ] + \
            '(' + str(asset_type_index) + ')'
        entries.append(
            [
                entry,
                sim._index_to_full_name[entry],
                observation["is_observable"][entry],
                observation["observed_state"][entry],
                observation["remaining_ttc"][entry],
                asset_type_str,
                observation["asset_id"][entry],
                observation["step_name"][entry],
            ]
        )
    obs_str += format_table(
        str_format, header_entry, entries, reprint_header = 30
    )

    obs_str += "\nAttack Graph Edges:\n"
    for edge in observation["attack_graph_edges"]:
        obs_str += str(edge) + "\n"

    obs_str += "\nInstance Model Assets:\n"
    str_format = "{:<5} {:<5} {:<}\n"
    header_entry = [
        "Entry", "Id", "Type(Index)"]
    entries = []
    for entry in range(0, len(observation["model_asset_id"])):
        asset_type_str = sim._index_to_asset_type[
            observation["model_asset_type"][entry]] + \
                '(' + str(observation["model_asset_type"][entry]) + ')'
        entries.append(
            [
                entry,
                observation["model_asset_id"][entry],
                asset_type_str
            ]
        )
    obs_str += format_table(
        str_format, header_entry, entries, reprint_header = 30
    )

    obs_str += "\nInstance Model Edges:\n"
    str_format = "{:<5} {:<40} {:<40} {:<}\n"
    header_entry = [
        "Entry",
        "Left Asset(Id/Index)",
        "Right Asset(Id/Index)",
        "Type(Index)"
    ]
    entries = []
    for entry in range(0, len(observation["model_edges_ids"])):
        assoc_type_str = sim._index_to_model_assoc_type[
            observation["model_edges_type"][entry]] + \
                '(' + str(observation["model_edges_type"][entry]) + ')'
        left_asset_index = int(observation["model_edges_ids"][entry][0])
        right_asset_index = int(observation["model_edges_ids"][entry][1])
        left_asset_id = sim._index_to_model_asset_id[left_asset_index]
        right_asset_id = sim._index_to_model_asset_id[right_asset_index]
        left_asset_str = \
            sim.model.get_asset_by_id(left_asset_id).name + \
            '(' + str(left_asset_id) + '/' + str(left_asset_index) + ')'
        right_asset_str = \
            sim.model.get_asset_by_id(right_asset_id).name + \
            '(' + str(right_asset_id) + '/' + str(right_asset_index) + ')'
        entries.append(
            [
                entry,
                left_asset_str,
                right_asset_str,
                assoc_type_str
            ]
        )
    obs_str += format_table(
        str_format, header_entry, entries, reprint_header = 30
    )

    return obs_str

def format_obs_var_sec(
        sim,
        observation,
        included_values = [-1, 0, 1]
    ):
    """
    Return a formatted string of the sections of the observation that can
    vary over time.

    Arguments:
    observation     - the observation to format
    included_values - the values to list, any values not present in the
                        list will be filtered out
    """

    str_format = "{:>5} {:>80} {:<5} {:<5} {:<}\n"
    header_entry = ["Id", "Name", "State", "RTTC", "Entry"]
    entries = []
    for entry in range(0, len(observation["observed_state"])):
        if observation["is_observable"][entry] and \
            observation["observed_state"][entry] in included_values:
            entries.append(
                [
                    sim._index_to_id[entry],
                    sim._index_to_full_name[entry],
                    observation["observed_state"][entry],
                    observation["remaining_ttc"][entry],
                    entry
                ]
            )

    obs_str = format_table(
        str_format, header_entry, entries, reprint_header = 30
    )

    return obs_str

def format_info(sim, info):
    can_act = "Yes" if info["action_mask"][0][1] > 0 else "No"
    agent_info_str = f"Can act? {can_act}\n"
    for entry in range(0, len(info["action_mask"][1])):
        if info["action_mask"][1][entry] == 1:
            agent_info_str += f"{sim._index_to_id[entry]} " \
                f"{sim._index_to_full_name[entry]}\n"
    return agent_info_str


def log_mapping_tables(logger, sim):
    """Log all mapping tables in MalSimulator"""

    str_format = "{:<5} {:<15} {:<}\n"
    table = "\n"
    header_entry = ["Index", "Attack Step Id", "Attack Step Full Name"]
    entries = []
    for entry in sim._index_to_id:
        entries.append(
            [
                sim._id_to_index[entry],
                entry,
                sim._index_to_full_name[sim._id_to_index[entry]]
            ]
        )
    table += format_table(
        str_format,
        header_entry,
        entries,
        reprint_header = 30
    )
    logger.debug(table)

    str_format = "{:<5} {:<}\n"
    table = "\n"
    header_entry = ["Index", "Asset Id"]
    entries = []
    for entry in sim._model_asset_id_to_index:
        entries.append(
            [
                sim._model_asset_id_to_index[entry],
                entry
            ]
        )
    table += format_table(
        str_format,
        header_entry,
        entries,
        reprint_header = 30
    )
    logger.debug(table)

    str_format = "{:<5} {:<}\n"
    table = "\n"
    header_entry = ["Index", "Asset Type"]
    entries = []
    for entry in sim._asset_type_to_index:
        entries.append(
            [
                sim._asset_type_to_index[entry],
                entry
            ]
        )
    table += format_table(
        str_format,
        header_entry,
        entries,
        reprint_header = 30
    )
    logger.debug(table)

    str_format = "{:<5} {:<}\n"
    table = "\n"
    header_entry = ["Index", "Attack Step Name"]
    entries = []
    for entry in sim._index_to_step_name:
        entries.append([sim._step_name_to_index[entry], entry])
    table += format_table(
        str_format,
        header_entry,
        entries,
        reprint_header = 30
    )
    logger.debug(table)

    str_format = "{:<5} {:<}\n"
    table = "\n"
    header_entry = ["Index", "Association Type"]
    entries = []
    for entry in sim._index_to_model_assoc_type:
        entries.append([sim._model_assoc_type_to_index[entry], entry])
    table += format_table(
        str_format,
        header_entry,
        entries,
        reprint_header = 30
    )
    logger.debug(table)


def format_table(
        entry_format: str,
        header_entry: list[str],
        entries: list[list[str]],
        reprint_header: int = 0
    ) -> str:
    """
    Format a table according to the parameters specified.

    Arguments:
    entry_format    - The string format for the table
    reprint_header  - How many rows apart to reprint the header. If 0 the
                      header will not be reprinted.
    header_entry    - The entry representing the header of the table
    entries         - The list of entries to format

    Return:
    The formatted table.
    """

    formatted_str = ''
    header = entry_format.format(*header_entry)
    formatted_str += header
    for entry_nr, entry in zip(range(0, len(entries)), entries):
        formatted_str += entry_format.format(*entry)
        if (reprint_header != 0) and ((entry_nr + 1) % reprint_header == 0):
            formatted_str += header
    return formatted_str


def log_agent_state(
        logger, sim, agent, terminations, truncations, infos
    ):
    """Debug log all an agents current state"""

    agent_obs_str = format_obs_var_sec(
        sim, agent.observation, included_values = [0, 1]
    )

    logger.debug(
        'Observation for agent "%s":\n%s', agent.name, agent_obs_str)
    logger.debug(
        'Rewards for agent "%s": %d', agent.name, agent.reward)
    logger.debug(
        'Termination for agent "%s": %s',
        agent.name, terminations[agent.name])
    logger.debug(
        'Truncation for agent "%s": %s',
        agent.name, str(truncations[agent.name]))
    agent_info_str = format_info(sim, infos[agent.name])
    logger.debug(
        'Info for agent "%s":\n%s', agent.name, agent_info_str)


# Now the actual class:

class MalSimVectorizedObsEnv(ParallelEnv, MalSimEnv):
    """
    Environment that runs simulation between agents.
    Builds serialized observations.
    Implements the ParallelEnv.
    """

    def __init__(
            self,
            sim: MalSimulator
        ):

        super().__init__(sim)

        # Useful instead of having to fetch .sim.attack_graph
        self.attack_graph = sim.attack_graph

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
        self._index_to_step_name = list(unique_step_type_names)

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
        self._step_name_to_index = {
            n: i for i, n in enumerate(self._index_to_step_name)
        }
        self._model_asset_id_to_index = {
            asset: i for i, asset in enumerate(self._index_to_model_asset_id)
        }
        self._model_assoc_type_to_index = {
            assoc_type: i for i, assoc_type in
            enumerate(self._index_to_model_assoc_type)
        }

        if logger.isEnabledFor(logging.DEBUG):
            log_mapping_tables(logger, self)

        self._blank_observation = self._create_blank_observation()

        self._agent_observations = {}
        self._agent_infos = {}

    @property
    def agents(self):
        """Required by ParallelEnv"""
        return list(self.sim._alive_agents)

    @property
    def possible_agents(self):
        """Required by ParallelEnv"""
        return list(self.sim._agent_states.keys())

    def _create_blank_observation(self, default_obs_state=-1):
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
                         for step in self.attack_graph.nodes.values()],
            "step_name": [
                self._step_name_to_index.get(
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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                format_full_observation(self, np_obs)
            )

        return np_obs

    def create_action_mask(self, agent: MalSimAgentStateView):
        """
        Create an action mask for an agent based on its action_surface.

        Parameters:
            agent: The agent for whom the mask is created.

        Returns:
            A dictionary with the action mask for the agent.
        """

        available_actions = [0] * len(self.sim.attack_graph.nodes)
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
                attacker = agent.attacker
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
        for agent in self.sim.agent_states.values():
            self._agent_infos[agent.name] = self.create_action_mask(agent)

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
            self.sim.attack_graph.model.get_association_field_names(association)
        left_field = getattr(association, left_field_name)
        right_field = getattr(association, right_field_name)
        lang_assoc = self.sim.attack_graph.lang_graph.get_association_by_fields_and_assets(
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

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        num_actions = 2  # two actions: wait or use
        # For now, an `object` is an attack step
        num_steps = len(self.sim.attack_graph.nodes)
        return MultiDiscrete([num_actions, num_steps], dtype=np.int64)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_name: str = None):
        # For now, an `object` is an attack step
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

    def register_attacker(self, attacker_name: str, attacker_id: int):
        super().register_attacker(attacker_name, attacker_id)
        agent = self.sim.agent_states[attacker_name]
        self._init_agent(agent)

    def register_defender(self, defender_name: str):
        super().register_defender(defender_name)
        agent = self.sim.agent_states[defender_name]
        self._init_agent(agent)

    def _init_agent(self, agent: MalSimAgentStateView):
        # Fill dicts with env specific agent obs/infos
        self._agent_observations[agent.name] = \
            self._create_blank_observation()

        self._agent_infos[agent.name] = \
            self.create_action_mask(agent)

    def _update_attacker_obs(
            self,
            compromised_nodes,
            disabled_nodes,
            attacker_agent: MalSimAttackerState
        ):
        """Update the observation of the serialized obs attacker"""

        def _enable_node(
                node: AttackGraphNode, agent_observation: dict
            ):
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

        attacker = attacker_agent.attacker
        attacker_observation = self._agent_observations[attacker_agent.name]

        for node in compromised_nodes:
            if node.is_compromised_by(attacker):
                # Enable node
                logger.debug("Enable %s in attacker obs", node.full_name)
                _enable_node(node, attacker_observation)

        for node in disabled_nodes:
            is_entrypoint = node.extras.get('entrypoint', False)
            if node.is_compromised_by(attacker) and not is_entrypoint:
                logger.debug("Disable %s in attacker obs", node.full_name)
                # Mark attacker compromised steps that were
                # disabled by a defense as disabled in obs
                node_idx = self.node_to_index(node)
                attacker_observation['observed_state'][node_idx] = 0

    def _update_defender_obs(
            self,
            compromised_nodes: list[AttackGraphNode],
            disabled_nodes: list[AttackGraphNode],
            defender_agent: MalSimDefenderState
        ):
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
            seed: int | None = None,
            options: dict | None = None
        ) -> tuple[dict, dict]:
        """Reset simulator and return current
        observation and infos for each agent"""

        MalSimEnv.reset(self, seed, options)

        self.attack_graph = self.sim.attack_graph # new ref

        for agent in self.sim.agent_states.values():
            # Reset observation and action mask for agents
            self._agent_observations[agent.name] = \
                self._create_blank_observation()
            self._agent_infos[agent.name] = \
                self.create_action_mask(agent)

        # Enable pre-enabled nodes in observation
        attacker_entry_points = [
            n for n in self.sim.attack_graph.nodes.values()
            if n.is_compromised()
        ]
        pre_enabled_defenses = [
            n for n in self.sim.attack_graph.nodes.values()
            if n.defense_status == 1.0
        ]

        for node in attacker_entry_points:
            node.extras['entrypoint'] = True

        self._update_observations(
            attacker_entry_points + pre_enabled_defenses, []
        )

        # TODO: should we return copies instead so they are not modified externally?
        return self._agent_observations, self._agent_infos

    def _update_observations(self, compromised_nodes, disabled_nodes):
        """Update observations of all agents"""

        if not self.sim.sim_settings.uncompromise_untraversable_steps:
            disabled_nodes = []

        # TODO: Is this correct? All attackers get the same compromised_nodes?
        logger.debug("Enable:\n\t%s", [n.full_name for n in compromised_nodes])
        logger.debug("Disable:\n\t%s", [n.full_name for n in disabled_nodes])

        for agent in self.sim.agent_states.values():
            if agent.type == AgentType.ATTACKER:
                self._update_attacker_obs(
                    compromised_nodes, disabled_nodes, agent
                )
            elif agent.type == AgentType.DEFENDER:
                self._update_defender_obs(
                    compromised_nodes, disabled_nodes, agent
                )

    def step(self, actions: dict[str, tuple[int, int]]):
        """Perform step with mal simulator and observe in parallel env"""

        malsim_actions = {}
        for agent_name, agent_action in actions.items():
            malsim_actions[agent_name] = []
            if agent_action[0]:
                # If agent wants to act, convert index to node
                malsim_actions[agent_name].append(
                    self.index_to_node(agent_action[1])
                )

        states = self.sim.step(malsim_actions)

        all_actioned = [
            n
            for state in states.values()
            for n in state.step_performed_nodes
        ]
        disabled_nodes = next(iter(states.values())).step_unviable_nodes

        self._update_agent_infos() # Update action masks
        self._update_observations(all_actioned, disabled_nodes)

        observations = self._agent_observations
        rewards = {}
        terminations = {}
        truncations = {}
        infos = self._agent_infos

        for agent in self.sim.agent_states.values():
            rewards[agent.name] = agent.reward
            terminations[agent.name] = agent.terminated
            truncations[agent.name] = agent.truncated

        return observations, rewards, terminations, truncations, infos
