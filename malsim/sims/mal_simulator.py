from __future__ import annotations

import sys
import copy
import logging
import functools
from typing import Optional, TYPE_CHECKING
import numpy as np

from gymnasium.spaces import MultiDiscrete, Box, Dict
from pettingzoo import ParallelEnv

from maltoolbox import neo4j_configs
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode, Attacker
from maltoolbox.attackgraph.analyzers import apriori
from maltoolbox.attackgraph import query
from maltoolbox.ingestors import neo4j

from .mal_simulator_settings import MalSimulatorSettings

if TYPE_CHECKING:
    from maltoolbox.language import LanguageGraph
    from maltoolbox.model import Model

ITERATIONS_LIMIT = int(1e9)

logger = logging.getLogger(__name__)

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

class MalSimulator(ParallelEnv):
    def __init__(
        self,
        lang_graph: LanguageGraph,
        model: Model,
        attack_graph: AttackGraph,
        max_iter=ITERATIONS_LIMIT,
        prune_unviable_unnecessary: bool = True,
        sim_settings: MalSimulatorSettings = MalSimulatorSettings(),
        **kwargs,
    ):
        """
        Args:
            lang_graph                  -   The language graph to use
            model                       -   The model to use
            attack_graph                -   The attack graph to use
            max_iter                    -   Max iterations in simulation
            prune_unviable_unnecessary  -   Prunes graph if set to true
            sim_settings                -   Settings for simulator
        """
        super().__init__()
        logger.info("Create Mal Simulator.")
        self.lang_graph = lang_graph
        self.model = model

        apriori.calculate_viability_and_necessity(attack_graph)
        if prune_unviable_unnecessary:
            apriori.prune_unviable_and_unnecessary_nodes(attack_graph)

        self.attack_graph = attack_graph
        self.sim_settings = sim_settings
        self.max_iter = max_iter

        self.attack_graph_backup = copy.deepcopy(self.attack_graph)

        self.possible_agents = []
        self.agents = []
        self.agents_dict = {}

        self.initialize(self.max_iter)

    def __call__(self):
        return self

    def create_blank_observation(self, default_obs_state=-1):
        # For now, an `object` is an attack step
        num_steps = len(self.attack_graph.nodes)

        observation = {
            # If no observability set for node, assume observable.
            "is_observable": [step.extras.get('observable', 1)
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

        for asset in self.model.assets:
            observation["model_asset_id"].append(asset.id)
            observation["model_asset_type"].append(
                self._asset_type_to_index[asset.type])

        for assoc in self.model.associations:
            left_field_name, right_field_name = \
                self.model.get_association_field_names(assoc)
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

    def format_full_observation(self, observation):
        """
        Return a formatted string of the entire observation. This includes
        sections that will not change over time, these define the structure of
        the attack graph.
        """
        obs_str = '\nAttack Graph Steps\n'

        str_format = "{:<5} {:<80} {:<6} {:<5} {:<5} {:<30} {:<8} {:<}\n"
        header_entry = [
            "Entry", "Name", "Is_Obs", "State", "RTTC", "Asset Type(Index)", "Asset Id", "Step"]
        entries = []
        for entry in range(0, len(observation["observed_state"])):
            asset_type_index = observation["asset_type"][entry]
            asset_type_str = self._index_to_asset_type[asset_type_index ] + \
                '(' + str(asset_type_index) + ')'
            entries.append(
                [
                    entry,
                    self._index_to_full_name[entry],
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
            asset_type_str = self._index_to_asset_type[
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
            assoc_type_str = self._index_to_model_assoc_type[
                observation["model_edges_type"][entry]] + \
                    '(' + str(observation["model_edges_type"][entry]) + ')'
            left_asset_index = int(observation["model_edges_ids"][entry][0])
            right_asset_index = int(observation["model_edges_ids"][entry][1])
            left_asset_id = self._index_to_model_asset_id[left_asset_index]
            right_asset_id = self._index_to_model_asset_id[right_asset_index]
            left_asset_str = \
                self.model.get_asset_by_id(left_asset_id).name + \
                '(' + str(left_asset_id) + '/' + str(left_asset_index) + ')'
            right_asset_str = \
                self.model.get_asset_by_id(right_asset_id).name + \
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

    def format_obs_var_sec(self,
        observation,
        included_values = [-1, 0, 1]):
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
                        self._index_to_id[entry],
                        self._index_to_full_name[entry],
                        observation["observed_state"][entry],
                        observation["remaining_ttc"][entry],
                        entry
                    ]
                )

        obs_str = format_table(
            str_format, header_entry, entries, reprint_header = 30
        )

        return obs_str

    def _format_info(self, info):
        can_act = "Yes" if info["action_mask"][0][1] > 0 else "No"
        agent_info_str = f"Can act? {can_act}\n"
        for entry in range(0, len(info["action_mask"][1])):
            if info["action_mask"][1][entry] == 1:
                agent_info_str += f"{self._index_to_id[entry]} " \
                    f"{self._index_to_full_name[entry]}\n"
        return agent_info_str

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None):
        # For now, an `object` is an attack step
        num_assets = len(self.model.assets)
        num_steps = len(self.attack_graph.nodes)
        num_lang_asset_types = len(self.lang_graph.assets)
        num_lang_attack_steps = len(self.lang_graph.attack_steps)
        num_lang_association_types = len(self.lang_graph.associations)
        num_attack_graph_edges = len(
            self._blank_observation["attack_graph_edges"])
        num_model_edges = len(
            self._blank_observation["model_edges_ids"])
        return Dict(
            {
                "is_observable": Box(
                    0, 1, shape=(num_steps,), dtype=np.int8
                ),  #  0 for unobservable,
                #  1 for observable
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

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        num_actions = 2  # two actions: wait or use
        # For now, an `object` is an attack step
        num_steps = len(self.attack_graph.nodes)
        return MultiDiscrete([num_actions, num_steps], dtype=np.int64)

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None
        ):
        logger.info("Resetting simulator.")
        self.attack_graph = copy.deepcopy(self.attack_graph_backup)
        return self.initialize(self.max_iter)

    def log_mapping_tables(self):
        """Log all mapping tables in MalSimulator"""

        str_format = "{:<5} {:<15} {:<}\n"
        table = "\n"
        header_entry = ["Index", "Attack Step Id", "Attack Step Full Name"]
        entries = []
        for entry in self._index_to_id:
            entries.append(
                [
                    self._id_to_index[entry],
                    entry,
                    self._index_to_full_name[self._id_to_index[entry]]
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
        for entry in self._model_asset_id_to_index:
            entries.append(
                [
                    self._model_asset_id_to_index[entry],
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
        for entry in self._asset_type_to_index:
            entries.append(
                [
                    self._asset_type_to_index[entry],
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
        for entry in self._index_to_step_name:
            entries.append([self._step_name_to_index[entry], entry])
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
        for entry in self._index_to_model_assoc_type:
            entries.append([self._model_assoc_type_to_index[entry], entry])
        table += format_table(
            str_format,
            header_entry,
            entries,
            reprint_header = 30
        )
        logger.debug(table)


    def _create_mapping_tables(self):
        """Create mapping tables"""
        logger.debug("Creating and listing mapping tables.")

        # Lookup lists index to attribute
        self._index_to_id = [n.id for n in self.attack_graph.nodes]
        self._index_to_full_name = [n.full_name
                                    for n in self.attack_graph.nodes]
        self._index_to_asset_type = [n.name for n in self.lang_graph.assets]
        self._index_to_step_name = [n.asset.name + ":" + n.name
                                    for n in self.lang_graph.attack_steps]
        self._index_to_model_asset_id = [int(asset.id) for asset in \
            self.attack_graph.model.assets]
        self._index_to_model_assoc_type = [assoc.name + '_' + \
            assoc.left_field.asset.name + '_' + \
            assoc.right_field.asset.name \
                for assoc in self.lang_graph.associations]

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
            self.model.get_association_field_names(association)
        left_field = getattr(association, left_field_name)
        right_field = getattr(association, right_field_name)
        lang_assoc = self.lang_graph.get_association_by_fields_and_assets(
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


    def _initialize_agents(self) -> dict[str, list[int]]:
        """Initialize agent rewards, observations, and action surfaces

        Return:
        - An action dictionary mapping agent to initial actions
          (attacker entry points and pre-activated defenses)
        """
        # Initialize list of agent
        self.agents = copy.deepcopy(self.possible_agents)

        # Will contain initally enabled steps
        initial_actions = {}

        for agent in self.agents:
            # Initialize rewards
            self.agents_dict[agent]["rewards"] = 0
            agent_type = self.agents_dict[agent]["type"]
            initial_actions[agent] = []

            if agent_type == "attacker":
                attacker_id = self.agents_dict[agent]["attacker"]
                attacker = self.attack_graph.attackers[attacker_id]
                assert attacker, f"No attacker at index {attacker_id}"

                # Initialize observations and action surfaces
                self.agents_dict[agent]["observation"] = \
                    self.create_blank_observation()
                self.agents_dict[agent]["action_surface"] = \
                    query.get_attack_surface(attacker)

                # Initial actions for attacker are its entrypoints
                for entry_point in attacker.entry_points:
                    initial_actions[agent].append(
                        self._id_to_index[entry_point.id])
                    entry_point.extras['entrypoint'] = True

            elif agent_type == "defender":
                # Initialize observations and action surfaces
                self.agents_dict[agent]["observation"] = \
                    self.create_blank_observation(default_obs_state = 0)
                self.agents_dict[agent]["action_surface"] = \
                    query.get_defense_surface(self.attack_graph)

                # Initial actions for defender are all pre-enabled defenses
                initial_actions[agent] = [self._id_to_index[node.id]
                                          for node in self.attack_graph.nodes
                                          if node.is_enabled_defense()]

            else:
                self.agents_dict[agent]["action_surface"] = []

        return initial_actions

    def initialize(self, max_iter=ITERATIONS_LIMIT):
        """Create mapping tables, register agents, and initialize their
        observations, action surfaces, and rewards.

        Return initial observations and infos.
        """

        logger.info("Initializing MAL ParralelEnv Simulator.")
        self._create_mapping_tables()

        if logger.isEnabledFor(logging.DEBUG):
            self.log_mapping_tables()

        self.max_iter = max_iter
        self.cur_iter = 0

        logger.debug("Creating and listing blank observation space.")
        self._blank_observation = self.create_blank_observation()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                self.format_full_observation(self._blank_observation)
            )

        # Initialize agents and record the entry point actions
        initial_actions = self._initialize_agents()

        observations, _, _, _, infos = (
            self._observe_and_reward(initial_actions, []))

        return observations, infos

    def register_attacker(self, agent_name, attacker: int):
        logger.info(
            'Register attacker "%s" agent with '
            "attacker index %d.", agent_name, attacker
        )
        assert agent_name not in self.agents_dict, \
                f"Duplicate attacker agent named {agent_name} not allowed"

        self.possible_agents.append(agent_name)
        self.agents_dict[agent_name] = {
            "type": "attacker",
            "attacker": attacker,
            "observation": {},
            "action_surface": [],
            "rewards": 0
        }

    def register_defender(self, agent_name):
        """Add defender agent to the simulator

        Defenders are run first so that the defenses prevent attackers
        appropriately in case any attackers select attack steps that the
        defenders safeguards against during the same step.
        """
        logger.info('Register defender "%s" agent.', agent_name)
        assert agent_name not in self.agents_dict, \
                f"Duplicate defender agent named {agent_name} not allowed"

        # Add defenders at the front of the list to make sure they have
        # priority.
        self.possible_agents.insert(0, agent_name)
        self.agents_dict[agent_name] = {
            "type": "defender",
            "observation": {},
            "action_surface": [],
            "rewards": 0
        }

    def get_attacker_agents(self) -> dict:
        """Return agents dictionaries of attacker agents"""
        return {k: v for k, v in self.agents_dict.items()
                if v['type'] == "attacker"}

    def get_defender_agents(self) -> dict:
        """Return agents dictionaries of defender agents"""
        return {k: v for k, v in self.agents_dict.items()
                if v['type'] == "defender"}

    def state(self):
        # Should return a state for all agents
        return NotImplementedError

    def _attacker_step(self, agent, attack_step):
        actions = []
        attacker_index = self.agents_dict[agent]["attacker"]
        attacker = self.attack_graph.attackers[attacker_index]
        attack_step_node = self.attack_graph.get_node_by_id(
            self._index_to_id[attack_step])

        logger.info(
            'Attacker agent "%s" stepping through "%s"(%d).',
            agent,
            attack_step_node.full_name,
            attack_step_node.id
        )
        if query.is_node_traversable_by_attacker(attack_step_node, attacker):
            if not attack_step_node.is_compromised_by(attacker):
                logger.debug(
                    'Attacker agent "%s" has compromised "%s"(%d).',
                    agent,
                    attack_step_node.full_name,
                    attack_step_node.id
                )
                attacker.compromise(attack_step_node)
                self.agents_dict[agent]["action_surface"] = \
                    query.update_attack_surface_add_nodes(
                        attacker,
                        self.agents_dict[agent]["action_surface"],
                        [attack_step_node]
                    )
            actions.append(attack_step)
        else:
            logger.warning(
                'Attacker agent "%s" tried to compromise untraversable '
                'attack step"%s"(%d).',
                agent,
                attack_step_node.full_name,
                attack_step_node.id
            )
        return actions

    def update_viability(
            self,
            node: AttackGraphNode,
            unviable_attack_steps: list[AttackGraphNode] = None
        ) -> list[AttackGraphNode]:
        """
        Update the viability of the node in the graph and return any
        attack steps that are no longer viable.
        Propagate this recursively via children as long as changes occur.

        Arguments:
        node                    - the node to propagate updates from
        unviable_attack_steps   - a list of the attack steps that have been
                                  made unviable by a defense enabled in the
                                  current step
        """

        unviable_attack_steps = [] if unviable_attack_steps is None \
            else unviable_attack_steps
        logger.debug(
            'Update viability for node "%s"(%d)',
            node.full_name,
            node.id
        )
        assert not node.is_viable, ("update_viability should not be called"
                                   f" on viable node {node.full_name}")

        if node.extras.get('entrypoint'):
            # Never make entrypoint unviable, and do not
            # propagate its viability further
            node.is_viable = True
            return unviable_attack_steps

        if node.type in ('and', 'or'):
            unviable_attack_steps.append(node)

        for child in node.children:
            original_value = child.is_viable
            if child.type == 'or':
                child.is_viable = False
                for parent in child.parents:
                    child.is_viable = child.is_viable or parent.is_viable
            if child.type == 'and':
                child.is_viable = False

            if child.is_viable != original_value:
                self.update_viability(child, unviable_attack_steps)

        return unviable_attack_steps

    def _defender_step(
            self, agent, defense_step_index
        ) -> tuple[list[int], list[AttackGraphNode]]:

        actions = []
        defense_step_node = self.attack_graph.get_node_by_id(
            self._index_to_id[defense_step_index]
        )
        logger.info(
            'Defender agent "%s" stepping through "%s"(%d).',
            agent,
            defense_step_node.full_name,
            defense_step_node.id
        )
        if defense_step_node not in self.agents_dict[agent]["action_surface"]:
            logger.warning(
                'Defender agent "%s" tried to step through "%s"(%d).'
                'which is not part of its defense surface. Defender '
                'step will skip',
                agent,
                defense_step_node.full_name,
                defense_step_node.id
            )
            return actions, []

        defense_step_node.defense_status = 1.0
        defense_step_node.is_viable = False
        prevented_attack_steps = self.update_viability(
            defense_step_node
        )
        actions.append(defense_step_index)

        # Remove defense from all defender agents' action surfaces since it is
        # already enabled. And remove all of the prevented attack steps from
        # the attackers' action surfaces.
        for agent_el in self.agents:
            if self.agents_dict[agent_el]["type"] == "defender":
                try:
                    self.agents_dict[agent_el]["action_surface"].\
                        remove(defense_step_node)
                except ValueError:
                    # Optimization: the defender is told to remove
                    # the node from its defense surface even if it
                    # may have not been present to save one extra
                    # lookup.
                    pass
            elif self.agents_dict[agent_el]["type"] == "attacker":
                for attack_step in prevented_attack_steps:
                    try:
                        # Node is no longer part of attacker action surface
                        self.agents_dict[agent_el]\
                            ["action_surface"].remove(attack_step)
                    except ValueError:
                        # Optimization: the attacker is told to remove
                        # the node from its attack surface even if it may
                        # have not been present to save one extra lookup.
                        pass


        return actions, prevented_attack_steps

    def _observe_attacker(
            self,
            attacker_agent,
            performed_actions: dict[str, list[int]]
        ) -> None:
        """
        Update the attacker observation based on the actions performed
        in current step.

        Arguments:
        attacker_agent  - the attacker agent to fill in the observation for
        observation     - the blank observation to fill in
        """

        obs_state = self.agents_dict[attacker_agent]["observation"]\
            ["observed_state"]

        # Set obs state of reached attack steps to 1 (enabled)
        for _, actions in performed_actions.items():
            for step_index in actions:

                if step_index is None:
                    # Waiting does not affect obs
                    continue

                node_id = self._index_to_id[step_index]
                node = self.attack_graph.get_node_by_id(node_id)
                if node.type in ('or', 'and'):
                    # Attack step activated, set to 1 (enabled)
                    obs_state[step_index] = 1

                    for child in node.children:
                        # Set its children to 0 (disabled)
                        child_index = self._id_to_index[child.id]
                        if obs_state[child_index] == -1:
                            obs_state[child_index] = 0

    def _observe_defender(
            self,
            defender_agent,
            performed_actions: dict[str, list[int]]
        ):

        obs_state = self.agents_dict[defender_agent]["observation"]\
            ["observed_state"]

        if not self.sim_settings.cumulative_defender_obs:
            # Clear the state if we do not it to accumulate observations over
            # time.
            obs_state.fill(0)

        # Only show the latest steps taken
        for _, actions in performed_actions.items():
            for action in actions:
                obs_state[action] = 1

    def _observe_agents(self, performed_actions):
        """Collect agents observations"""

        for agent in self.agents:
            agent_type = self.agents_dict[agent]["type"]
            if  agent_type == "defender":
                self._observe_defender(agent, performed_actions)

            elif agent_type == "attacker":
                self._observe_attacker(agent, performed_actions)

            else:
                logger.error(
                    "Agent %s has unknown type: %s",
                    agent, self.agents_dict[agent]["type"]
                )

    def _reward_agents(self, performed_actions):
        """Update rewards from latest performed actions"""
        for agent, actions in performed_actions.items():
            agent_type = self.agents_dict[agent]["type"]

            for action in actions:
                if action is None:
                    continue

                node_id = self._index_to_id[action]
                node = self.attack_graph.get_node_by_id(node_id)
                node_reward = node.extras.get('reward', 0)

                if agent_type == "attacker":
                    # If attacker performed step, it will receive
                    # a reward and penalize all defenders
                    self.agents_dict[agent]["rewards"] += node_reward

                    for d_agent in self.get_defender_agents():
                        self.agents_dict[d_agent]["rewards"] -= node_reward
                else:
                    # If a defender performed step, it will be penalized
                    self.agents_dict[agent]["rewards"] -= node_reward


    def _collect_agents_infos(self):
        """Collect agent info, this is used to determine the possible
        actions in the next iteration step. Then fill in all of the"""

        attackers_done = True
        infos = {}
        can_wait = {
            "attacker": 0,
            "defender": 1,
        }

        for agent in self.agents:
            agent_type = self.agents_dict[agent]["type"]
            available_actions = [0] * len(self.attack_graph.nodes)
            can_act = 0

            if agent_type == "defender":
                for node in self.agents_dict[agent]["action_surface"]:
                    index = self._id_to_index[node.id]
                    available_actions[index] = 1
                    can_act = 1

            if agent_type == "attacker":
                attacker = self.attack_graph.attackers[
                    self.agents_dict[agent]["attacker"]
                ]
                for node in self.agents_dict[agent]["action_surface"]:
                    if not node.is_compromised_by(attacker):
                        index = self._id_to_index[node.id]
                        available_actions[index] = 1
                        can_act = 1
                        attackers_done = False

            infos[agent] = {
                "action_mask": (
                    np.array(
                        [can_wait[agent_type], can_act], dtype=np.int8),
                    np.array(
                        available_actions, dtype=np.int8)
                )}

        return attackers_done, infos

    def _disable_attack_steps(
            self, attack_steps_to_disable: list[AttackGraphNode]
        ):
        """Disable nodes for each attacker agent

        For each compromised attack step uncompromise the node, disable its
        observed_state, and remove the rewards.
        """

        for attacker_agent in self.get_attacker_agents():
            attacker_index = self.agents_dict[attacker_agent]["attacker"]
            attacker: Attacker = self.attack_graph.attackers[attacker_index]

            for unviable_node in attack_steps_to_disable:
                if unviable_node.is_compromised_by(attacker):

                    # Reward is no longer present for attacker
                    node_reward = unviable_node.extras.get('reward', 0)
                    self.agents_dict[attacker_agent]["rewards"] -= node_reward

                    # Reward is no longer present for defenders
                    for defender_agent in self.get_defender_agents():
                        self.agents_dict[defender_agent]["rewards"] += node_reward

                    # Uncompromise node if requested
                    attacker.undo_compromise(unviable_node)

                    # Uncompromised nodes observed state is 0 (disabled)
                    step_index = self._id_to_index[unviable_node.id]
                    agent_obs = self.agents_dict[attacker_agent]["observation"]
                    agent_obs['observed_state'][step_index] = 0


    def _observe_and_reward(
            self,
            performed_actions: dict[str, list[int]],
            prevented_attack_steps: list[AttackGraphNode]
        ):
        """Update observations and reward agents based on latest actions

        Returns 5 dicts, each mapping from agent to:
            observations, rewards, terminations, truncations, infos
        """

        terminations = {}
        truncations = {}
        infos = {}
        finished_agents = []

        if self.sim_settings.uncompromise_untraversable_steps:
            # Disable attack steps for attackers to update the
            # observations, rewards and action surface
            self._disable_attack_steps(prevented_attack_steps)

        # Fill in the agent observations, rewards,
        # infos, terminations, truncations.
        # If no attackers have any actions left
        # to take the simulation will terminate.
        self._observe_agents(performed_actions)
        self._reward_agents(performed_actions)
        attackers_done, infos = self._collect_agents_infos()

        for agent in self.agents:
            # Terminate simulation if no attackers have actions to take
            terminations[agent] = attackers_done
            if attackers_done:
                logger.debug(
                    "No attacker has actions left to perform, "
                    "terminate agent \"%s\".", agent)

            truncations[agent] = False
            if self.cur_iter >= self.max_iter:
                logger.debug(
                    "Simulation has reached the maximum number of "
                    "iterations, %d, terminate agent \"%s\".",
                    self.max_iter, agent)
                truncations[agent] = True

            if terminations[agent] or truncations[agent]:
                finished_agents.append(agent)

            if logger.isEnabledFor(logging.DEBUG):
                # Debug print agent states
                agent_obs_str = self.format_obs_var_sec(
                    self.agents_dict[agent]["observation"],
                    included_values = [0, 1])

                logger.debug(
                    'Observation for agent "%s":\n%s', agent, agent_obs_str)
                logger.debug(
                    'Rewards for agent "%s": %d', agent,
                    self.agents_dict[agent]["rewards"])
                logger.debug(
                    'Termination for agent "%s": %s',
                    agent, terminations[agent])
                logger.debug(
                    'Truncation for agent "%s": %s',
                    agent, str(truncations[agent]))

                agent_info_str = self._format_info(infos[agent])
                logger.debug(
                    'Info for agent "%s":\n%s', agent, agent_info_str)

        for agent in finished_agents:
            self.agents.remove(agent)

        observations = {agent: self.agents_dict[agent]["observation"] \
            for agent in self.agents_dict}
        rewards = {agent: self.agents_dict[agent]["rewards"] \
            for agent in self.agents_dict}
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        )

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        logger.debug(
            "Stepping through iteration %d/%d", self.cur_iter, self.max_iter)
        logger.debug("Performing actions: %s", actions)

        # Map agent to defense/attack steps performed in this step
        performed_actions = {}
        prevented_attack_steps = []

        # Peform agent actions
        for agent in self.agents:
            action = actions[agent]
            if action[0] == 0:
                # Agent wants to wait - do nothing
                continue

            action_step = action[1]

            if self.agents_dict[agent]["type"] == "attacker":
                performed_actions[agent] = \
                    self._attacker_step(agent, action_step)

            elif self.agents_dict[agent]["type"] == "defender":
                defender_actions, prevented_attack_steps = \
                    self._defender_step(agent, action_step)
                performed_actions[agent] = defender_actions

            else:
                logger.error(
                    'Agent %s has unknown type: %s',
                    agent, self.agents_dict[agent]["type"])

        observations, rewards, terminations, truncations, infos = (
            self._observe_and_reward(
                performed_actions,
                prevented_attack_steps
            )
        )

        self.cur_iter += 1

        return observations, rewards, terminations, truncations, infos

    def render(self):
        logger.debug("Ingest attack graph into Neo4J database.")
        neo4j.ingest_attack_graph(
            self.attack_graph,
            neo4j_configs["uri"],
            neo4j_configs["username"],
            neo4j_configs["password"],
            neo4j_configs["dbname"],
            delete=True,
        )
