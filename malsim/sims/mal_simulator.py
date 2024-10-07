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
from maltoolbox.attackgraph import AttackGraph
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
        }

        return np_obs

    def format_full_observation(self, observation):
        """
        Return a formatted string of the entire observation. This includes
        sections that will not change over time, these define the structure of
        the attack graph.
        """
        obs_str = '\n'

        str_format = "{:<5} {:<80} {:<6} {:<5} {:<5} {:<5} {:<5} {:<}\n"
        header_entry = [
            "Entry", "Name", "Is_Obs", "State", "RTTC", "Type", "Id", "Step"]
        entries = []
        for entry in range(0, len(observation["observed_state"])):
            entries.append(
                [
                    entry,
                    self._index_to_full_name[entry],
                    observation["is_observable"][entry],
                    observation["observed_state"][entry],
                    observation["remaining_ttc"][entry],
                    observation["asset_type"][entry],
                    observation["asset_id"][entry],
                    observation["step_name"][entry],
                ]
            )
        obs_str += format_table(
            str_format, header_entry, entries, reprint_header = 30
        )

        obs_str += "\nEdges:\n"
        for edge in observation["attack_graph_edges"]:
            obs_str += str(edge) + "\n"

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
        num_steps = len(self.attack_graph.nodes)
        num_lang_asset_types = len(self.lang_graph.assets)
        num_lang_attack_steps = len(self.lang_graph.attack_steps)
        num_attack_graph_edges = len(
            self._blank_observation["attack_graph_edges"])
        # TODO is_observable is never set. It will be filled in once the
        # observability of the attack graph is determined.
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
        header_entry = ["Index", "Asset Type"]
        entries = []
        for entry in self._index_to_asset_type:
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

    def _create_mapping_tables(self):
        """Create mapping tables"""
        logger.debug("NEW Creating and listing mapping tables.")

        # Lookup lists index to attribute
        self._index_to_id = [n.id for n in self.attack_graph.nodes]
        self._index_to_full_name = [n.full_name
                                    for n in self.attack_graph.nodes]
        self._index_to_asset_type = [n.name for n in self.lang_graph.assets]
        self._index_to_step_name = [n.asset.name + ":" + n.name
                                    for n in self.lang_graph.attack_steps]

        # Lookup dicts attribute to index
        self._id_to_index = {
            n: i for i, n in enumerate(self._index_to_id)}
        self._asset_type_to_index = {
            n: i for i, n in enumerate(self._index_to_asset_type)}
        self._step_name_to_index = {
            n: i for i, n in enumerate(self._index_to_step_name)
        }

    def _initialize_agents(self) -> dict[str, list[int]]:
        """Initialize agent rewards, observations and action_surfaces

        Updates self._agent_rewards, self._agent_observations
        and self.action_surfaces

        Return:
        - An action dictionary mapping agent to initial actions
          (attacker entry points and pre-activated defenses)
        """
        # Initialize list of agent dicts
        self.agents = copy.deepcopy(self.possible_agents)

        # Will contain initally enabled steps
        initial_actions = {}

        for agent in self.agents:
            # Initialize rewards
            self._agent_rewards[agent] = 0
            agent_type = self.agents_dict[agent]["type"]

            if agent_type == "attacker":
                attacker_id = self.agents_dict[agent]["attacker"]
                attacker = self.attack_graph.attackers[attacker_id]
                assert attacker, f"No attacker at index {attacker_id}"

                # Initialize observations and action surfaces
                self._agent_observations[agent] = \
                    self.create_blank_observation()
                self.action_surfaces[agent] = \
                    query.get_attack_surface(attacker)

                # Initial actions for attacker are its entrypoints
                initial_actions[agent] = [self._id_to_index[entry_point.id]
                                          for entry_point
                                          in attacker.entry_points]

            elif agent_type == "defender":
                # Initialize observations and action surfaces
                self._agent_observations[agent] = \
                    self.create_blank_observation(default_obs_state=0)
                self.action_surfaces[agent] = \
                    query.get_defense_surface(self.attack_graph)

                # Initial actions for defender are all pre-enabled defenses
                initial_actions[agent] = [self._id_to_index[node.id]
                                          for node in self.attack_graph.nodes
                                          if node.is_enabled_defense()]

            else:
                self.action_surfaces[agent] = []

        return initial_actions

    def initialize(self, max_iter=ITERATIONS_LIMIT):
        """Create mapping tables, register agents and
        initalize their observations + action surfaces + rewards.

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

        # Map agents to observations and rewards
        self._agent_observations = {}
        self.action_surfaces = {}
        self._agent_rewards = {}

        # Initialize agents and record the entry point actions
        initial_actions = self._initialize_agents()

        observations, _, _, _, infos = (
            self._observe_and_reward(initial_actions))

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
            "attacker": attacker
        }

    def register_defender(self, agent_name):
        """Add defender agent to the simulator"""
        logger.info('Register defender "%s" agent.', agent_name)
        assert agent_name not in self.agents_dict, \
                f"Duplicate defender agent named {agent_name} not allowed"

        self.possible_agents.insert(0, agent_name)
        self.agents_dict[agent_name] = {"type": "defender"}

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

        logger.debug(
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
                self.action_surfaces[agent] = \
                    query.update_attack_surface_add_nodes(
                        attacker,
                        self.action_surfaces[agent],
                        [attack_step_node]
                    )
            # TODO: should below line be part of the inner if-stmt?
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

    def update_viability_with_eviction(self, node):
        """
        Update the viability of the node in the graph and evict any nodes
        that are no longer viable from any attackers' action surface.
        Propagate this recursively via children as long as changes occur.

        Arguments:
        node       - the node to propagate updates from
        """
        logger.debug(
            'Update viability with eviction for node "%s"(%d)',
            node.full_name,
            node.id
        )
        if not node.is_viable:
            # This is more of a sanity check, it should never be called on
            # viable nodes.
            for agent in self.agents:
                if self.agents_dict[agent]["type"] == "attacker":
                    try:
                        self.action_surfaces[agent].remove(node)
                    except ValueError:
                        # Optimization: the attacker is told to remove
                        # the node from its attack surface even if it
                        # may have not been present to save one extra
                        # lookup.
                        pass
            for child in node.children:
                original_value = child.is_viable
                if child.type == 'or':
                    child.is_viable = False
                    for parent in child.parents:
                        child.is_viable = child.is_viable or parent.is_viable
                if child.type == 'and':
                    child.is_viable = False

                if child.is_viable != original_value:
                    self.update_viability_with_eviction(child)


    def _defender_step(self, agent, defense_step):
        actions = []
        defense_step_node = self.attack_graph.get_node_by_id(
            self._index_to_id[defense_step]
        )
        logger.info(
            'Defender agent "%s" stepping through "%s"(%d).',
            agent,
            defense_step_node.full_name,
            defense_step_node.id
        )
        if defense_step_node not in self.action_surfaces[agent]:
            logger.warning(
                'Defender agent "%s" tried to step through "%s"(%d).'
                'which is not part of its defense surface. Defender '
                'step will skip',
                agent,
                defense_step_node.full_name,
                defense_step_node.id
            )
            return actions

        defense_step_node.defense_status = 1.0
        defense_step_node.is_viable = False
        self.update_viability_with_eviction(defense_step_node)
        actions.append(defense_step)

        # Remove defense from all defender agents' action surfaces since it is
        # already enabled.
        for agent_el in self.agents:
            if self.agents_dict[agent_el]["type"] == "defender":
                try:
                    self.action_surfaces[agent_el].remove(defense_step_node)
                except ValueError:
                    # Optimization: the defender is told to remove
                    # the node from its defense surface even if it
                    # may have not been present to save one extra
                    # lookup.
                    pass

        return actions

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

        attacker_index = self.agents_dict[attacker_agent]["attacker"]
        attacker = self.attack_graph.attackers[attacker_index]
        obs_state = self._agent_observations[attacker_agent]["observed_state"]

        performed_defense_nodes = []
        performed_attack_nodes = []

        # Set obs state of reached attack steps to 1 (enabled)
        for _, actions in performed_actions.items():

            for step_index in actions:
                if step_index is None:
                    continue

                node = self.attack_graph.get_node_by_id(
                    self._index_to_id[step_index])

                if node.type in ('or', 'and'):
                    # attack step
                    obs_state[step_index] = 1
                    performed_attack_nodes.append(node)
                elif node.type == 'defense':
                    # defense step
                    performed_defense_nodes.append(node)

        # Set obs state of reached attack steps children to 0 (disabled)
        for node in performed_attack_nodes:
            for child in node.children:
                child_index = self._id_to_index[child.id]
                child_obs_state = obs_state[child_index]
                if child_obs_state == -1:
                    obs_state[child_index] = 0

        # TODO: Update viablility when defense node happens
        # Remove nodes that are defended against by latest defense
        for node in performed_defense_nodes:
            for defended_child in node.children:
                if defended_child.type not in ('or', 'and'):
                    # Child is not attack step - no need to update obs state
                    continue

                defended_child_index = self._id_to_index[defended_child.id]
                defended_child_obs_state = obs_state[defended_child_index]

                if defended_child_obs_state != 1:
                    # Child step was defended against, but obs state in order
                    continue

                if (defended_child.is_compromised_by(attacker) and
                    defended_child not in attacker.entry_points):
                    # Child step is defended against
                    # TODO: Should the obs state now be 0 (disabled)?
                    # obs_state[child_index] = 0

                    if self.sim_settings.uncompromise_untraversable_steps:
                        # Uncompromise step that has become untraversable
                        attacker.undo_compromise(defended_child)

    def _observe_defender(
            self,
            defender_agent,
            performed_actions: dict[str, list[int]]
        ):

        obs_state = self._agent_observations[defender_agent]["observed_state"]

        if self.sim_settings.cumulative_defender_obs:
            # Add the latest performed actions to the observed state
            # TODO: need to handle defense steps disabling attack steps
            pass
        else:
            # View all states as unknown
            # TODO: Should they be disabled (0) instead?
            obs_state.fill(-1)

        # Only show the latest steps taken
        for _, actions in performed_actions.items():
            for action in actions:
                obs_state[action] = 1

    def _observe_agent(self, agent, performed_actions):
        """Collect agent observations"""
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
        # TODO: should initial actions give rewards?
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
                    # a reward and penalize all defender
                    self._agent_rewards[agent] += node_reward

                    for d_agent in self.get_defender_agents():
                        self._agent_rewards[d_agent] -= node_reward
                else:
                    # TODO: defending against an attack step
                    # can it increase reward for defender?

                    # If a defender performed step, it will be penalized
                    self._agent_rewards[agent] -= node_reward


    def _collect_agent_infos(self, agent):
        """Collect agent info, this is used to determine the possible
        actions in the next iteration step. Then fill in all of the"""

        attackers_done = True

        can_wait = {
            "attacker": 0,
            "defender": 1,
        }

        agent_type = self.agents_dict[agent]["type"]
        available_actions = [0] * len(self.attack_graph.nodes)
        can_act = 0

        if agent_type == "defender":
            for node in self.action_surfaces[agent]:
                index = self._id_to_index[node.id]
                available_actions[index] = 1
                can_act = 1

        if agent_type == "attacker":
            attacker = self.attack_graph.attackers[
                self.agents_dict[agent]["attacker"]
            ]
            for node in self.action_surfaces[agent]:
                if not node.is_compromised_by(attacker):
                    index = self._id_to_index[node.id]
                    available_actions[index] = 1
                    can_act = 1
                    attackers_done = False

        return attackers_done, {
            "action_mask": (
                np.array(
                    [can_wait[agent_type], can_act], dtype=np.int8),
                np.array(
                    available_actions, dtype=np.int8)
            )
        }

    def _observe_and_reward(self, performed_actions: dict[str, list[int]]):
        """Update observations and reward agents based on latest actions

        Returns 5 dicts, each mapping from agent to:
            observations, rewards, terminations, truncations, infos
        """
        terminations = {}
        truncations = {}
        infos = {}
        finished_agents = []

        # Fill in the agent observations, rewards,
        # terminations, truncations, and infos.
        # If no attackers have any actions left
        # to take the simulation will terminate.
        attackers_done = True
        for agent in self.agents:
            self._observe_agent(agent, performed_actions)
            attackers_done, infos[agent] = \
                self._collect_agent_infos(agent)

        self._reward_agents(performed_actions)

        for agent in self.agents:
            # Terminate simulation if no attackers have actions to take
            terminations[agent] = attackers_done
            if attackers_done:
                logger.debug(
                    "No attacker has actions left to perform,"
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
                    observations[agent], included_values = [0, 1])

                logger.debug(
                    'Observation for agent "%s":\n%s', agent, agent_obs_str)
                logger.debug(
                    'Rewards for agent "%s": %d', agent, rewards[agent])
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

        return observations, rewards, terminations, truncations, infos

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

        # Will store indexes of defense/attack steps performed in this step
        performed_actions = []

        # Peform agent actions
        for agent in self.agents:
            action = actions[agent]
            if action[0] == 0:
                continue

            action_step = action[1]
            if self.agents_dict[agent]["type"] == "attacker":
                performed_actions.extend(self._attacker_step(agent, action_step))
            elif self.agents_dict[agent]["type"] == "defender":
                performed_actions.extend(self._defender_step(agent, action_step))
            else:
                logger.error(
                    'Agent %s has unknown type: %s',
                    agent, self.agents_dict[agent]["type"])

        observations, rewards, terminations, truncations, infos = (
            self._observe_and_reward(performed_actions)
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
