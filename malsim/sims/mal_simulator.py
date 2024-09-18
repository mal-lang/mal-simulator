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
from maltoolbox.model import Model
from maltoolbox.language import LanguageGraph
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.attackgraph.analyzers import apriori
from maltoolbox.attackgraph import query
from maltoolbox.ingestors import neo4j

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
        **kwargs,
    ):
        """
        Args:
            lang_graph                  -   The language graph to use
            model                       -   The model to use
            attack_graph                -   The attack graph to use
            max_iter                    -   Max iterations in simulation
            prune_unviable_unnecessary  -   Prunes graph if set to true
        """
        super().__init__()
        logger.info("Create Mal Simulator.")
        self.lang_graph = lang_graph
        self.model = model
        apriori.calculate_viability_and_necessity(attack_graph)
        if prune_unviable_unnecessary:
            apriori.prune_unviable_and_unnecessary_nodes(attack_graph)
        self.attack_graph = attack_graph
        self.max_iter = max_iter
        self.attack_graph_backup = copy.deepcopy(self.attack_graph)

        self.possible_agents = []
        self.agents = []
        self.agents_dict = {}

        self.init(self.max_iter)


    def __call__(self):
        return self


    def create_blank_observation(self):
        # For now, an `object` is an attack step
        num_steps = len(self.attack_graph.nodes)

        observation = {
            "is_observable": num_steps * [1],
            "observed_state": num_steps * [-1],
            "remaining_ttc": num_steps * [0],
            "asset_type": [self._asset_type_to_index[step.asset.type]
                for step in self.attack_graph.nodes],
            "asset_id": [step.asset.id
                for step in self.attack_graph.nodes],
            "step_name": [self._step_name_to_index[
                    str(step.asset.type + ":" + step.name)]
                for step in self.attack_graph.nodes],
        }

        logger.debug(f'Create blank observation with {num_steps} attack '
            'steps.')

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
            "Entry",
            "Name",
            "Is_Obs",
            "State",
            "RTTC",
            "Type",
            "Id",
            "Step"
        ]

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
            str_format,
            header_entry,
            entries,
            reprint_header = 30
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
        included_values - the values to list, any values not present in the list
                          will be filtered out
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
            str_format,
            header_entry,
            entries,
            reprint_header = 30
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        logger.info("Resetting simulator.")
        self.attack_graph = copy.deepcopy(self.attack_graph_backup)
        return self.init(self.max_iter)

    def init(self, max_iter=ITERATIONS_LIMIT):
        logger.info("Initializing MAL ParralelEnv Simulator.")
        logger.debug("Creating and listing mapping tables.")
        self._index_to_id = [n.id for n in self.attack_graph.nodes]
        self._index_to_full_name = [n.full_name for n in self.attack_graph.nodes]
        self._id_to_index = {n: i for i, n in enumerate(self._index_to_id)}
        str_format = "{:<5} {:<15} {:<}\n"
        table = "\n"
        header_entry = [
            "Index",
            "Attack Step Id",
            "Attack Step Full Name"
        ]
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

        self._index_to_asset_type = [n.name for n in self.lang_graph.assets]
        self._asset_type_to_index = {
            n: i for i, n in enumerate(self._index_to_asset_type)
        }
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

        self._index_to_step_name = [n.asset.name + ":" + n.name \
            for n in self.lang_graph.attack_steps]
        self._step_name_to_index = {
            n: i for i, n in enumerate(self._index_to_step_name)
        }

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

        self.max_iter = max_iter
        self.cur_iter = 0

        logger.debug("Creating and listing blank observation space.")
        self._blank_observation = self.create_blank_observation()
        logger.debug(self.format_full_observation(self._blank_observation))

        logger.info("Populate agents list with all possible agents.")
        self.agents = copy.deepcopy(self.possible_agents)
        self.action_surfaces = {}
        for agent in self.agents:
            if self.agents_dict[agent]["type"] == "attacker":
                attacker = self.attack_graph.attackers[
                    self.agents_dict[agent]["attacker"]
                ]
                self.action_surfaces[agent] = query.get_attack_surface(attacker)
            elif self.agents_dict[agent]["type"] == "defender":
                self.action_surfaces[agent] = query.get_defense_surface(
                    self.attack_graph)
            else:
                self.action_surfaces[agent] = []

        observations, rewards, terminations, truncations, infos = (
            self._observe_and_reward()
        )

        return observations, infos

    def register_attacker(self, agent_name, attacker: int):
        logger.info(
            f'Register attacker "{agent_name}" agent with '
            f"attacker index {attacker}."
        )
        self.possible_agents.append(agent_name)
        self.agents_dict[agent_name] = {"type": "attacker",
            "attacker": attacker}

    def register_defender(self, agent_name):
        # Defenders are run first so that the defenses prevent the attacker
        # appropriately in case the attacker selects an attack step that the
        # defender safeguards against in the same step.
        logger.info(f'Register defender "{agent_name}" agent.')
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
        attacker = self.attack_graph.attackers[self.agents_dict[agent]["attacker"]]
        attack_step_node = self.attack_graph.get_node_by_id(
            self._index_to_id[attack_step]
        )
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
                self.action_surfaces[agent] = \
                    query.update_attack_surface_add_nodes(
                        attacker,
                        self.action_surfaces[agent],
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


    def _observe_attacker(self, attacker_agent, observation) -> None:
        """
        Fill in the attacker observation based on the currently reached attack
        steps and their children.

        Arguments:
        attacker_agent  - the attacker agent to fill in the observation for
        observation     - the blank observation to fill in
        """

        attacker = self.attack_graph.attackers[
            self.agents_dict[attacker_agent]["attacker"]
        ]

        untraversable_nodes = []
        for node in attacker.reached_attack_steps:
            node_index = self._id_to_index[node.id]

            if query.is_node_traversable_by_attacker(node, attacker):
                # Traversable reached nodes set to 1 in observed state
                observation["observed_state"][node_index] = 1

            else:
                # The defender has activated a defense that prevents the
                # attacker from exploiting this attack step any longer.
                observation["observed_state"][node_index] = 0
                untraversable_nodes.append(node)

        # Uncompromise nodes that are not traversable.
        for node in untraversable_nodes:
                logger.debug(
                    "Remove untraversable node from attacker "
                    f'"{attacker_agent}": {node.id}'
                )
                attacker.undo_compromise(node)

        for node in attacker.reached_attack_steps:
            for child_node in node.children:
                child_node_index = self._id_to_index[child_node.id]
                observation["observed_state"][child_node_index] = 0


    def _observe_defender(self, defender_agent, observation):
        # TODO We should probably create a separate blank observation for the
        # defenders and just update that with the defense action taken so that
        # we do not have to go through the list of nodes every time. In case
        # we have multiple defenders
        for node in self.attack_graph.nodes:
            index = self._id_to_index[node.id]
            if node.is_enabled_defense():
                observation["observed_state"][index] = 1
            else:
                if node.is_compromised():
                    observation["observed_state"][index] = 1
                else:
                    observation["observed_state"][index] = 0

    def _observe_and_reward(self):
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        can_wait = {
            "attacker": 0,
            "defender": 1,
        }

        finished_agents = []
        # If no attackers have any actions left that they could take the
        # simulation will terminate.
        attackers_done = True
        # Fill in the agent observations, rewards, terminations, truncations,
        # and infos.
        for agent in self.agents:
            # Collect agent observations
            agent_observation = copy.deepcopy(self._blank_observation)
            if self.agents_dict[agent]["type"] == "defender":
                self._observe_defender(agent, agent_observation)
            elif self.agents_dict[agent]["type"] == "attacker":
                self._observe_attacker(agent, agent_observation)
            else:
                logger.error(
                    f"Agent {agent} has unknown type: "
                    f'{self.agents_dict[agent]["type"]}'
                )

            observations[agent] = agent_observation

            # Collect agent info, this is used to determine the possible actions
            # in the next iteration step. Then fill in all of the
            available_actions = [0] * len(self.attack_graph.nodes)
            can_act = 0
            agent_type = self.agents_dict[agent]["type"]
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

            infos[agent] = {
                "action_mask": (
                    np.array(
                        [can_wait[agent_type], can_act], dtype=np.int8),
                    np.array(
                        available_actions, dtype=np.int8)
                )
            }

        # First calculate the attacker rewards and attackers' total reward
        attackers_total_rewards = 0
        for agent in self.agents:
            if self.agents_dict[agent]["type"] == "attacker":
                reward = 0
                attacker = self.attack_graph.attackers[
                    self.agents_dict[agent]["attacker"]
                ]
                for node in attacker.reached_attack_steps:
                    if hasattr(node, "extras"):
                        reward += node.extras.get('reward', 0)

                attackers_total_rewards += reward
                rewards[agent] = reward

        # Then we can calculate the defender rewards which also include all of
        # the attacker rewards negated.
        for agent in self.agents:
            if self.agents_dict[agent]["type"] == "defender":
                reward = -attackers_total_rewards
                for node in query.get_enabled_defenses(self.attack_graph):
                    if hasattr(node, "extras"):
                        reward -= node.extras.get('reward', 0)
                rewards[agent] = reward

        for agent in self.agents:
            # Terminate simulation if no attackers have actions that they
            # could take.
            terminations[agent] = attackers_done
            if attackers_done:
                logger.debug(
                    "No attacker has any actions left to perform "
                    f'terminate agent "{agent}".'
                )
            truncations[agent] = False
            if self.cur_iter >= self.max_iter:
                logger.debug(
                    "Simulation has reached the maximum number of "
                    f"iterations, {self.max_iter}, terminate agent "
                    f'"{agent}".'
                )
                truncations[agent] = True

            if terminations[agent] or truncations[agent]:
                finished_agents.append(agent)

            logger.debug(
                f'Observation for agent "{agent}":\n'
                + self.format_obs_var_sec(observations[agent],
                    included_values = [0, 1])
            )
            logger.debug(f'Rewards for agent "{agent}": ' + str(rewards[agent]))
            logger.debug(
                f'Termination for agent "{agent}": ' + str(terminations[agent])
            )
            logger.debug(f'Truncation for agent "{agent}": ' + str(truncations[agent]))
            agent_info_str = self._format_info(infos[agent])
            logger.debug(f'Info for agent "{agent}":\n' + agent_info_str)

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
        logger.debug("Stepping through iteration " f"{self.cur_iter}/{self.max_iter}.")
        logger.debug(f"Performing actions: {actions}.")

        # Peform agent actions
        for agent in self.agents:
            action = actions[agent]
            if action[0] == 0:
                continue

            action_step = action[1]
            if self.agents_dict[agent]["type"] == "attacker":
                self._attacker_step(agent, action_step)
            elif self.agents_dict[agent]["type"] == "defender":
                self._defender_step(agent, action_step)
            else:
                logger.error(
                    f'Agent {agent} has unknown type: '
                    f'{self.agents_dict[agent]["type"]}.'
                )

        observations, rewards, terminations, truncations, infos = (
            self._observe_and_reward()
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
