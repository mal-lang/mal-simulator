import sys
import copy
import logging
import functools
from typing import Optional
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

ITERATIONS_LIMIT = int(1e9)

logger = logging.getLogger(__name__)


def _format_full_observation(observation):
    """
    Return a formatted string of the entire observation. This includes
    sections that will not change over time, these define the structure of
    the attack graph.
    """
    obs_str = f'Action: {observation.get("action", "")}\n'

    str_format = "{:<5} {:<6} {:<5} {:<5} {:<5} {:<5} {:<}\n"
    header = str_format.format("Entry", "Is_Obs", "State", "RTTC", "Type", "Id", "Step")
    obs_str += header
    for entry in range(0, len(observation["observed_state"])):
        obs_str += str_format.format(
            entry,
            observation["is_observable"][entry],
            observation["observed_state"][entry],
            observation["remaining_ttc"][entry],
            observation["asset_type"][entry],
            observation["asset_id"][entry],
            observation["step_name"][entry],
        )
        if entry % 30 == 29:
            obs_str += header

    obs_str += "\nEdges:\n"
    for index in range(0, len(observation["edges"][0])):
        obs_str += f"({str(observation["edges"][0][index])}, {str(observation["edges"][1][index])})\n"

    return obs_str


def format_obs_var_sec(observation, index_to_id):
    """
    Return a formatted string of the sections of the observation that can
    vary over time.
    """
    obs_str = ""

    str_format = "{:>80} {:<5} {:<5} {:<}\n"
    header = str_format.format("Id", "State", "RTTC", "Entry")
    obs_str += header
    listing_nr = 0
    for entry in range(0, len(observation["observed_state"])):
        if observation["is_observable"][entry]:
            obs_str += str_format.format(
                index_to_id[entry],
                observation["observed_state"][entry],
                observation["remaining_ttc"][entry],
                entry,
            )
            listing_nr += 1
        if listing_nr % 30 == 29:
            obs_str += header

    return obs_str


class MalPettingZooSimulator(ParallelEnv):
    def __init__(
        self,
        lang_graph: LanguageGraph,
        model: Model,
        attack_graph: AttackGraph,
        max_iter=ITERATIONS_LIMIT,
        **kwargs,
    ):
        super().__init__()
        logger.info("Create Petting Zoo Mal Simulator.")
        self.lang_graph = lang_graph
        self.model = model
        self.attack_graph = attack_graph
        self.max_iter = max_iter

        self.attack_graph.save_to_file('tmp/original_attack_graph.json')

        self.possible_agents = []
        self.agents = []
        self.agents_dict = {}
        self.offset = 1
        self.unholy = kwargs.get(
            "unholy", False
        )  # Separates attack step names from their assets in the observation.
        # Not compliant with how the MAL language is supposed to work, but reduces the size of the observation signficiantly.

        self.init(self.max_iter)

    @property
    @functools.lru_cache(maxsize=None)
    def num_assets(self):
        return len(self.lang_graph.assets) + self.offset

    @property
    @functools.lru_cache(maxsize=None)
    def num_step_names(self):
        return (
            len(self.lang_graph.attack_steps)
            if not self.unholy
            else len(set(s.attributes["name"] for s in self.lang_graph.attack_steps))
        ) + self.offset

    def asset_type(self, step):
        return (
            self._asset_type_to_index[step.asset.metaconcept] + self.offset
            if step.name != "firstSteps"
            else 0
        )

    def step_name(self, step):
        return (
            (
                self._step_name_to_index[step.asset.metaconcept + ":" + step.name]
                + self.offset
                if not self.unholy
                else self._unholy_step_name_to_index[step.attributes["name"]]
                + self.offset
            )
            if step.name != "firstSteps"
            else 0
        )

    def asset_id(self, step):
        return int(step.asset.id) if step.name != "firstSteps" else 0

    def create_blank_observation(self):
        # For now, an `object` is an attack step
        num_objects = len(self.attack_graph.nodes)

        observation = {
            "is_observable": num_objects * [1],
            "observed_state": num_objects * [-1],
            "remaining_ttc": num_objects * [0],
        }

        logger.debug(f'Create blank observation with {num_objects} attack '
            'steps.')
        observation["asset_type"], observation["asset_id"], observation["step_name"] = (
            zip(
                *(
                    (self.asset_type(step), self.asset_id(step), self.step_name(step))
                    for step in self.attack_graph.nodes
                )
            )
        )

        observation["edges"] = [[],[]]

        for attack_step in self.attack_graph.nodes:
            for child in attack_step.children:
                observation["edges"][0].append(self._id_to_index[attack_step.id])
                observation["edges"][1].append(self._id_to_index[child.id])

        # Create reverse edges for defense steps. This was required by some of
        # the defender agent logic.
        for attack_step in self.attack_graph.nodes:
            if attack_step.type == "defense":
                for child in attack_step.children:
                    observation['edges'][0].append(self._id_to_index[child.id])
                    observation['edges'][1].append(self._id_to_index[attack_step.id])

        np_obs = {
            "is_observable": np.array(observation["is_observable"], dtype=np.int8),
            "observed_state": np.array(observation["observed_state"], dtype=np.int8),
            "remaining_ttc": np.array(observation["remaining_ttc"], dtype=np.int64),
            "asset_type": np.array(observation["asset_type"], dtype=np.int64),
            "asset_id": np.array(observation["asset_id"], dtype=np.int64),
            "step_name": np.array(observation["step_name"], dtype=np.int64),
            "edges": np.array(observation["edges"], dtype=np.int64),
        }

        return np_obs

    def _format_info(self, info):
        can_act = "Yes" if info["action_mask"][0][1] > 0 else "No"
        agent_info_str = f"Can act? {can_act}\n"
        for entry in range(0, len(info["action_mask"][1])):
            if info["action_mask"][1][entry] == 1:
                agent_info_str += f"{self._index_to_id[entry]}\n"
        return agent_info_str

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # For now, an `object` is an attack step
        num_objects = len(self.attack_graph.nodes)
        num_lang_asset_types = len(self.lang_graph.assets)
        num_lang_attack_steps = (
            len(self.lang_graph.attack_steps)
            if not self.unholy
            else len(set(s.attributes["name"] for s in self.lang_graph.attack_steps))
        )
        num_edges = len(self._blank_observation["edges"])
        # TODO is_observable is never set. It will be filled in once the
        # observability of the attack graph is determined.
        return Dict(
            {
                "is_observable": Box(
                    0, 1, shape=(num_objects,), dtype=np.int8
                ),  #  0 for unobservable,
                #  1 for observable
                "observed_state": Box(
                    -1, 1, shape=(num_objects,), dtype=np.int8
                ),  # -1 for unknown,
                #  0 for disabled/not compromised,
                #  1 for enabled/compromised
                "remaining_ttc": Box(
                    0, sys.maxsize, shape=(num_objects,), dtype=np.int64
                ),  # remaining TTC
                "asset_type": Box(
                    0,
                    num_lang_asset_types + self.offset,
                    shape=(num_objects,),
                    dtype=np.int64,
                ),  # asset type
                "asset_id": Box(
                    0, sys.maxsize, shape=(num_objects,), dtype=np.int64
                ),  # asset id
                "step_name": Box(
                    0,
                    num_lang_attack_steps + self.offset,
                    shape=(num_objects,),
                    dtype=np.int64,
                ),  # attack/defense step name
                "edges": Box(
                    0,
                    num_objects,
                    shape=(2, num_edges),
                    dtype=np.int64,
                ),  # edges between steps
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        num_actions = 2  # two actions: wait or use
        # For now, an `object` is an attack step
        num_objects = len(self.attack_graph.nodes)
        return MultiDiscrete([num_actions, num_objects], dtype=np.int64)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        logger.info("Resetting simulator.")
        attack_graph = AttackGraph()
        attack_graph.load_from_file('tmp/original_attack_graph.json',
            self.model)
        apriori.calculate_viability_and_necessity(attack_graph)
        self.attack_graph = attack_graph
        return self.init(self.max_iter)

    def init(self, max_iter=ITERATIONS_LIMIT):
        logger.info("Initializing MAL Petting Zoo ParralelEnv Simulator.")
        logger.debug("Creating and listing mapping tables.")
        self._index_to_id = [n.id for n in self.attack_graph.nodes]
        self._id_to_index = {n: i for i, n in enumerate(self._index_to_id)}
        str_format = "{:<5} {:<}\n"
        table = "\n" + str_format.format("Index", "Attack Step Id")
        for entry in self._index_to_id:
            table += str_format.format(self._id_to_index[entry], entry)
        logger.debug(table)

        self._index_to_asset_type = [n.name for n in self.lang_graph.assets]
        self._asset_type_to_index = {
            n: i for i, n in enumerate(self._index_to_asset_type)
        }
        str_format = "{:<5} {:<}\n"
        table = "\n" + str_format.format("Index", "Asset Type")
        for entry in self._index_to_asset_type:
            table += str_format.format(self._asset_type_to_index[entry], entry)
        logger.debug(table)

        self._index_to_step_name = [n.name for n in self.lang_graph.attack_steps]
        self._step_name_to_index = {
            n: i for i, n in enumerate(self._index_to_step_name)
        }

        self._unholy_index_to_step_name = {
            n.attributes["name"] for n in self.lang_graph.attack_steps
        }
        self._unholy_step_name_to_index = {
            n: i for i, n in enumerate(self._unholy_index_to_step_name)
        }

        str_format = "{:<5} {:<}\n"
        table = "\n" + str_format.format("Index", "Step Name")
        for entry in self._index_to_step_name:
            table += str_format.format(self._step_name_to_index[entry], entry)
        logger.debug(table)

        self.max_iter = max_iter
        self.cur_iter = 0

        logger.debug("Creating and listing blank observation space.")
        self._blank_observation = self.create_blank_observation()
        logger.debug(_format_full_observation(self._blank_observation))

        logger.info("Populate agents list with all possible agents.")
        self.agents = copy.deepcopy(self.possible_agents)

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
        self.agents_dict[agent_name] = {"type": "attacker", "attacker": attacker}

    def register_defender(self, agent_name):
        # Defenders are run first so that the defenses prevent the attacker
        # appropriately in case the attacker selects an attack step that the
        # defender safeguards against in the same step.
        logger.info(f'Register defender "{agent_name}" agent.')
        self.possible_agents.insert(0, agent_name)
        self.agents_dict[agent_name] = {"type": "defender"}

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
            f'Attacker agent "{agent}" stepping ' f"through {attack_step_node.id}."
        )
        if query.is_node_traversable_by_attacker(attack_step_node, attacker):
            if not attack_step_node.is_compromised_by(attacker):
                logger.debug(
                    f"Attacker {agent} has compromised " f"{attack_step_node.id}."
                )
                attacker.compromise(attack_step_node)
            actions.append(attack_step)
            # TODO Update the attack surface of agent.attacker rather than
            # regenerating it every step.
        return actions

    def _defender_step(self, agent, defense_step):
        actions = []
        defense_step_node = self.attack_graph.get_node_by_id(
            self._index_to_id[defense_step]
        )
        logger.info(
            f'Defender agent "{agent}" stepping through ' f"{defense_step_node.id}."
        )
        defense_step_node.defense_status = 1.0
        actions.append(defense_step)
        # TODO Update the viability and necessity values related to this
        # defense rather than recalculating them every step.
        # TODO Update the defense surface of the defender agent rather than
        # regenerating it every step.
        apriori.calculate_viability_and_necessity(self.attack_graph)
        return actions

    def _observe_attacker(self, attacker_agent, observation):
        attacker = self.attack_graph.attackers[
            self.agents_dict[attacker_agent]["attacker"]
        ]
        nodes_to_remove = []
        for node in attacker.reached_attack_steps:
            if not query.is_node_traversable_by_attacker(node, attacker):
                # The defender has activated a defense that prevents the
                # attacker from exploiting this attack step any longer.
                nodes_to_remove.append(node)
                index = self._id_to_index[node.id]
                observation["observed_state"][index] = 0
                continue
            index = self._id_to_index[node.id]
            observation["observed_state"][index] = 1

        for node in nodes_to_remove:
            logger.debug(
                "Remove untraversable node from attacker "
                f'"{attacker_agent}": {node.id}'
            )
            attacker.undo_compromise(node)

        for node in query.get_attack_surface(self.attack_graph, attacker):
            index = self._id_to_index[node.id]
            if observation["observed_state"][index] != 1:
                observation["observed_state"][index] = 0

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
                for node in query.get_defense_surface(self.attack_graph):
                    index = self._id_to_index[node.id]
                    available_actions[index] = 1
                    can_act = 1

            if agent_type == "attacker":
                attacker = self.attack_graph.attackers[
                    self.agents_dict[agent]["attacker"]
                ]
                for node in query.get_attack_surface(self.attack_graph, attacker):
                    if not node.is_compromised_by(attacker):
                        index = self._id_to_index[node.id]
                        available_actions[index] = 1
                        can_act = 1
                        attackers_done = False

            infos[agent] = {
                "action_mask": ([can_wait[agent_type], can_act], available_actions)
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
                    if hasattr(node, "reward"):
                        reward += node.reward

                attackers_total_rewards += reward
                rewards[agent] = reward

        # Then we can calculate the defender rewards which also include all of
        # the attacker rewards negated.
        for agent in self.agents:
            if self.agents_dict[agent]["type"] == "defender":
                reward = -attackers_total_rewards
                for node in query.get_enabled_defenses(self.attack_graph):
                    if hasattr(node, "reward"):
                        reward -= node.reward
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
                + format_obs_var_sec(observations[agent], self._index_to_id)
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
