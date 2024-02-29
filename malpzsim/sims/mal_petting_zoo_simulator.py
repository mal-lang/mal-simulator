import sys
import copy
import logging
import functools
from typing import List, Tuple, Optional
import numpy as np

import maltoolbox
from maltoolbox.model.model import Model
from maltoolbox.language.languagegraph import LanguageGraph
from maltoolbox.attackgraph.attackgraph import AttackGraph
from maltoolbox.attackgraph.attacker import Attacker
from maltoolbox.attackgraph.node import AttackGraphNode
import maltoolbox.attackgraph.analyzers.apriori as apriori
import maltoolbox.attackgraph.query as query
from maltoolbox.ingestors import neo4j

from gymnasium.spaces import MultiDiscrete, Box, Dict
from pettingzoo import ParallelEnv

ITERATIONS_LIMIT = int(1e9)

logger = logging.getLogger(__name__)

class MalPettingZooSimulator(ParallelEnv):

    def __init__(self,
        lang_graph: LanguageGraph,
        model: Model,
        attack_graph: AttackGraph,
        max_iter = ITERATIONS_LIMIT):
        super().__init__()
        logger.info('Create Petting Zoo Mal Simulator.')
        self.lang_graph = lang_graph
        self.model = model
        self.attack_graph = attack_graph
        self.max_iter = max_iter

        self.possible_agents = []
        self.agents = []
        self.agents_dict = {}

        self.init(self.max_iter)


    def create_blank_observation(self):
        num_actions = 2
        # For now, an `object` is an attack step
        num_objects = len(self.attack_graph.nodes)
        num_lang_asset_types = len(self.lang_graph.assets)
        num_lang_attack_steps = len(self.lang_graph.attack_steps)

        observation = {
            'action' : num_actions * [0],
            'step' : num_objects * [0],
            'is_observable' : num_objects * [1],
            'observed_state' : num_objects * [-1],
            'remaining_ttc' : num_objects * [0]
        }

        observation['asset_type'] = []
        observation['asset_id'] = []
        observation['step_name'] = []
        for step in self.attack_graph.nodes:
            if step.name == 'firstSteps':
                observation['asset_type'].append(-1)
                observation['asset_id'].append(-1)
                observation['step_name'].append(-1)
                continue
            observation['asset_type'].append(
                self._asset_type_to_index[step.asset.metaconcept])
            observation['asset_id'].append(int(step.asset.id))
            step_name_with_asset = step.asset.metaconcept + ':' + step.name
            observation['step_name'].append(self._step_name_to_index[step_name_with_asset])

        observation['edges'] = []
        for attack_step in self.attack_graph.nodes:
            for child in attack_step.children:
                observation['edges'].append(
                    [self._id_to_index[attack_step.id],
                    self._id_to_index[child.id]])

        return observation

    def _format_full_observation(self, observation):
        '''
        Return a formatted string of the entire observation. This includes
        sections that will not change over time, these define the structure of
        the attack graph.
        '''
        obs_str = f'Action: {observation["action"]}\n'

        str_format = '{:<5} {:<6} {:<5} {:<5} {:<5} {:<5} {:<}\n'
        header = str_format.format(
            'Entry', 'Is_Obs', 'State', 'RTTC', 'Type', 'Id', 'Step')
        obs_str += header
        for entry in range(0, len(observation['observed_state'])):
            obs_str +=  str_format.format(
                entry,
                observation['is_observable'][entry],
                observation['observed_state'][entry],
                observation['remaining_ttc'][entry],
                observation['asset_type'][entry],
                observation['asset_id'][entry],
                observation['step_name'][entry])
            if entry % 30 == 29:
                obs_str += header

        obs_str += '\nEdges:\n'
        for edge in observation['edges']:
            obs_str += str(edge) + '\n'

        return obs_str

    def _format_obs_var_sec(self, observation):
        '''
        Return a formatted string of the sections of the observation that can
        vary over time.
        '''
        obs_str = ''

        str_format = '{:>80} {:<5} {:<5} {:<}\n'
        header = str_format.format(
            'Id', 'State', 'RTTC', 'Entry')
        obs_str += header
        listing_nr = 0
        for entry in range(0, len(observation['observed_state'])):
            if observation['is_observable'][entry]:
                obs_str +=  str_format.format(
                    self._index_to_id[entry],
                    observation['observed_state'][entry],
                    observation['remaining_ttc'][entry],
                    entry)
                listing_nr += 1
            if listing_nr % 30 == 29:
                obs_str += header

        return obs_str

    def _format_info(self, info):
        can_act = 'Yes' if info['action_mask'][0] > 0 \
            else 'No'
        agent_info_str = f'Can act? {can_act}\n'
        for entry in range(0, len(info['action_mask'][1])):
            if info['action_mask'][1][entry] == 1:
                agent_info_str += f'{self._index_to_id[entry]}\n'
        return agent_info_str

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        num_actions = 2
        # For now, an `object` is an attack step
        num_objects = len(self.attack_graph.nodes)
        num_lang_asset_types = len(self.lang_graph.assets)
        num_lang_attack_steps = len(self.lang_graph.attack_steps)
        num_edges = len(self._blank_observation['edges'])
        # TODO action, step, and is_observable are never set. Figure out what
        # action and step should be set to or remove them if redundant.
        # is_observable will be filled in once the observability of the attack
        # graph is determined.
        return Dict(
            {
                'is_observable': Box(
                    0, 1, shape=(num_objects,), dtype=np.int8
                ),  #  0 for unobservable,
                    #  1 for observable
                'observed_state': Box(
                    -1, 1, shape=(num_objects,), dtype=np.int8
                ),  # -1 for unknown,
                    #  0 for disabled/not compromised,
                    #  1 for enabled/compromised
                'remaining_ttc': Box(
                    0, sys.maxsize, shape=(num_objects,), dtype=np.int64
                ),  # remaining TTC
                'asset_type': Box(
                    -1, num_lang_asset_types, shape=(num_objects,), dtype=np.int64
                ),  # asset type
                'asset_id': Box(
                    -1, sys.maxsize, shape=(num_objects,), dtype=np.int64
                ),  # asset id
                'step_name': Box(
                    -1, num_lang_attack_steps, shape=(num_objects,), dtype=np.int64
                ),  # attack/defense step name
                'edges': Box(
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
        return MultiDiscrete([num_actions, num_objects])

    def reset(self,
            seed: Optional[int] = None,
            options: Optional[dict] = None):
        logger.info('Resetting simulator.')
        # Regenerate the attack graph from the model
        self.attack_graph = AttackGraph(self.model.lang_spec, self.model)
        self.attack_graph.attach_attackers(self.model)
        apriori.calculate_viability_and_necessity(self.attack_graph)

        return self.init(self.max_iter)

    def init(self, max_iter = ITERATIONS_LIMIT):
        logger.info('Initializing MAL Petting Zoo ParralelEnv Simulator.')
        logger.debug('Creating and listing mapping tables.')
        self._index_to_id = [n.id for n in self.attack_graph.nodes]
        self._id_to_index = {n: i for i, n in enumerate(self._index_to_id)}
        str_format = '{:<5} {:<}\n'
        table = '\n' + str_format.format(
            'Index', 'Attack Step Id')
        for entry in self._index_to_id:
            table += str_format.format(self._id_to_index[entry], entry)
        logger.debug(table)

        self._index_to_asset_type = [n.name for n in self.lang_graph.assets]
        self._asset_type_to_index = {n: i for i, n in
            enumerate(self._index_to_asset_type)}
        str_format = '{:<5} {:<}\n'
        table = '\n' + str_format.format(
            'Index', 'Asset Type')
        for entry in self._index_to_asset_type:
            table += str_format.format(self._asset_type_to_index[entry], entry)
        logger.debug(table)

        self._index_to_step_name = [n.name for n in
            self.lang_graph.attack_steps]
        self._step_name_to_index = {n: i for i, n in
            enumerate(self._index_to_step_name)}
        str_format = '{:<5} {:<}\n'
        table = '\n' + str_format.format(
            'Index', 'Step Name')
        for entry in self._index_to_step_name:
            table += str_format.format(self._step_name_to_index[entry], entry)
        logger.debug(table)

        self.max_iter = max_iter
        self.cur_iter = 0

        logger.debug('Creating and listing blank observation space.')
        self._blank_observation = self.create_blank_observation()
        logger.debug(self._format_full_observation(self._blank_observation))

        logger.info('Populate agents list with all possible agents.')
        self.agents = copy.deepcopy(self.possible_agents)

        observations = {}
        infos = {}

        for agent in self.agents:
            observations[agent] = copy.deepcopy(self._blank_observation)
            #TODO Flag initial entry points for attacker

            available_actions = [0] * len(self.attack_graph.nodes)
            can_act = 0
            if self.agents_dict[agent]['type'] == 'defender':
                enabled_defenses = query.get_enabled_defenses(
                        self.attack_graph)
                for node in query.get_defense_surface(self.attack_graph):
                    if node not in enabled_defenses:
                        index = self._id_to_index[node.id]
                        available_actions[index] = 1
                        can_act = 1

            if self.agents_dict[agent]['type'] == 'attacker':
                attacker = self.attack_graph.\
                    attackers[self.agents_dict[agent]['attacker']]
                for node in query.get_attack_surface(self.attack_graph,
                    attacker):
                    if attacker not in node.compromised_by:
                        index = self._id_to_index[node.id]
                        available_actions[index] = 1
                        can_act = 1

            infos[agent] = { 'action_mask' : (can_act, available_actions) }

            logger.debug(f'Observation for agent \"{agent}\":\n' +
                self._format_obs_var_sec(observations[agent]))
            agent_info_str = self._format_info(infos[agent])
            logger.debug(f'Info for agent \"{agent}\":\n' +
                agent_info_str)

        return observations, infos

    def register_attacker(self, agent_name, attacker: int):
        logger.info(f'Register attacker \"{agent_name}\" agent with '
            f'attacker index {attacker}.')
        self.possible_agents.append(agent_name)
        self.agents_dict[agent_name] = {'type': 'attacker',
                                        'attacker': attacker}

    def register_defender(self, agent_name):
        # Defenders are run first so that the defenses prevent the attacker
        # appropriately in case the attacker selects an attack step that the
        # defender safeguards against in the same step.
        logger.info(f'Register defender \"{agent_name}\" agent.')
        self.possible_agents.insert(0, agent_name)
        self.agents_dict[agent_name] = {'type': 'defender'}

    def state(self):
        # Should return a state for all agents
        return NotImplementedError

    def _attacker_step(self, agent, attack_step):
        actions = []
        attacker = self.attack_graph.\
            attackers[self.agents_dict[agent]['attacker']]
        attack_step_node = self.attack_graph.get_node_by_id(
            self._index_to_id[attack_step])
        logger.info(f'Attacker agent \"{agent}\" stepping '
            f'through {attack_step_node.id}.')
        if query.is_node_traversable_by_attacker(attack_step_node, attacker):
            if attacker not in attack_step_node.compromised_by:
                logger.debug(f'Attacker {agent} has compromised '
                    f'{attack_step_node.id}.')
                attacker.reached_attack_steps.append(attack_step_node)
                attack_step_node.compromised_by.append(attacker)
            actions.append(attack_step)
            # TODO Update the attack surface of agent.attacker rather than
            # regenerating it every step.
        return actions

    def _defender_step(self, agent, defense_step):
        actions = []
        defense_step_node = self.attack_graph.get_node_by_id(
            self._index_to_id[defense_step])
        logger.info(f'Defender agent \"{agent}\" stepping '
            f'through {defense_step_node.id}.')
        defense_step_node.defense_status = 1.0
        actions.append(defense_step)
        # TODO Update the viability and necessity values related to this
        # defense rather than recalculating them every step.
        # TODO Update the defense surface of the defender agent rather than
        # regenerating it every step.
        apriori.calculate_viability_and_necessity(self.attack_graph)
        return actions

    def _observe_attacker(self, attacker_agent, observation, actions):
        attacker = self.attack_graph.\
            attackers[self.agents_dict[attacker_agent]['attacker']]
        nodes_to_remove = []
        for action in actions:
            observation['observed_state'][action] = 1
        for node in attacker.reached_attack_steps:
            if not query.is_node_traversable_by_attacker(node, attacker):
                # The defender has activated a defense that prevents the
                # attacker from exploiting this attack step any longer.
                nodes_to_remove.append(node)
                index = self._id_to_index[node.id]
                observation['observed_state'][index] = 0
                continue
            index = self._id_to_index[node.id]
            if observation['observed_state'][index] != 1:
                observation['observed_state'][index] = 0

        for node in nodes_to_remove:
            logger.debug('Remove untraversable node from attacker '
            f'\"{attacker_agent}\": {node.id}')
            attacker.reached_attack_steps.remove(node)
            node.compromised_by.remove(attacker)

        for node in query.get_attack_surface(self.attack_graph, attacker):
            index = self._id_to_index[node.id]
            if observation['observed_state'][index] != 1:
                observation['observed_state'][index] = 0

    def _observe_defender(self, defender_agent, observation, actions):
        # TODO We should probably create a separate blank observation for the
        # defenders and just update that with the defense action taken so that
        # we do not have to go through the list of nodes every time.
        for action in actions:
            observation['observed_state'][action] = 1
        enabled_defenses = query.get_enabled_defenses(self.attack_graph)
        for node in self.attack_graph.nodes:
            index = self._id_to_index[node.id]
            if observation['observed_state'][index] != 1:
                if node in enabled_defenses:
                    observation['observed_state'][index] = 1
                else:
                    observation['observed_state'][index] = 0

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
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        logger.debug('Stepping through iteration '
            f'{self.cur_iter}/{self.max_iter}.')
        logger.debug(f'Performing actions: {actions}.')

        # Peform agent actions
        current_step_attacker_actions = []
        current_step_defender_actions = []
        for agent in self.agents:
            action = actions[agent]
            if action[0] == 0:
                continue

            action_step = action[1]
            if self.agents_dict[agent]['type'] == 'attacker':
                agent_actions = self._attacker_step(agent, action_step)
                current_step_attacker_actions.extend(agent_actions)
            elif self.agents_dict[agent]['type'] == 'defender':
                agent_actions = self._defender_step(agent, action_step)
                current_step_defender_actions.extend(agent_actions)
            else:
                logger.error(f'Agent {agent} has unknown type: '
                    '{self.agents_dict[agent]["type"]}')

        finished_agents = []
        # Fill in the agent observations, rewards, terminations, truncations,
        # and infos.
        for agent in self.agents:
            # Collect agent observations
            agent_observation = copy.deepcopy(self._blank_observation)
            if self.agents_dict[agent]['type'] == 'defender':
                self._observe_defender(agent, agent_observation,
                    current_step_attacker_actions + \
                        current_step_defender_actions)
            elif self.agents_dict[agent]['type'] == 'attacker':
                self._observe_attacker(agent, agent_observation,
                    current_step_attacker_actions)
            else:
                logger.error(f'Agent {agent} has unknown type: '
                    '{self.agents_dict[agent]["type"]}')

            observations[agent] = agent_observation

            # Collect agent info, this is used to determine the possible actions
            # in the next iteration step. Then fill in all of the 
            available_actions = [0] * len(self.attack_graph.nodes)
            can_act = 0
            if self.agents_dict[agent]['type'] == 'defender':
                enabled_defenses = query.get_enabled_defenses(
                        self.attack_graph)
                for node in query.get_defense_surface(self.attack_graph):
                    if node not in enabled_defenses:
                        index = self._id_to_index[node.id]
                        available_actions[index] = 1
                        can_act = 1

            if self.agents_dict[agent]['type'] == 'attacker':
                attacker = self.attack_graph.\
                    attackers[self.agents_dict[agent]['attacker']]
                for node in query.get_attack_surface(self.attack_graph,
                    attacker):
                    if attacker not in node.compromised_by:
                        index = self._id_to_index[node.id]
                        available_actions[index] = 1
                        can_act = 1

            infos[agent] = { 'action_mask' : (can_act, available_actions) }

        # First calculate the attacker rewards and attackers' total reward
        attackers_total_rewards = 0
        for agent in self.agents:
            if self.agents_dict[agent]['type'] == 'attacker':
                reward = 0
                attacker = self.attack_graph.\
                    attackers[self.agents_dict[agent]['attacker']]
                for node in attacker.reached_attack_steps:
                    if hasattr(node, 'reward'):
                        reward += node.reward

                attackers_total_rewards += reward
                rewards[agent] = reward

        # Then we can calculate the defender rewards which also include all of
        # the attacker rewards negated.
        for agent in self.agents:
            if self.agents_dict[agent]['type'] == 'defender':
                reward = -attackers_total_rewards
                for node in query.get_enabled_defenses(self.attack_graph):
                    if hasattr(node, 'reward'):
                        reward -= node.reward
                rewards[agent] = reward

        for agent in self.agents:
            #TODO Implement termination appropriate conditions
            terminations[agent] = False
            truncations[agent] = False
            if self.cur_iter >= self.max_iter:
                truncations[agent] = True
                finished_agents.append(agent)

            logger.debug(f'Observation for agent \"{agent}\":\n' +
                self._format_obs_var_sec(agent_observation))
            logger.debug(f'Rewards for agent \"{agent}\": ' +
                str(rewards[agent]))
            logger.debug(f'Termination for agent \"{agent}\": ' +
                str(terminations[agent]))
            logger.debug(f'Truncation for agent \"{agent}\": ' +
                str(truncations[agent]))
            agent_info_str = self._format_info(infos[agent])
            logger.debug(f'Info for agent \"{agent}\":\n' +
                agent_info_str)

        for agent in finished_agents:
            self.agents.remove(agent)

        self.cur_iter += 1

        return observations, rewards, terminations, truncations, infos

    def render(self):
        logger.debug('Ingest attack graph into Neo4J database.')
        neo4j.ingest_attack_graph(self.attack_graph,
            maltoolbox.neo4j_configs['uri'],
            maltoolbox.neo4j_configs['username'],
            maltoolbox.neo4j_configs['password'],
            maltoolbox.neo4j_configs['dbname'],
            delete=True)

