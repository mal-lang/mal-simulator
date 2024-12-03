from __future__ import annotations

import sys
import copy
import logging
import functools
from typing import Optional
import numpy as np
from enum import Enum

from gymnasium.spaces import MultiDiscrete, Box, Dict
from pettingzoo import ParallelEnv

from maltoolbox import neo4j_configs
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from maltoolbox.attackgraph import query
from maltoolbox.ingestors import neo4j

from .mal_simulator_settings import MalSimulatorSettings
from .base_mal_simulator import BaseMalSimulator, AgentType, SimulatorAgent
from .mal_sim_logging_utils import format_full_observation,\
                                   log_mapping_tables, log_agent_state

ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)

class ActionState(Enum):
    """The state of an agent action."""
    WAIT = 0  # Agent is waiting, not acting on any node
    ACT = 1   # Agent is acting on a specific node

class MalSimulator(BaseMalSimulator, ParallelEnv):
    def __init__(
        self,
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

        super(ParallelEnv).__init__()
        super().__init__(
            attack_graph,
            prune_unviable_unnecessary=prune_unviable_unnecessary,
            sim_settings=sim_settings,
            max_iter=max_iter
        )

        logger.info("Creating Mal Simulator.")
        self.lang_graph = attack_graph.lang_graph
        self.model = attack_graph.model
        self._create_mapping_tables()
        self._blank_observation = self.create_blank_observation()

        # Initialize agents and record the entry point actions
        self._initialize_agents()

    def __call__(self):
        return self

    def create_blank_observation(self, default_obs_state=-1):
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
        initial_actions = self._initialize_agents()

        # Apply initial actions to observations
        self._observe_and_reward(initial_actions, {})
        _, infos = self._collect_agents_infos()
        observations = {a.name: a.observation for a in self.agents}

        return observations, infos

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

        if logger.isEnabledFor(logging.DEBUG):
            log_mapping_tables(logger, self)

    def get_attack_graph_node_by_index(self, index: int):
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
            raise IndexError('Index given, %d, is out of range of the '
                'lookup list which is of length %d' % (index,
                    len(self._index_to_id)))

        node_id = self._index_to_id[index]
        node = self.attack_graph.get_node_by_id(node_id)
        if not node:
            raise LookupError('Index given, %d(id: %d), does not map to a '
                'node' % (index, node_id))
        return node

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

        # Will contain initally enabled steps
        initial_actions = {}

        for agent in self.agents:
            # Initialize rewards
            agent.reward = 0
            initial_actions[agent.name] = []

            if agent.type == AgentType.ATTACKER:
                attacker = \
                    self.attack_graph.get_attacker_by_id(agent.attacker_id)
                assert attacker, f"No attacker with id {agent.attacker_id}"

                # Initialize observations and action surfaces
                agent.observation = self.create_blank_observation()
                agent.action_surface = query.get_attack_surface(attacker)

                # Initial actions for attacker are its entrypoints
                for entry_point in attacker.entry_points:
                    initial_actions[agent.name].append(
                        self._id_to_index[entry_point.id])
                    entry_point.extras['entrypoint'] = True

            elif agent.type == AgentType.DEFENDER:
                # Initialize observations and action surfaces
                agent.observation = \
                    self.create_blank_observation(default_obs_state = 0)
                agent.action_surface = \
                    query.get_defense_surface(self.attack_graph)

                # Initial actions for defender are all pre-enabled defenses
                initial_actions[agent.name] = [self._id_to_index[node.id]
                                          for node in self.attack_graph.nodes
                                          if node.is_enabled_defense()]

            else:
                agent.action_surface = []

        return initial_actions

    def state(self):
        # Should return a state for all agents
        return NotImplementedError

    def _attacker_step(self, agent: SimulatorAgent, attack_step):
        attack_step_node = self.get_attack_graph_node_by_index(attack_step)
        enabled_nodes = super()._attacker_step(agent, [attack_step_node])
        # Must be converted to indices in MalSimulator
        return [self._id_to_index[n.id] for n in enabled_nodes]

    def _defender_step(
            self, agent, defense_step_index
        ) -> tuple[list[int], list[AttackGraphNode]]:

        defense_step_node = self.attack_graph.get_node_by_id(
            self._index_to_id[defense_step_index])

        enabled_nodes, unviable_nodes = \
            super()._defender_step(agent, [defense_step_node])

        # Must be converted to indices in MalSimulator
        return [self._id_to_index[n.id] for n in enabled_nodes], \
               unviable_nodes

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
        obs_state = attacker_agent.observation.get("observed_state")

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

        obs_state = defender_agent.observation["observed_state"]

        if not self.sim_settings.cumulative_defender_obs:
            # Clear the state if we do not it to accumulate observations over
            # time.
            obs_state.fill(0)

        # Enable the latest steps taken
        for _, actions in performed_actions.items():
            for action in actions:
                obs_state[action] = 1

    def _observe_agents(self, performed_actions):
        """Collect agents observations"""

        for agent in self.agents:
            if agent.type == AgentType.DEFENDER:
                self._observe_defender(agent, performed_actions)

            elif agent.type == AgentType.ATTACKER:
                self._observe_attacker(agent, performed_actions)

            else:
                logger.error(
                    "Agent %s has unknown type: %s",
                    agent, self.agents_dict[agent]["type"]
                )

    def _collect_agents_infos(self):
        """Collect agent info, this is used to determine the possible
        actions in the next iteration step. Then fill in all of the"""

        attackers_done = True
        infos = {}
        can_wait = {
            AgentType.ATTACKER: 0,
            AgentType.DEFENDER: 1,
        }

        for agent in self.agents:
            available_actions = [0] * len(self.attack_graph.nodes)
            can_act = 0

            if agent.type == AgentType.DEFENDER:
                for node in agent.action_surface:
                    index = self._id_to_index[node.id]
                    available_actions[index] = 1
                    can_act = 1

            if agent.type == AgentType.ATTACKER:
                attacker = \
                    self.attack_graph.get_attacker_by_id(agent.attacker_id)

                for node in agent.action_surface:
                    if not node.is_compromised_by(attacker):
                        index = self._id_to_index[node.id]
                        available_actions[index] = 1
                        can_act = 1
                        attackers_done = False

            infos[agent.name] = {
                "action_mask": (
                    np.array(
                        [can_wait[agent.type], can_act], dtype=np.int8),
                    np.array(
                        available_actions, dtype=np.int8)
                )}

        return attackers_done, infos


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
        attackers_done, infos = self._collect_agents_infos()

        for agent in self.agents:
            # Terminate simulation if no attackers have actions to take
            terminations[agent.name] = attackers_done
            if attackers_done:
                logger.debug(
                    "No attacker has actions left to perform, "
                    "terminate agent \"%s\".", agent.name)

            truncations[agent.name] = False
            if self.cur_iter >= self.max_iter:
                logger.debug(
                    "Simulation has reached the maximum number of "
                    "iterations, %d, terminate agent \"%s\".",
                    self.max_iter, agent.name)
                truncations[agent.name] = True

            if terminations[agent.name] or truncations[agent.name]:
                finished_agents.append(agent)

            if logger.isEnabledFor(logging.DEBUG):
                log_agent_state(
                    logger, self, agent, terminations, truncations, infos
                )

        observations = {name: agent.observation
                        for name, agent in self.agents_dict.items()}
        rewards = {name: agent.reward
                   for name, agent in self.agents_dict.items()}

        # TODO: ask why we need this
        # for agent in finished_agents:
        #     self.agents.remove(agent)
        # TODO: in meantime:
        for agent in finished_agents:
            agent.done = True

        return observations, rewards, terminations, truncations, infos

    def step(
            self, actions: dict[str, tuple(ActionState, int)]
        ) -> tuple[dict, dict, dict, dict, dict]:
        """
        Arguments:
        actions - dict mapping from agent name to action to take
                  the action is a tuple with action state and step index
        Returns:
        observations - dict from agent name to observation dict
        rewards - dict from agent name to int reward value in this step
        terminations - dict from agent name to boolean
        truncations - dict from agent name to boolean
        infos - dict from agent name to agent info dict
        """
        logger.debug(
            "Stepping through iteration %d/%d", self.cur_iter, self.max_iter)
        logger.debug("Performing actions: %s", actions)

        # Map agent to defense/attack steps performed in this step
        performed_actions = {}
        prevented_attack_steps = []
        default_action = (ActionState.WAIT, None)

        # Peform agent actions
        for agent in self.agents:
            action = actions.get(agent.name, default_action)
            action_state = ActionState(action[0])
            action_step_index = action[1]

            if action_state == ActionState.WAIT:
                # Agent wants to wait - do nothing
                continue

            if agent.type == AgentType.ATTACKER:
                performed_actions[agent.name] = \
                    self._attacker_step(agent, action_step_index)

            elif agent.type == AgentType.DEFENDER:
                defender_actions, prevented_attack_steps = \
                    self._defender_step(agent, action_step_index)
                performed_actions[agent.name] = defender_actions

            else:
                logger.error(
                    'Agent %s has unknown type: %s',
                    agent, self.agents_dict[agent]["type"])

        self.cur_iter += 1
        return self._observe_and_reward(
            performed_actions,
            prevented_attack_steps
        )

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
