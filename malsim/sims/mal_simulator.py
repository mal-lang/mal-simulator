from __future__ import annotations

import copy
import logging
from typing import Optional

from maltoolbox import neo4j_configs
from maltoolbox.ingestors import neo4j
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode, query
from maltoolbox.attackgraph.analyzers import apriori

from .mal_sim_settings import MalSimulatorSettings
from .mal_sim_agent import (
    AgentType,
    MalSimAgent,
    MalSimAgentView,
    MalSimAttacker,
    MalSimDefender
)

ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)

class MalSimulator():
    """A MAL Simulator that works on the AttackGraph

    Allows user to register agents (defender and attacker)
    and lets the agents perform actions step by step and updates
    the state of the attack graph based on the steps chosen.
    """

    def __init__(
        self,
        attack_graph: AttackGraph,
        prune_unviable_unnecessary: bool = True,
        sim_settings: MalSimulatorSettings = MalSimulatorSettings(),
        max_iter=ITERATIONS_LIMIT,
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
        logger.info("Creating Base MAL Simulator.")

        # Calculate viability and necessity and optionally prune graph
        apriori.calculate_viability_and_necessity(attack_graph)
        if prune_unviable_unnecessary:
            apriori.prune_unviable_and_unnecessary_nodes(attack_graph)

        # Keep a backup attack graph to use when resetting
        self.attack_graph_backup = copy.deepcopy(attack_graph)

        # Initialize all values
        self.attack_graph = attack_graph
        self.lang_graph = attack_graph.lang_graph
        self.model = attack_graph.model

        self.sim_settings = sim_settings
        self.max_iter = max_iter # Max iterations before stopping simulation
        self.cur_iter = 0        # Keep track on current iteration

        # Keep track on all registered agent states
        self._agents_dict: dict[str, MalSimAgent] = {}

        # Keep track on all 'living' agents sorted by order to step in
        self.agents: list[str] = []

        # Keep track on all (dead and alive) agents sorted by order to step in
        # (Used when resetting)
        self.possible_agents: list[str] = []

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None
        ) -> dict[str, MalSimAgent]:
        """Reset attack graph, iteration and reinitialize agents"""

        logger.info("Resetting MAL Simulator.")
        # Reset attack graph
        self.attack_graph = copy.deepcopy(self.attack_graph_backup)
        # Reset current iteration
        self.cur_iter = 0
        # Reset agents
        self._reset_agents()

        return self._agents_dict

    def _init_agent_rewards(self):
        """Give rewards for pre-enabled attack/defense steps"""
        for node in self.attack_graph.nodes:
            node_reward = node.extras.get('reward', 0)
            if not node_reward:
                continue

            for agent in self._agents_dict.values():

                if agent.type == AgentType.ATTACKER:
                    attacker = self.attack_graph.get_attacker_by_id(
                        agent.attacker_id
                    )

                    if node.is_compromised_by(attacker):
                        # Attacker is rewarded for pre enabled attack steps
                        agent.reward += node_reward

                elif agent.type == AgentType.DEFENDER:
                    # Defender is penalized for all pre-enabled steps
                    if node.is_compromised() or node.is_enabled_defense():
                        agent.reward -= node_reward

    def _init_agent_action_surfaces(self):
        """Set agent action surfaces according to current state"""
        for agent in self._agents_dict.values():
            if agent.type == AgentType.ATTACKER:
                # Get the Attacker object
                attacker = \
                    self.attack_graph.get_attacker_by_id(agent.attacker_id)

                # Get current action surface
                agent.action_surface = \
                    query.get_attack_surface(attacker)

            elif agent.type == AgentType.DEFENDER:
                # Get current action surface
                agent.action_surface = \
                    query.get_defense_surface(self.attack_graph)
            else:
                raise LookupError(f"Agent type {agent.type} not supported")

    def _reset_agents(self):
        """Reset agent rewards and action surfaces"""

        # Revive dead agents
        self.agents = copy.deepcopy(self.possible_agents)

        for agent in self._agents_dict.values():
            # Reset agent reward
            agent.reward = 0

            # Mark agent as alive again
            agent.terminated = False
            agent.truncated = False

        self._init_agent_rewards()
        self._init_agent_action_surfaces()

    def _register_agent(self, agent: MalSimAgent):
        """Register a mal sim agent"""

        logger.info('Registering agent "%s".', agent)
        assert agent.name not in self._agents_dict, \
            f"Duplicate agent named {agent.name} not allowed"

        if agent.type == AgentType.DEFENDER:
            # Defender is first in list so it can pick
            # actions before attacker when step performed
            self.agents.insert(0, agent.name)
            self.possible_agents.insert(0, agent.name)
        elif agent.type == AgentType.ATTACKER:
            # Attacker goes last
            self.agents.append(agent.name)
            self.possible_agents.append(agent.name)

        self._agents_dict[agent.name] = agent

    def register_attacker(self, name: str, attacker_id: int):
        """Register a mal sim attacker agent"""
        agent_state = MalSimAttacker(name, attacker_id)
        self._register_agent(agent_state)

    def register_defender(self, name: str):
        """Register a mal sim defender agent"""
        agent_state = MalSimDefender(name)
        self._register_agent(agent_state)

    def get_agent(self, name: str) -> MalSimAgentView:
        """Return read only agent state for agent with given name"""

        assert name in self._agents_dict, (
            f"Agent with name '{name}' does not exist")
        agent = self._agents_dict[name]
        return MalSimAgentView(agent)

    def get_agents(self) -> list[MalSimAgentView]:
        """Return read only agent state for all dead and alive agents"""
        return [self.get_agent(agent) for agent in self.possible_agents]

    def _get_attacker_agents(self) -> list[MalSimAttacker]:
        """Return list of mutable attacker agent states"""
        return [a for a in self._agents_dict.values()
                if a.type == AgentType.ATTACKER]

    def _get_defender_agents(self) -> list[MalSimDefender]:
        """Return list of mutable defender agent states"""
        return [a for a in self._agents_dict.values()
                if a.type == AgentType.DEFENDER]

    def _disable_attack_steps(
            self, attack_steps_to_disable: list[AttackGraphNode]
        ):
        """Disable nodes for each attacker agent

        For each compromised attack step uncompromise the node, disable its
        observed_state, and remove the rewards.
        """
        for attacker_agent in self._get_attacker_agents():
            attacker = self.attack_graph.get_attacker_by_id(
                attacker_agent.attacker_id)

            for unviable_node in attack_steps_to_disable:
                if unviable_node.is_compromised_by(attacker):

                    # Reward is no longer present for attacker
                    node_reward = unviable_node.extras.get('reward', 0)
                    attacker_agent.reward -= node_reward

                    # Reward is no longer present for defenders
                    for defender_agent in self._get_defender_agents():
                        defender_agent.reward += node_reward

                    # Uncompromise node if requested
                    attacker.undo_compromise(unviable_node)

    def _attacker_step(
            self, agent: MalSimAttacker, nodes: list[AttackGraphNode]
        ) -> list[AttackGraphNode]:
        """Compromise attack step nodes with attacker

        Args:
        agent - the agent to compromise nodes with
        nodes - the nodes to compromise

        Returns: list of attack steps nodes that were compromised
        """

        enabled_nodes = []
        attacker = self.attack_graph.get_attacker_by_id(agent.attacker_id)

        for node in nodes:
            logger.info(
                'Attacker agent "%s" stepping through "%s"(%d).',
                agent.name, node.full_name, node.id
            )

            # Compromise node if possible
            if query.is_node_traversable_by_attacker(node, attacker):
                attacker.compromise(node)
                node_reward = node.extras.get("reward", 0)
                agent.reward += node_reward
                for d_agent in self._get_defender_agents():
                    d_agent.reward -= node_reward
                enabled_nodes.append(node)

                logger.info(
                    'Attacker agent "%s" compromised "%s"(%d).',
                    agent.name, node.full_name, node.id
                )

                # Update attacker action surface
                query.update_attack_surface_add_nodes(
                    attacker, agent.action_surface, [node])
            else:
                logger.warning("Attacker could not compromise %s",
                            node.full_name)

        # Terminate attacker if it has nothing left to do
        terminate = True
        for node in agent.action_surface:
            if not node.is_compromised_by(attacker):
                terminate = False
                break
        agent.terminated = terminate

        return enabled_nodes

    def _defender_step(
            self, agent: MalSimDefender, nodes: list[AttackGraphNode]
        ) -> tuple[list[AttackGraphNode], list[AttackGraphNode]]:
        """Enable defense step nodes with defender

        Args:
        agent - the agent to activate defense nodes with
        nodes - the defense step nodes to enable

        Returns: list of defense steps nodes that were enabled
        """

        attack_steps_made_unviable = []
        enabled_defenses = []

        for node in nodes:
            logger.info(
                'Defender agent "%s" stepping through "%s"(%d).',
                agent.name, node.full_name, node.id
            )

            if node not in agent.action_surface:
                logger.warning(
                    'Defender agent "%s" tried to step through "%s"(%d).'
                    'which is not part of its defense surface. Defender '
                    'step will skip', agent, node.full_name, node.id
                )
                continue

            # Enable defense if possible
            if node.is_available_defense():
                node.defense_status = 1.0
                node.is_viable = False
                attack_steps_made_unviable += \
                    apriori.propagate_viability_from_unviable_node(node)
                agent.reward -= node.extras.get("reward", 0)
                enabled_defenses.append(node)
                logger.info(
                    'Defender agent "%s" enabled "%s"(%d).',
                    agent.name, node.full_name, node.id
                )


        for node in enabled_defenses:
            # Remove enabled defenses from defender action surfaces
            for defender_agent in self._get_defender_agents():
                try:
                    defender_agent.action_surface.remove(node)
                except ValueError:
                    pass

        for attack_step in attack_steps_made_unviable:
            # Remove unviable attack steps from attacker action surfaces
            for attacker_agent in self._get_attacker_agents():
                try:
                    attacker_agent.action_surface.remove(attack_step)
                except ValueError:
                    pass

        if self.sim_settings.uncompromise_untraversable_steps:
            self._disable_attack_steps(attack_steps_made_unviable)

        return enabled_defenses, attack_steps_made_unviable

    def step(
            self, actions: dict[str, list[AttackGraphNode]]
        ) -> tuple[list[AttackGraphNode], list[AttackGraphNode]]:
        """Take a step in the simulation

        Args:
        actions - a dict mapping agent name to agent action which
                  contains the actions for that user.

        Returns:
        - (enabled_nodes, disabled_nodes)
        """
        logger.debug(
            "Stepping through iteration %d/%d", self.cur_iter, self.max_iter)
        logger.debug("Performing actions: %s", actions)

        enabled_nodes = []
        disabled_nodes = []
        all_attackers_terminated = True

        # Peform agent actions
        # Note: by design, defenders perform actions
        # before attackers (see _register_agent)
        for agent_name in self.agents:
            agent = self._agents_dict[agent_name]
            agent_actions = actions.get(agent_name, [])
            match agent.type:

                case AgentType.ATTACKER:
                    enabled = self._attacker_step(agent, agent_actions)
                    enabled_nodes += enabled
                    if not agent.terminated:
                        all_attackers_terminated = False

                case AgentType.DEFENDER:
                    enabled, disabled = \
                        self._defender_step(agent, agent_actions)
                    enabled_nodes += enabled
                    disabled_nodes += disabled

                case _:
                    logger.error(
                        'Agent %s has unknown type: %s',
                        agent.name, agent.type
                    )

        if all_attackers_terminated:
            # Terminate all agents if all attackers are terminated
            logger.info("All attackers are terminated")
            for agent in self._agents_dict.values():
                agent.terminated = True

        if self.cur_iter >= self.max_iter:
            # Truncate all agents when max iter is reached
            logger.info("Max iteration reached - all agents truncated")
            for agent in self._agents_dict.values():
                agent.truncated = True

        agents_to_remove = set()
        for agent_name in self.agents:
            agent = self._agents_dict[agent_name]
            if agent.terminated or agent.truncated:
                logger.info("Removing agent %s", agent.name)
                agents_to_remove.add(agent.name)

        for agent in agents_to_remove:
            self.agents.remove(agent)

        self.cur_iter += 1
        return enabled_nodes, disabled_nodes

    def render(self):
        """Render attack graph from simulation in Neo4J"""
        logger.debug("Sending attack graph to Neo4J database.")
        neo4j.ingest_attack_graph(
            self.attack_graph,
            neo4j_configs["uri"],
            neo4j_configs["username"],
            neo4j_configs["password"],
            neo4j_configs["dbname"],
            delete=True,
        )
