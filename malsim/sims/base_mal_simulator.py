from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from enum import Enum
from typing import Optional

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from maltoolbox.attackgraph.analyzers import apriori
from maltoolbox.attackgraph import query

from .mal_simulator_settings import MalSimulatorSettings

ITERATIONS_LIMIT = int(1e9)

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Enum for agent types"""
    ATTACKER = 'attacker'
    DEFENDER = 'defender'

@dataclass
class SimulatorAgent():
    """Stores the state of an agent in the simulator"""
    name: str
    type: AgentType

    observation: dict = field(default_factory=lambda: {})
    action_surface: list = field(default_factory=lambda: [])
    reward: int = 0
    done: bool = False
    attacker_id: Optional[int] = None # Only for attacker agent


class ActionState(Enum):
    """The state of an agent action."""
    WAIT = 0  # Agent is waiting, not acting on any node
    ACT = 1   # Agent is acting on a specific node

class BaseMalSimulator():
    """A simple MAL Simulator that works on the AttackGraph

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
        self.sim_settings = sim_settings
        self.max_iter = max_iter # Max iterations before stopping simulation
        self.cur_iter = 0        # Keep track on current iteration
        self.agents_dict = {}    # Keep track on registered agents
        self.agents: list[SimulatorAgent] = []

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None
        ) -> dict[str, SimulatorAgent]:
        """Reset attack graph, iteration and reinitialize agents"""

        logger.info("Resetting Base MAL Simulator.")
        # Reset attack graph
        self.attack_graph = copy.deepcopy(self.attack_graph_backup)
        # Reset current iteration
        self.cur_iter = 0
        # Reset agents
        self._reset_agents()

        return self.agents_dict

    def _reset_agents(self):
        """Reset agent rewards and action surfaces"""

        for agent in self.agents:
            # Reset reward
            agent.reward = 0
            agent.observation = {}

        self._init_agent_action_surfaces()

    def _init_agent_action_surfaces(self):
        """Set agent action surfaces according to current state"""
        for agent in self.agents:
            if agent.type == AgentType.ATTACKER:
                attacker_id = agent.attacker_id
                attacker = self.attack_graph.get_attacker_by_id(attacker_id)
                # Get current action surface
                agent.action_surface = \
                    query.get_attack_surface(attacker)

            elif agent.type == AgentType.DEFENDER:
                # Get current action surface
                agent.action_surface = \
                    query.get_defense_surface(self.attack_graph)
            else:
                agent.action_surface = []

    def register_attacker(self, agent_name, attacker_id: int):
        """Register an attacker agent in the simulator
        
        The agent is given a name and an id,
        the id points to an attacker in the attack graph
        """

        logger.info(
            'Register attacker "%s" agent with id %d.',
            agent_name, attacker_id)

        assert agent_name not in self.agents_dict, \
            f"Duplicate attacker agent named {agent_name} not allowed"

        agent = SimulatorAgent(
            agent_name, AgentType.ATTACKER, attacker_id=attacker_id
        )
        self.agents.append(agent)
        self.agents_dict[agent_name] = agent
        return agent

    def register_defender(self, agent_name):
        """Add defender agent to the simulator
        
        Note:
        Defenders will be run first so that the defenses prevent attackers
        appropriately in case any attackers select attack steps that the
        defenders safeguards against during the same step.
        """

        logger.info('Register defender "%s" agent.', agent_name)
        assert agent_name not in self.agents_dict, \
            f"Duplicate defender agent named {agent_name} not allowed"

        # Add defenders at the start of the list to make
        # sure they have priority when performing steps.
        agent = SimulatorAgent(agent_name, AgentType.DEFENDER)
        self.agents.insert(0, agent)
        self.agents_dict[agent_name] = agent
        return agent

    def get_attacker_agents(self) -> SimulatorAgent:
        """Return list of attacker agents"""
        return [a for a in self.agents
                if a.type == AgentType.ATTACKER]

    def get_defender_agents(self) -> SimulatorAgent:
        """Return list of defender agents"""
        return [a for a in self.agents
                if a.type == AgentType.DEFENDER]

    def _attacker_step(
            self, agent: SimulatorAgent, nodes: list[AttackGraphNode]
        ) -> list[AttackGraphNode]:
        """Compromise attack step nodes with attacker

        Args:
        agent - the agent to compromise nodes with
        nodes - the nodes to compromise

        Returns: list of attack steps nodes that were compromised
        """

        enabled_nodes = []
        attacker_id = agent.attacker_id
        attacker = self.attack_graph.get_attacker_by_id(attacker_id)

        for node in nodes:
            # Compromise node if possible
            if query.is_node_traversable_by_attacker(node, attacker):
                attacker.compromise(node)
                agent.reward += node.extras.get("reward", 0)
                enabled_nodes.append(node)

                query.update_attack_surface_add_nodes(
                    attacker, agent.action_surface, [node])
            else:
                logger.info("Attacker could not compromise %s",
                            node.full_name)

        return enabled_nodes

    def _defender_step(
            self, agent: SimulatorAgent, nodes: list[AttackGraphNode]
        ) -> list[AttackGraphNode]:
        """Enable defense step nodes with defender

        Args:
        agent - the agent to activate defense nodes with
        nodes - the defense step nodes to enable

        Returns: list of defense steps nodes that were enabled
        """

        attack_steps_made_unviable = []
        enabled_nodes = []

        for node in nodes:

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
                enabled_nodes.append(node)
                agent.action_surface.remove(node)
                assert node not in agent.action_surface

        for attack_step in attack_steps_made_unviable:
            for attacker_agent in self.get_attacker_agents():
                try:
                    # Node is no longer part of attacker action surface
                    attacker_agent.action_surface.remove(attack_step)
                except ValueError:
                    pass

        return enabled_nodes

    def step(self, actions: dict[str, list[AttackGraphNode]]):
        """Take a step in the simulation

        Args:
        actions - a dict mapping agent name to agent action which
                  contains the actions for that user.

        Returns:
        - state of each agent after step is performed
        """
        logger.debug(
            "Stepping through iteration %d/%d", self.cur_iter, self.max_iter)
        logger.debug("Performing actions: %s", actions)

        # Peform agent actions
        # Note: by design defenders go first, then attackers
        for agent in self.agents:
            agent_actions = actions.get(agent.name, [])

            if not agent_actions:
                # Agent decided to wait, move on.
                continue

            if agent.type == AgentType.ATTACKER:
                self._attacker_step(agent, agent_actions)

            elif agent.type == AgentType.DEFENDER:
                self._defender_step(agent, agent_actions)

            else:
                logger.error('Agent %s has unknown type: %s',
                             agent.name, agent.type)

        self.cur_iter += 1

        return {
            agent.name: agent for agent in self.agents
        }
