from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

from maltoolbox import neo4j_configs
from maltoolbox.ingestors import neo4j
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode, query
from maltoolbox.attackgraph.analyzers import apriori

ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode


class AgentType(Enum):
    """Enum for agent types"""
    ATTACKER = 'attacker'
    DEFENDER = 'defender'


@dataclass
class MalSimAgentState:
    """Stores the state of an agent in the simulator"""

    # Identifier of the agent, used in MalSimulator for lookup
    name: str
    type: AgentType

    # Contains current agent reward in the simulation
    # Attackers get positive rewards, defenders negative
    reward: int = 0

    # Contains possible actions for the agent on the next step
    action_surface: set[AttackGraphNode] = field(default_factory=set)

    # Contains possible actions that became available on the last step
    step_action_surface: set[AttackGraphNode] = field(default_factory=set)

    # Contains the steps performed successfully in the last step
    step_compromised_nodes: set[AttackGraphNode] = field(default_factory=set)

    # Contains nodes that defender actions made unviable in the last step
    step_disabled_nodes: set[AttackGraphNode] = field(default_factory=set)

    # Fields that tell if agent is 'dead' / disabled
    truncated: bool = False
    terminated: bool = False


class MalSimAttackerState(MalSimAgentState):
    """Stores the state of an attacker in the simulator"""

    def __init__(self, name: str, attacker_id: int):
        super().__init__(name, AgentType.ATTACKER)
        self.attacker_id = attacker_id


class MalSimDefenderState(MalSimAgentState):
    """Stores the state of a defender in the simulator"""

    # Contains defense steps successfully enabled by the defender in the last step
    step_enabled_defenses: set[AttackGraphNode] = field(default_factory=set)

    def __init__(self, name: str):
        super().__init__(name, AgentType.DEFENDER)


# Generic T is used here to allow IDEs to provide autocompletions from
# MalSimAgentState.
T = TypeVar("T", bound=MalSimAgentState)


class MalSimAgentStateView(Generic[T]):
    """Read-only interface to MalSimAgentState."""

    _frozen = False

    def __init__(self, agent: T):
        self._agent = agent
        self._frozen = True

    def __setattr__(self, key, value) -> None:
        if self._frozen:
            raise AttributeError("Cannot modify agent state view")
        self.__dict__[key] = value

    def __delattr__(self, key) -> None:
        if self._frozen:
            raise AttributeError("Cannot modify agent state view")
        super().__delattr__(key)

    def __getattr__(self, attr) -> Any:
        """Return read-only version of proxied agent's properties."""
        value = getattr(self._agent, attr)

        if isinstance(value, dict):
            return MappingProxyType(value)
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, set):
            return frozenset(value)

        return value

    def __dir__(self):
        """Dynamically resolve attribute names for REPL autocompletion."""
        return list(vars(self._agent).keys()) + ["_agent", "_frozen"]


@dataclass
class MalSimulatorSettings():
    """Contains settings used in MalSimulator"""

    # uncompromise_untraversable_steps
    # - Uncompromise (evict attacker) from nodes/steps that are no longer
    #   traversable (often because a defense kicked in) if set to True
    # otherwise:
    # - Leave the node/step compromised even after it becomes untraversable
    uncompromise_untraversable_steps: bool = False

    # cumulative_defender_obs
    # - Defender sees the status of the whole attack graph if set to True
    # otherwise:
    # - Defender only sees the status of nodes changed in the current step
    cumulative_defender_obs: bool = True


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
        self.max_iter = max_iter  # Max iterations before stopping simulation
        self.cur_iter = 0        # Keep track on current iteration

        # Keep track on all registered agent states
        self.agents: dict[str, MalSimAgentState] = {}

        # Keep track on all 'living' agents sorted by order to step in
        self.alive_agents: dict[str, MalSimAgentState] = {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> dict[str, MalSimAgentStateView]:
        """Reset attack graph, iteration and reinitialize agents"""

        logger.info("Resetting MAL Simulator.")
        # Reset attack graph
        self.attack_graph = copy.deepcopy(self.attack_graph_backup)
        # Reset current iteration
        self.cur_iter = 0
        # Reset agents
        self._reset_agents()

        return self.agent_states

    def _init_agent_rewards(self):
        """Give rewards for pre-enabled attack/defense steps"""
        for node in self.attack_graph.nodes.values():
            node_reward = node.extras.get('reward', 0)
            if not node_reward:
                continue

            for agent in self.agents.values():

                if agent.type == AgentType.ATTACKER:
                    attacker = self.attack_graph.attackers[agent.attacker_id]

                    if node.is_compromised_by(attacker):
                        # Attacker is rewarded for pre enabled attack steps
                        agent.reward += node_reward

                elif agent.type == AgentType.DEFENDER:
                    # Defender is penalized for all pre-enabled steps
                    if node.is_compromised() or node.is_enabled_defense():
                        agent.reward -= node_reward

    def _init_agent_action_surfaces(self):
        """Set agent action surfaces according to current state"""
        for agent in self.agents.values():
            if agent.type == AgentType.ATTACKER:
                attacker = self.attack_graph.attackers[agent.attacker_id]
                agent.action_surface = query.calculate_attack_surface(attacker)

            elif agent.type == AgentType.DEFENDER:
                agent.action_surface = \
                    query.get_defense_surface(self.attack_graph)
            else:
                raise LookupError(f"Agent type {agent.type} not supported")

    def _reset_agents(self):
        """Reset agent rewards and action surfaces"""

        self.alive_agents = dict(sorted(
            self.agents.items(),
            key=lambda agent: agent[1].type == AgentType.ATTACKER,
        ))

        for agent in self.agents.values():
            # Reset agent reward
            agent.reward = 0

            # Mark agent as alive again
            agent.terminated = False
            agent.truncated = False

        self._init_agent_rewards()
        self._init_agent_action_surfaces()

    def _register_agent(self, agent: MalSimAgentState):
        """Register a mal sim agent"""

        logger.info('Registering agent "%s".', agent)
        assert agent.name not in self.agents, \
            f"Duplicate agent named {agent.name} not allowed"

        self.agents[agent.name] = agent

        # Defender is first in list so it can pick
        # actions before attacker when step performed
        self.alive_agents = dict(sorted(
            self.agents.items(),
            key=lambda agent: agent[1].type == AgentType.ATTACKER,
        ))

    def register_attacker(self, name: str, attacker_id: int):
        """Register a mal sim attacker agent"""
        agent_state = MalSimAttackerState(name, attacker_id)
        self._register_agent(agent_state)

    def register_defender(self, name: str):
        """Register a mal sim defender agent"""
        agent_state = MalSimDefenderState(name)
        self._register_agent(agent_state)

    @property
    def agent_states(self) -> list[MalSimAgentStateView]:
        """Return read only agent state for all dead and alive agents"""
        return {name: MalSimAgentStateView(agent) for name, agent in self.agents.items()}

    def _get_attacker_agents(self) -> list[MalSimAttackerState]:
        """Return list of mutable attacker agent states"""
        return [a for a in self.agents.values() if a.type == AgentType.ATTACKER]

    def _get_defender_agents(self) -> list[MalSimDefenderState]:
        """Return list of mutable defender agent states"""
        return [a for a in self.agents.values() if a.type == AgentType.DEFENDER]

    def _disable_attack_steps(
        self, attack_steps_to_disable: set[AttackGraphNode]
    ):
        """Disable nodes for each attacker agent

        For each compromised attack step uncompromise the node, disable its
        observed_state, and remove the rewards.
        """
        for attacker_agent in self._get_attacker_agents():
            attacker = self.attack_graph.attackers[attacker_agent.attacker_id]

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
        self, agent: MalSimAttackerState, nodes: list[AttackGraphNode]
    ) -> tuple[set[AttackGraphNode], set[AttackGraphNode]]:
        """Compromise attack step nodes with attacker

        Args:
        agent - the agent to compromise nodes with
        nodes - the nodes to compromise

        Returns: tuple of set of attack steps nodes that were compromised and
                 the new nodes that became available in the attack surface
        """

        compromised_nodes = set()
        attacker = self.attack_graph.attackers[agent.attacker_id]

        for node in nodes:
            assert node == self.attack_graph.nodes[node.id], (
                f"{agent.name} tried to enable a node that is not part "
                "of this simulators attack_graph. Make sure the node "
                "comes from the agents action surface."
            )

            logger.info(
                'Attacker agent "%s" stepping through "%s"(%d).',
                agent.name, node.full_name, node.id
            )

            # Compromise node if possible
            if query.is_node_traversable_by_attacker(node, attacker) \
                    and node in agent.action_surface:
                attacker.compromise(node)
                node_reward = node.extras.get("reward", 0)
                agent.reward += node_reward
                for d_agent in self._get_defender_agents():
                    d_agent.reward -= node_reward
                compromised_nodes.add(node)

                logger.info(
                    'Attacker agent "%s" compromised "%s"(%d).',
                    agent.name, node.full_name, node.id
                )
            else:
                logger.warning("Attacker could not compromise %s",
                               node.full_name)

        # Update attacker action surface
        new_attack_surface = query.calculate_attack_surface(
            attacker, from_nodes=compromised_nodes, skip_compromised=True
        )

        agent.action_surface |= new_attack_surface

        # Terminate attacker if it has nothing left to do
        terminate = True
        for node in agent.action_surface:
            if not node.is_compromised_by(attacker):
                terminate = False
                break
        agent.terminated = terminate

        return compromised_nodes, new_attack_surface

    def _defender_step(
        self, agent: MalSimDefenderState, nodes: list[AttackGraphNode]
    ) -> tuple[set[AttackGraphNode], set[AttackGraphNode]]:
        """Enable defense step nodes with defender.

        Args:
        agent - the agent to activate defense nodes with
        nodes - the defense step nodes to enable

        Returns: tuple of defense steps that were enabled and attack steps that
                 became unviable due to enabled defenses
        """

        enabled_defenses = set()
        attack_steps_made_unviable = set()

        for node in nodes:
            assert node == self.attack_graph.nodes[node.id], (
                f"{agent.name} tried to enable a node that is not part "
                "of this simulators attack_graph. Make sure the node "
                "comes from the agents action surface."
            )
            logger.info(
                'Defender agent "%s" stepping through "%s"(%d).',
                agent.name, node.full_name, node.id
            )

            if node not in agent.action_surface:
                logger.warning(
                    'Defender agent "%s" tried to step through "%s"(%d).'
                    'which is not part of its defense surface. Defender '
                    'step will skip', agent.name, node.full_name, node.id
                )
                continue

            # Enable defense if possible
            if node.is_available_defense():
                node.defense_status = 1.0
                node.is_viable = False
                attack_steps_made_unviable |= \
                    apriori.propagate_viability_from_unviable_node(node)
                agent.reward -= node.extras.get("reward", 0)
                enabled_defenses.add(node)
                logger.info(
                    'Defender agent "%s" enabled "%s"(%d).',
                    agent.name, node.full_name, node.id
                )

        for node in enabled_defenses:
            # Remove enabled defenses from defender action surfaces
            for defender_agent in self._get_defender_agents():
                defender_agent.action_surface.discard(node)

        for attack_step in attack_steps_made_unviable:
            # Remove unviable attack steps from attacker action surfaces
            for attacker_agent in self._get_attacker_agents():
                attacker_agent.action_surface.discard(attack_step)

        if self.sim_settings.uncompromise_untraversable_steps:
            self._disable_attack_steps(attack_steps_made_unviable)

        return enabled_defenses, attack_steps_made_unviable

    def step(
        self, actions: dict[str, list[AttackGraphNode]]
    ) -> dict[str, MalSimAgentStateView]:
        """Take a step in the simulation

        Args:
        actions - a dict mapping agent name to agent action which
                  contains the actions for that user.

        Returns:
        - (compromised_nodes, new_attack_surface, disabled_nodes)
        """
        logger.debug("Stepping through iteration %d/%d", self.cur_iter, self.max_iter)
        logger.debug("Performing actions: %s", actions)

        disabled_nodes = set()
        all_performed = set()

        all_attackers_terminated = True

        # Perform agent actions
        # Note: by design, defenders perform actions
        # before attackers (see _register_agent)
        for agent in self.alive_agents.values():
            agent_actions = actions.get(agent.name, [])

            agent.step_disabled_nodes = disabled_nodes

            match agent.type:
                case AgentType.ATTACKER:
                    performed_steps, new_surface = self._attacker_step(
                        agent, agent_actions
                    )
                    agent.step_action_surface = new_surface
                    agent.step_compromised_nodes = performed_steps
                    all_performed |= performed_steps
                    if not agent.terminated:
                        all_attackers_terminated = False

                case AgentType.DEFENDER:
                    enabled_defenses, disabled = \
                        self._defender_step(agent, agent_actions)
                    agent.step_compromised_nodes = all_performed
                    agent.step_enabled_defenses = enabled_defenses
                    disabled_nodes |= disabled

                case _:
                    logger.error(
                        'Agent %s has unknown type: %s',
                        agent.name, agent.type
                    )

        if all_attackers_terminated:
            # Terminate all defenders if all attackers are terminated
            logger.info("All attackers are terminated")
            for agent in self.agents.values():
                agent.terminated = True

        if self.cur_iter >= self.max_iter:
            # Truncate all agents when max iter is reached
            logger.info("Max iteration reached - all agents truncated")
            for agent in self.agents.values():
                agent.truncated = True

        for agent in self.alive_agents.copy().values():
            if agent.terminated or agent.truncated:
                logger.info("Removing agent %s", agent.name)
                del self.alive_agents[agent.name]

        self.cur_iter += 1

        return self.agent_states

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
