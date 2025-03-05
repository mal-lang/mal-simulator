from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
from maltoolbox import neo4j_configs
from maltoolbox.ingestors import neo4j
from maltoolbox.attackgraph import (AttackGraph, AttackGraphNode,
    Attacker, query)
from maltoolbox.attackgraph.analyzers import apriori

ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)


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

    # Contains possible actions for the agent in the next step
    action_surface: set[AttackGraphNode] = field(default_factory=set)

    # Contains all nodes that this agent has performed successfully
    performed_nodes: set[AttackGraphNode] = field(default_factory=set)

    # Contains the steps performed successfully in the last step
    step_performed_nodes: set[AttackGraphNode] = field(default_factory=set)

    # Contains possible actions that became available in the last step
    step_action_surface_additions: set[AttackGraphNode] = (
        field(default_factory = set))

    # Contains previously possible actions that became unavailable in the last
    # step
    step_action_surface_removals: set[AttackGraphNode] = (
        field(default_factory = set))

    # Contains nodes that defender actions made unviable in the last step
    step_unviable_nodes: set[AttackGraphNode] = field(default_factory=set)

    # Current ttc value of each node that had ttc functions defined
    ttc_values: dict[AttackGraphNode, int] = field(default_factory=dict)

    # Fields that tell if the agent is done or stopped
    truncated: bool = False
    terminated: bool = False


class MalSimAttackerState(MalSimAgentState):
    """Stores the state of an attacker in the simulator"""

    def __init__(self, name: str, attacker: Attacker):
        super().__init__(name, AgentType.ATTACKER)
        self.attacker = attacker


class MalSimDefenderState(MalSimAgentState):
    """Stores the state of a defender in the simulator"""

    # Contains the steps performed successfully by all of the attacker agents
    # in the last step
    step_all_compromised_nodes: set[AttackGraphNode] = set()

    def __init__(self, name: str):
        super().__init__(name, AgentType.DEFENDER)


class MalSimAgentStateView(MalSimAttackerState, MalSimDefenderState):
    """Read-only interface to MalSimAgentState.

    Subclassing from State classes only to satisfy static analysis and support
    autocompletion that would otherwise be unavailable due to the dynamic
    __getattr(ibute)__ method.
    """

    _frozen = False

    def __init__(self, agent):
        self._agent = agent
        self._frozen = True

    def __setattr__(self, key, value) -> None:
        if self._frozen:
            raise AttributeError("Cannot modify agent state view")
        super().__setattr__(key, value)

    def __delattr__(self, key) -> None:
        if self._frozen:
            raise AttributeError("Cannot modify agent state view")
        super().__delattr__(key)

    def __getattribute__(self, attr) -> Any:
        """Return read-only version of proxied agent's properties.

        If the attribute exists in the View only return it from there. Using
        __getattribute__ instead of __getattr__ as the latter is called only for
        missing properties; since this class is a subclass of State, it will
        have all state properties and those would be returned from self, not
        self._agent. Using __getattribute__ allows use to filter which
        properties to return from self and which from self._agent.
        """
        if attr in ("_agent", "_frozen") or \
                attr.startswith("__") and attr.endswith("__"):
            return super().__getattribute__(attr)

        agent = super().__getattribute__('_agent')
        value = getattr(agent, attr)

        if attr == '_agent':
            return agent

        value = getattr(agent, attr)

        if isinstance(value, dict):
            return MappingProxyType(value)
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, set):
            return frozenset(value)

        return value

    def __dir__(self):
        """Dynamically resolve attribute names for REPL autocompletion."""
        dunder_attrs = [
            attr
            for attr in dir(self.__class__)
            if attr.startswith("__") and attr.endswith("__")
        ]

        props = list(vars(self._agent).keys()) + ["_agent", "_frozen"]

        return dunder_attrs + props


@dataclass
class MalSimulatorSettings():
    """Contains settings used in MalSimulator"""

    # uncompromise_untraversable_steps
    # - Uncompromise (evict attacker) from nodes/steps that are no longer
    #   traversable (often because a defense kicked in) if set to True
    # otherwise:
    # - Leave the node/step compromised even after it becomes untraversable
    uncompromise_untraversable_steps: bool = False


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

        # All internal agent states (dead or alive)
        self._agent_states: dict[str, MalSimAgentState] = {}

        # Keep track on all 'living' agents sorted by order to step in
        self._alive_agents: set[str] = set()

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
        self._reset_agents(seed=seed)

        return self.agent_states

    def _init_agent_action_surfaces(self):
        """Set agent action surfaces according to current state"""
        for agent in self._agent_states.values():

            if isinstance(agent, MalSimAttackerState):
                attacker = agent.attacker
                agent.action_surface = query.calculate_attack_surface(
                    attacker, skip_compromised = True
                )

            elif isinstance(agent, MalSimDefenderState):
                agent.action_surface = \
                    query.get_defense_surface(self.attack_graph)
            else:
                raise LookupError(f"Agent type {agent.type} not supported")

    def _sample_ttc(self, node: AttackGraphNode):
        """Sample a value from ttc distribution for a node"""

        def process_sample(distribution):
            max_cost = 500
            # Generate a random sample for the given distribution
            if 'Bernoulli' in distribution:
                # Mixture of exponential and constant distribution
                prob = distribution['Bernoulli']
                scale = distribution['Exponential']
                scale = 1/scale
                sample = (
                    np.random.exponential(scale=scale)
                    if np.random.choice([0, 1], p=[prob, 1 - prob])
                    else max_cost
                )
            else:
                # Pure exponential distribution
                scale = distribution['Exponential']
                scale = 1/scale
                sample = np.random.exponential(scale=scale)

            return sample

        distribution = node.ttc['name']
        sample = 0
        if distribution == "EasyAndCertain":
            # Generate sample for EasyAndCertain distribution.
            sample = process_sample({'Exponential': 1})
        elif distribution == "EasyAndUncertain":
            # Generate sample for EasyAndUncertain distribution.
            sample = process_sample({'Exponential': 1, 'Bernoulli': 0.5})
        elif distribution == "HardAndCertain":
            # Generate sample for HardAndCertain distribution.
            sample = process_sample({'Exponential': 0.1})
        elif distribution == "HardAndUncertain":
            # Generate sample for HardAndUncertain distribution.
            sample = process_sample({'Exponential': 0.1, 'Bernoulli': 0.5})
        elif distribution == "VeryHardAndCertain":
            # Generate sample for VeryHardAndCertain distribution.
            sample = process_sample({'Exponential': 0.01})
        elif distribution == "VeryHardAndUncertain":
            # Generate sample for VeryHardAndUncertain distribution.
            sample = process_sample({'Exponential': 0.01, 'Bernoulli': 0.5})
        elif distribution == "Exponential":
            # Generate sample for custom Exponential distribution.
            scale = float(node.ttc['arguments'][0])
            sample = process_sample({'Exponential': scale})

        return sample

    def _init_agent_ttcs(self):
        """Sample ttcs for all attacker agents"""
        for node in self.attack_graph.nodes.values():
            if node.ttc is None:
                continue
            for agent in self._get_attacker_agents():
                agent.ttc_values[node] = self._sample_ttc(node)

    def _reset_agents(self, seed=None):
        """Reset agent rewards and action surfaces"""

        # Revive all agents
        self._alive_agents = set(self._agent_states.keys())
        pre_enabled_attack_steps = set()

        for agent_state in self._get_attacker_agents():
            # Create a new agent state for the attacker
            attacker = self.attack_graph.attackers[agent_state.attacker.id]
            agent_state = MalSimAttackerState(agent_state.name, attacker)

            # Initial sets with pre enabled nodes
            agent_state.step_performed_nodes = attacker.reached_attack_steps
            agent_state.performed_nodes = attacker.reached_attack_steps
            pre_enabled_attack_steps |= attacker.reached_attack_steps
            agent_state.reward = self._attacker_reward(agent_state)

            self._agent_states[agent_state.name] = agent_state

        pre_enabled_defenses = set(
            node for node in self.attack_graph.nodes.values()
            if node.is_enabled_defense()
        )

        for agent_state in self._get_defender_agents():
            # Create a new agent state for the defender
            agent_state = MalSimDefenderState(agent_state.name)

            # Initial sets with pre enabled nodes
            agent_state.step_performed_nodes = pre_enabled_defenses
            agent_state.performed_nodes = pre_enabled_defenses
            agent_state.step_all_compromised_nodes = pre_enabled_attack_steps
            agent_state.reward = self._defender_reward(agent_state)

            self._agent_states[agent_state.name] = agent_state

        if seed:
            np.random.seed(seed)

        self._init_agent_action_surfaces()
        self._init_agent_ttcs()

    def register_attacker(self, name: str, attacker_id: int):
        """Register a mal sim attacker agent"""
        assert name not in self._agent_states, \
            f"Duplicate agent named {name} not allowed"

        attacker = self.attack_graph.attackers.get(attacker_id, None)
        if attacker is None:
            msg = ('Failed to register Attacker agent because no attacker '
                f'with id {attacker_id} was found in the attack ' 'graph!')
            logger.error(msg)
            raise ValueError(msg)
        agent_state = MalSimAttackerState(name, attacker)
        self._agent_states[name] = agent_state
        self._alive_agents.add(name)

    def register_defender(self, name: str):
        """Register a mal sim defender agent"""
        assert name not in self._agent_states, \
            f"Duplicate agent named {name} not allowed"

        agent_state = MalSimDefenderState(name)
        self._agent_states[name] = agent_state
        self._alive_agents.add(name)

    @property
    def agent_states(self) -> dict[str, MalSimAgentStateView]:
        """Return read only agent state for all dead and alive agents"""
        return {
            name: MalSimAgentStateView(agent)
            for name, agent in self._agent_states.items()
        }

    def _get_attacker_agents(self) -> list[MalSimAttackerState]:
        """Return list of mutable attacker agent states of alive attackers"""
        return [
            a for a in self._agent_states.values()
            if a.name in self._alive_agents
            and isinstance(a, MalSimAttackerState)
        ]

    def _get_defender_agents(self) -> list[MalSimDefenderState]:
        """Return list of mutable defender agent states of alive defenders"""
        return [
            a for a in self._agent_states.values()
            if a.name in self._alive_agents
            and isinstance(a, MalSimDefenderState)
        ]

    def _uncompromise_attack_steps(
        self, attack_steps_to_uncompromise: set[AttackGraphNode]
    ):
        """Uncompromise nodes for each attacker agent

        Go through the nodes in `attack_steps_to_uncompromise` for each
        attacker agent. If a node is compromised by the attacker agent:
            - Uncompromise the node and remove rewards for it.
        """
        for attacker_agent in self._get_attacker_agents():
            attacker = attacker_agent.attacker

            for unviable_node in attack_steps_to_uncompromise:
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
    ):
        """Compromise attack step nodes with attacker

        Args:
        agent - the agent to compromise nodes with
        nodes - the nodes to compromise
        """

        compromised_nodes = set()
        attacker = agent.attacker

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

                if node in agent.ttc_values:
                    logger.info(
                        'Attacker decreased ttc value of '
                        'step %s from %d to %d', node.full_name,
                        agent.ttc_values[node], agent.ttc_values[node] - 1
                    )
                    agent.ttc_values[node] -= 1
                    if agent.ttc_values[node] > 0:
                        # Attacker can not compromise before
                        # ttc is 0 or lower
                        continue

                attacker.compromise(node)
                agent.performed_nodes.add(node)
                compromised_nodes.add(node)

                logger.info(
                    'Attacker agent "%s" compromised "%s"(%d).',
                    agent.name, node.full_name, node.id
                )
            else:
                logger.warning("Attacker could not compromise %s",
                               node.full_name)

        # Update attacker action surface
        attack_surface_additions = query.calculate_attack_surface(
            attacker, from_nodes = compromised_nodes, skip_compromised = True
        )

        # Filter out nodes already in action surface, these are not additions.
        attack_surface_additions -= agent.action_surface
        # Also filter out nodes already compromised from the attack surface.
        # TODO: Make this configurable in the future
        agent.action_surface -= compromised_nodes
        agent.step_action_surface_removals |= compromised_nodes

        agent.step_action_surface_additions = attack_surface_additions
        agent.action_surface |= attack_surface_additions
        agent.step_performed_nodes = compromised_nodes

        return compromised_nodes

    def _defender_step(
        self, agent: MalSimDefenderState, nodes: list[AttackGraphNode]
    ):
        """Enable defense step nodes with defender.

        Args:
        agent - the agent to activate defense nodes with
        nodes - the defense step nodes to enable

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
                    'Defender agent "%s" tried to step through "%s"(%d), '
                    'which is not part of its defense surface. Defender '
                    'step will skip!', agent.name, node.full_name, node.id
                )
                continue

            # Enable defense if possible
            if node.is_available_defense():
                node.defense_status = 1.0
                node.is_viable = False
                attack_steps_made_unviable |= \
                    apriori.propagate_viability_from_unviable_node(node)
                enabled_defenses.add(node)
                agent.performed_nodes.add(node)
                logger.info(
                    'Defender agent "%s" enabled "%s"(%d).',
                    agent.name, node.full_name, node.id
                )

        agent.step_performed_nodes = enabled_defenses
        agent.step_unviable_nodes |= attack_steps_made_unviable

        for defender_agent in self._get_defender_agents():
            # Remove enabled defenses from all defenders action surface
            defender_agent.step_action_surface_removals |= enabled_defenses
            defender_agent.action_surface -= enabled_defenses

        for attacker_agent in self._get_attacker_agents():
            # Remove attack steps made unviable from all attackers
            # action surfaces if they were part of it
            attacker_agent.step_action_surface_removals |= (
                attacker_agent.action_surface & attack_steps_made_unviable
            )
            attacker_agent.action_surface -= attack_steps_made_unviable

        if self.sim_settings.uncompromise_untraversable_steps:
            self._uncompromise_attack_steps(attack_steps_made_unviable)

        return enabled_defenses, attack_steps_made_unviable

    @staticmethod
    def _attacker_reward(attacker_state: MalSimAttackerState):
        """
        Calculate current attacker reward by adding this steps
        compromised node rewards to the previous attacker reward.
        Can be overriden by subclass to implement custom reward function.

        Args:
        - attacker_state: the attacker state before nodes were compromised
        - step_agent_compromised_nodes: set of nodes compromised
          since last reward was calculated
        """
        # Attacker is rewarded for compromised nodes
        return attacker_state.reward + sum(
            n.extras.get("reward", 0)
            for n in attacker_state.step_performed_nodes
        )

    @staticmethod
    def _defender_reward(defender_state: MalSimDefenderState):
        """
        Calculate current defender reward by subtracting this steps
        compromised/enabled node rewards from the previous defender reward.
        Can be overriden by subclass to implement custom reward function.

        Args:
        - defender_state: the defender state before defenses were enabled
        - step_enabled_defenses: set of defenses enabled since last reward was
          calculated
        """
        # Defender is penalized for compromised nodes and enabled defenses
        step_enabled_defenses = defender_state.step_performed_nodes
        step_compromised_nodes = defender_state.step_all_compromised_nodes
        return defender_state.reward - sum(
            n.extras.get("reward", 0)
            for n in step_enabled_defenses | step_compromised_nodes
        )

    @staticmethod
    def _attacker_is_terminated(attacker_state: MalSimAttackerState) -> bool:
        """Check if attacker is terminated
        Can be overriden by subclass for custom termination condition.

        Args:
        - attacker_state: the attacker state to check for termination
        """
        # Attacker is terminated if it has no more actions to take
        return len(attacker_state.action_surface) == 0

    @staticmethod
    def _defender_is_terminated(
        attacker_agent_states: list[MalSimAttackerState]
    ) -> bool:
        """Check if defender is terminated
        Can be overriden by subclass for custom termination condition.

        Args:
        - defender_state: the defender state to check for termination
        """
        # Defender is terminated if all attackers are terminated
        return all(a.terminated for a in attacker_agent_states)

    def step(
        self, actions: dict[str, list[AttackGraphNode]]
    ) -> dict[str, MalSimAgentStateView]:
        """Take a step in the simulation

        Args:
        actions - a dict mapping agent name to agent action which
                  contains the actions for that user.

        Returns:
        - A dictionary containing the agent state views keyed by agent names
        """
        logger.info(
            "Stepping through iteration %d/%d", self.cur_iter, self.max_iter
        )
        logger.debug("Performing actions: %s", actions)

        # Populate these from the results for all agents' actions.
        step_compromised_nodes = set()
        step_enabled_defenses = set()
        step_unviable_nodes = set()

        # Prepare agent states for new step
        for agent_name in self._alive_agents:
            agent = self._agent_states[agent_name]
            # Clear action surface removals from previous step to
            # make sure the old values do not carry over.
            agent.step_action_surface_removals = set()
            agent.step_unviable_nodes = step_unviable_nodes

        # Perform defender actions first
        for defender_state in self._get_defender_agents():
            enabled, unviable = self._defender_step(
                defender_state, actions.get(defender_state.name, [])
            )
            step_enabled_defenses |= enabled
            step_unviable_nodes |= unviable

        # Perform attacker actions afterwards
        for attacker_state in self._get_attacker_agents():
            compromised = self._attacker_step(
                attacker_state, actions.get(attacker_state.name, [])
            )
            step_compromised_nodes |= compromised

            # Calculate attacker reward and termination
            attacker_state.reward = self._attacker_reward(attacker_state)
            attacker_state.truncated = self.cur_iter >= self.max_iter
            attacker_state.terminated = (
                self._attacker_is_terminated(attacker_state)
            )

        # Calculate defender rewards and terminations
        # (depends on attackers, so must be done after all steps)
        for defender_state in self._get_defender_agents():
            defender_state.step_all_compromised_nodes = step_compromised_nodes
            defender_state.reward = self._defender_reward(defender_state)
            defender_state.truncated = self.cur_iter >= self.max_iter
            defender_state.terminated = self._defender_is_terminated(
                self._get_attacker_agents()
            )

        # Remove agents that are terminated or truncated
        for agent_name in self._alive_agents.copy():
            agent_state = self._agent_states[agent_name]
            agent_state.truncated = self.cur_iter >= self.max_iter
            if agent_state.terminated or agent_state.truncated:
                logger.info("Removing agent %s", agent_name)
                self._alive_agents.remove(agent_name)

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
