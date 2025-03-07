from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
from enum import Enum
from typing import Optional

from maltoolbox import neo4j_configs
from maltoolbox.ingestors import neo4j
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode, query
from maltoolbox.attackgraph.analyzers import apriori

ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enum for agent types"""
    ATTACKER = 'attacker'
    DEFENDER = 'defender'

@dataclass(frozen=True)
class AgentState:
    """Read only state of an agent in the simulator"""
    # Identifier of the agent, used in MalSimulator for lookup
    name: str
    # Contains current agent reward in the simulation
    # Attackers get positive rewards, defenders negative
    reward: int
    # Contains possible actions for the agent in the next step
    action_surface: set[AttackGraphNode]
    # Contains the steps performed successfully in the last step
    step_performed_nodes: set[AttackGraphNode]
    # Contains possible actions that became available in the last step
    step_action_surface_additions: set[AttackGraphNode]
    # Contains actions that became unavailable for the agent in the last step
    step_action_surface_removals: set[AttackGraphNode]
    # Contains nodes that defender actions made unviable in the last step
    step_unviable_nodes: set[AttackGraphNode]
    # Fields that tell if the agent is done or stopped
    truncated: bool
    terminated: bool

@dataclass(frozen=True)
class AttackerState(AgentState):
    """Stores the state of an attacker in the simulator"""
    attacker_id: int
    type: AgentType = AgentType.ATTACKER

@dataclass(frozen=True)
class DefenderState(AgentState):
    """Stores the state of a defender in the simulator"""
    # Contains nodes compromised by any attacker in the last step
    step_all_compromised_nodes: set[AttackGraphNode]
    type: AgentType = AgentType.DEFENDER

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

    Allows user to register agents (defender / attacker)
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
        self.cur_iter = 0         # Keep track on current iteration

        # All internal agent states (dead or alive)
        self.agent_states: dict[str, AgentState] = {}

        # Keep track on all 'living' agents
        self._alive_agents: set[str] = set()

    def _create_defender_state(self, name) -> DefenderState:
        """Build a defender state based on attack graph"""
        defense_surface = query.get_defense_surface(self.attack_graph)
        reward = sum(
            n.extras.get('reward', 0)
            for n in self.attack_graph.nodes.values()
            if n.is_compromised() or n.is_enabled_defense()
        )
        return DefenderState(
            name=name,
            reward=reward,
            action_surface=defense_surface,
            step_performed_nodes=set(),
            step_action_surface_additions=set(),
            step_action_surface_removals=set(),
            step_unviable_nodes=set(),
            step_all_compromised_nodes=set(),
            truncated=False,
            terminated=False,
        )

    def _create_attacker_state(self, name, attacker_id) -> AttackerState:
        """Build an attacker state based on attack graph"""
        attacker = self.attack_graph.attackers.get(attacker_id)
        if not attacker:
            raise LookupError(
                f"Can not find attacker with id {attacker_id} in attack graph"
            )

        attack_surface = (
            query.calculate_attack_surface(attacker, skip_compromised = True)
        )
        reward = sum(
            n.extras.get('reward', 0)
            for n in self.attack_graph.nodes.values()
            if n.is_compromised_by(attacker)
        )
        return AttackerState(
            name=name,
            attacker_id=attacker_id,
            reward=reward,
            action_surface=attack_surface,
            step_performed_nodes=set(),
            step_action_surface_additions=set(),
            step_action_surface_removals=set(),
            step_unviable_nodes=set(),
            truncated=False,
            terminated=False,
        )

    def _create_updated_attacker_state(
            self,
            attacker_state: AttackerState,
            step_compromised_nodes: set[AttackGraphNode],
            step_nodes_made_unviable: set[AttackGraphNode],
            step_uncompromised_nodes: set[AttackGraphNode]
        ) -> AttackerState:
        """
        Build attacker state from previous `attacker_state` with given
        args. This is faster than running _create_attacker_state every time,
        since it only adds the 'diff' to the state.
        It will also set the additional step_* fields for the agent state.

        Args:
            - attacker_state: the old state of the attacker to update
            - step_compromised_nodes: 
                the steps this attacker compromised since last update
            - step_nodes_made_unviable:
                the steps that a defender has made unviable since last update
            - step_uncompromised_nodes:
                steps that were uncompromised
                (see sim_settings.uncompromise_untraversable_steps)
        Returns:
            - an AttackerState based on `attacker_state` updated with the
              given args
        """
        attacker = self.attack_graph.attackers[attacker_state.attacker_id]

        # step action surface additions are attack surface nodes
        # that were not already in the action surface
        step_action_surface_additions = query.calculate_attack_surface(
            attacker, from_nodes=step_compromised_nodes, skip_compromised=True
        ) - attacker_state.action_surface

        # step action surface removals are compromised nodes
        # plus the unvable nodes that were previously in action surface
        step_action_surface_removals = (
            step_compromised_nodes
            | (step_nodes_made_unviable & attacker_state.action_surface)
        )

        # attacker step reward is the sum of compromised node rewards
        # minus the sum of uncompromised node rewards
        step_reward = (
            sum(n.extras.get('reward', 0) for n in step_compromised_nodes)
            - sum(n.extras.get('reward', 0) for n in step_uncompromised_nodes)
        )

        # new agent action surface is old action_surface plus step action
        # surface additions, minus step action surface removals
        new_action_surface = (
            (attacker_state.action_surface | step_action_surface_additions)
            - step_action_surface_removals
        )

        return AttackerState(
            name=attacker_state.name,
            attacker_id=attacker_state.attacker_id,
            reward=attacker_state.reward + step_reward,
            action_surface=new_action_surface,
            step_performed_nodes=step_compromised_nodes,
            step_action_surface_additions=step_action_surface_additions,
            step_action_surface_removals=step_action_surface_removals,
            step_unviable_nodes=step_nodes_made_unviable,
            truncated=self.cur_iter > self.max_iter,
            # terminate attacker if nothing left to do
            terminated=not new_action_surface
        )

    def _create_updated_defender_state(
            self,
            defender_state: DefenderState,
            step_compromised_nodes: set[AttackGraphNode],
            step_enabled_defenses: set[AttackGraphNode],
            step_nodes_made_unviable: set[AttackGraphNode],
            step_uncompromised_nodes: set[AttackGraphNode],
            all_attackers_done: bool
        ) -> DefenderState:
        """
        Build defender state from previous `defender_state` with given
        args. This is faster than running _create_defender_state every time,
        since it only adds the 'diff' of the state.
        It will also set the additional step_* fields for the agent state.

        Args:
            - defender_state: the old state of the defender to update
            - step_compromised_nodes: 
                the nodes compromised since last update
            - step_enabled_defenses:
                the defense node enabled since last update
            - step_nodes_made_unviable:
                the steps made unviable since last update
            - step_uncompromised_nodes:
                steps that were uncompromised
                (see sim_settings.uncompromise_untraversable_steps)
        Returns:
            - an AttackerState based on `attacker_state` updated with the
              given args
        """

        # defender step_reward is the sum of uncompromised nodes rewards
        # minus the sum of all enabled/compromised nodes rewards
        step_reward = (
            sum(n.extras.get('reward', 0) for n in step_uncompromised_nodes)
            - sum(
                n.extras.get('reward', 0)
                for n in step_compromised_nodes | step_enabled_defenses
            )
        )

        # new defender action surface is old one minus step_enabled_defenses
        new_action_surface = (
            defender_state.action_surface - step_enabled_defenses
        )

        return DefenderState(
            name=defender_state.name,
            reward=defender_state.reward + step_reward,
            action_surface=new_action_surface,
            step_performed_nodes=step_enabled_defenses,
            step_action_surface_additions=set(), # no additions for defender
            step_action_surface_removals=step_enabled_defenses,
            step_unviable_nodes=step_nodes_made_unviable,
            step_all_compromised_nodes=step_compromised_nodes,
            truncated=self.cur_iter > self.max_iter,
            # terminate defender if all attackers are terminated
            terminated=all_attackers_done
        )

    def _reset_agent_states(self) -> dict[str, AgentState]:
        """Return agent states created from current attack graph state"""
        agent_states: dict[str, AgentState] = {}
        for agent_state in self._get_attacker_states():
            # Create state for attackers
            agent_states[agent_state.name] = (
                self._create_attacker_state(
                    agent_state.name, agent_state.attacker_id
                )
            )
        for agent_state in self._get_defender_states():
            # Create state for defenders
            agent_states[agent_state.name] = (
                self._create_defender_state(agent_state.name)
            )
        return agent_states

    def _get_attacker_states(self) -> list[AttackerState]:
        """Return list of current attacker agent states of alive attackers"""
        return [
            a for a in self.agent_states.values()
            if a.name in self._alive_agents
            and isinstance(a, AttackerState)
        ]

    def _get_defender_states(self) -> list[DefenderState]:
        """Return list of current defender agent states of alive defenders"""
        return [
            a for a in self.agent_states.values()
            if a.name in self._alive_agents
            and isinstance(a, DefenderState)
        ]

    def _uncompromise_attack_steps(
        self, attack_steps_to_uncompromise: set[AttackGraphNode]
    ) -> set[AttackGraphNode]:
        """Uncompromise nodes for each attacker agent

        Go through the nodes in `attack_steps_to_uncompromise` for each
        attacker agent and uncompromise the nodes given if compromised.
        """
        uncompromised_nodes = set()
        for attacker_agent in self._get_attacker_states():
            attacker = self.attack_graph.attackers[attacker_agent.attacker_id]
            for node in attack_steps_to_uncompromise:
                if node.is_compromised_by(attacker):
                    uncompromised_nodes.add(node)
                    attacker.undo_compromise(node)
        return uncompromised_nodes

    def _attacker_step(
        self, agent: AttackerState, nodes: list[AttackGraphNode]
    ) -> set[AttackGraphNode]:
        """Compromise attack step nodes with attacker

        Args:
        agent - the agent to compromise nodes with
        nodes - the nodes to compromise

        Returns:
        - attack step nodes that were compromised by this agent this step
        """

        attacker = self.attack_graph.attackers[agent.attacker_id]
        step_compromised_nodes = set()

        for node in nodes:
            assert node == self.attack_graph.nodes[node.id], (
                f"{agent.name} tried to compromise a node that is not part "
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
                step_compromised_nodes.add(node)

                logger.info(
                    'Attacker agent "%s" compromised "%s"(%d).',
                    agent.name, node.full_name, node.id
                )
            else:
                logger.warning("Attacker could not compromise %s",
                               node.full_name)

        return step_compromised_nodes


    def _defender_step(
        self, agent: DefenderState, nodes: list[AttackGraphNode]
    ) -> tuple[
            set[AttackGraphNode],
            set[AttackGraphNode],
            set[AttackGraphNode]
        ]:
        """Enable defense step nodes with defender.

        Args:
        agent - the agent to activate defense nodes with
        nodes - the defense step nodes to enable

        Returns:
        - defense nodes that were enabled this step by this defender
        - attack step nodes made unviable by those defenses
        - attack steps node that were uncompromised this step
          (always empty if uncompromise_untraversable_steps=False)
        """

        enabled_defenses = set()
        attack_steps_made_unviable = set()
        uncompromised_attack_steps = set()

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
                enabled_defenses.add(node)
                logger.info(
                    'Defender agent "%s" enabled "%s"(%d).',
                    agent.name, node.full_name, node.id
                )

        if self.sim_settings.uncompromise_untraversable_steps:
            uncompromised_attack_steps = self._uncompromise_attack_steps(
                attack_steps_made_unviable
            )

        return (
            enabled_defenses,
            attack_steps_made_unviable,
            uncompromised_attack_steps
        )

    def register_attacker(self, name: str, attacker_id: int):
        """Register a mal sim attacker agent and create its state"""
        assert name not in self.agent_states, \
            f"Duplicate agent named {name} not allowed"

        self._alive_agents.add(name)
        agent_state = self._create_attacker_state(name, attacker_id)
        self.agent_states[name] = agent_state
        return agent_state

    def register_defender(self, name: str):
        """Register a mal sim defender agent and create its state"""
        assert name not in self.agent_states, \
            f"Duplicate agent named {name} not allowed"

        self._alive_agents.add(name)
        agent_state = self._create_defender_state(name)
        self.agent_states[name] = agent_state
        return agent_state

    def step(
        self, actions: dict[str, list[AttackGraphNode]]
    ) -> dict[str, AgentState]:
        """Take a step in the simulation

        Args:
        actions - a dict mapping agent name to agent action which
                  contains the actions for that user.

        Returns:
        - A dictionary containing agent states keyed by agent names
        """
        logger.info(
            "Stepping through iteration %d/%d", self.cur_iter, self.max_iter
        )
        logger.debug("Performing actions: %s", actions)

        # Store everything that happened in this step
        step_enabled_defenses: set[AttackGraphNode] = set()
        step_compromised_nodes: set[AttackGraphNode] = set()
        step_uncompromised_nodes: set[AttackGraphNode] = set()
        step_nodes_made_unviable: set[AttackGraphNode] = set()

        # Store compromised nodes per agent
        agent_compromised_nodes: dict[str, set[AttackGraphNode]] = {}

        # Perform defender actions first
        for defender_state in self._get_defender_states():
            agent_actions = actions.get(defender_state.name, [])
            enabled, unviable, uncompromised = (
                self._defender_step(defender_state, agent_actions)
            )
            step_enabled_defenses.update(enabled)
            step_uncompromised_nodes.update(uncompromised)
            step_nodes_made_unviable.update(unviable)

        # Perform attacker actions afterwards
        for attacker_state in self._get_attacker_states():
            agent_actions = actions.get(attacker_state.name, [])
            compromised = self._attacker_step(attacker_state, agent_actions)
            agent_compromised_nodes[attacker_state.name] = compromised
            step_compromised_nodes.update(compromised)

        # Update attacker states first
        all_attackers_done = True
        for old_attacker_state in self._get_attacker_states():
            new_attacker_state = (
                self._create_updated_attacker_state(
                    old_attacker_state,
                    agent_compromised_nodes[old_attacker_state.name],
                    step_nodes_made_unviable,
                    step_uncompromised_nodes
                )
            )
            self.agent_states[old_attacker_state.name] = new_attacker_state
            if new_attacker_state.truncated or new_attacker_state.terminated:
                logger.info("Attacker %s is done", new_attacker_state.name)
                self._alive_agents.remove(new_attacker_state.name)
            else:
                all_attackers_done = False

        # Update defender states afterwards
        for old_defender_state in self._get_defender_states():
            new_defender_state = (
                self._create_updated_defender_state(
                    old_defender_state,
                    step_compromised_nodes,
                    step_enabled_defenses,
                    step_nodes_made_unviable,
                    step_uncompromised_nodes,
                    all_attackers_done,
                )
            )
            self.agent_states[old_defender_state.name] = new_defender_state
            if new_defender_state.truncated or new_defender_state.terminated:
                logger.info("Defender %s is done", new_defender_state.name)
                self._alive_agents.remove(new_defender_state.name)

        self.cur_iter += 1
        return self.agent_states

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> dict[str, AgentState]:
        """Reset attack graph, iteration and reinitialize agents"""

        logger.info("Resetting MAL Simulator.")
        # Reset attack graph
        self.attack_graph = copy.deepcopy(self.attack_graph_backup)
        # Reset current iteration
        self.cur_iter = 0
        # Revive all agents
        self._alive_agents = set(self.agent_states.keys())
        # Reset agents to attack graph state
        self.agent_states = self._reset_agent_states()

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
