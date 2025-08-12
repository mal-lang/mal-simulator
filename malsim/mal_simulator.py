from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional, Iterable

from maltoolbox.attackgraph import (
    AttackGraph,
    AttackGraphNode
)

from malsim.graph_processing import (
    calculate_viability_and_necessity,
    propagate_viability_from_node,
    prune_unviable_and_unnecessary_nodes
)

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
    reward: float = 0

    # Contains possible actions for the agent in the next step
    action_surface: set[AttackGraphNode] = field(default_factory=set)

    # Contains all nodes that this agent has performed successfully
    performed_nodes: set[AttackGraphNode] = field(default_factory=set)

    # Contains the nodes performed successfully in the last step
    step_performed_nodes: set[AttackGraphNode] = field(default_factory=set)

    # Contains possible nodes that became available in the last step
    step_action_surface_additions: set[AttackGraphNode] = (
        field(default_factory = set))

    # Contains nodes that became unavailable in the last step
    step_action_surface_removals: set[AttackGraphNode] = (
        field(default_factory = set))

    # Contains nodes that became unviable in the last step by defender actions
    step_unviable_nodes: set[AttackGraphNode] = field(default_factory=set)

    # Fields that tell if the agent is done or stopped
    truncated: bool = False
    terminated: bool = False


class MalSimAttackerState(MalSimAgentState):
    """Stores the state of an attacker in the simulator"""

    def __init__(self, name: str):
        super().__init__(name, AgentType.ATTACKER)
        self.entry_points: set[AttackGraphNode] = set()

    def is_node_traversable(self, node: AttackGraphNode) -> bool:
        """
        Return True or False depending if the node specified is traversable
        for given the current attacker agent state.

        A node is traversable if it is viable and:
        - if it is of type 'or' and any of its parents have been compromised
        - if it is of type 'and' and all of its necessary parents have been
          compromised

        Arguments:
        node        - the node we wish to evalute
        """

        if not node.is_viable:
            return False

        match(node.type):
            case 'or':
                return any(
                    parent in self.performed_nodes for parent in node.parents
                )
            case 'and':
                return all(
                    parent in self.performed_nodes
                    or not parent.is_necessary
                    for parent in node.parents
                )
            case 'exist' | 'notExist' | 'defense':
                return False
            case _:
                raise TypeError(
                    f'Node "{node.full_name}"({node.id})'
                    f'has an unknown type "{node.type}".'
                )

    def calculate_attack_surface(
            self,
            *,
            from_nodes: Optional[Iterable[AttackGraphNode]] = None,
            skip_compromised: bool = False,
    ) -> set[AttackGraphNode]:
        """
        Calculate the attack surface of the attacker. If from_nodes are provided
        only calculate the attack surface stemming from those nodes, otherwise use
        all nodes the attacker has compromised. If skip_compromised is true,
        exclude already compromised nodes from the returned attack surface.

        The attack surface includes all of the traversable children nodes.

        Arguments:
        from_nodes        - the nodes to calculate the attack surface from; defaults
                            to the attackers compromised nodes list if omitted
        skip_compromised  - if true do not add already compromised nodes to the
                            attack surface
        """
        logger.debug('Get the attack surface for Attacker "%s".', self.name)
        attack_surface = set()
        frontier = (
            from_nodes if from_nodes is not None else self.performed_nodes
        )
        for attack_step in frontier:
            for child in attack_step.children:
                if skip_compromised and child in self.performed_nodes:
                    continue
                if (
                    child not in attack_surface and
                    self.is_node_traversable(child)
                ):
                    attack_surface.add(child)

        return attack_surface



class MalSimDefenderState(MalSimAgentState):
    """Stores the state of a defender in the simulator"""

    # Contains the steps performed successfully by all of the attacker agents
    # in the last step
    step_all_compromised_nodes: set[AttackGraphNode] = set()

    def __init__(self, name: str):
        super().__init__(name, AgentType.DEFENDER)

    def is_enabled_defense(self, node: AttackGraphNode) -> bool:
        """
        Return True if this node is a defense node and it is enabled and not
        suppressed via tags.
        False, otherwise.
        """
        return node.type == 'defense' and \
            'suppress' not in node.tags and \
            node.defense_status == 1.0

    def is_available_defense(self, node: AttackGraphNode) -> bool:
        """
        Return True if this node is a defense node and it is not fully enabled
        and not suppressed via tags. False otherwise.
        """
        return node.type == 'defense' and \
            'suppress' not in node.tags and \
            node.defense_status != 1.0

    def get_defense_surface(self, graph: AttackGraph) -> set[AttackGraphNode]:
        """
        Get the defense surface. All non-suppressed defense steps that are not
        already fully enabled.

        Arguments:
        graph       - the attack graph
        """
        logger.debug('Get the defense surface.')
        return {node for node in graph.nodes.values()
            if self.is_available_defense(node)}



class MalSimAgentStateView(MalSimAttackerState, MalSimDefenderState):
    """Read-only interface to MalSimAgentState.

    Subclassing from State classes only to satisfy static analysis and support
    autocompletion that would otherwise be unavailable due to the dynamic
    __getattr(ibute)__ method.
    """

    _frozen = False

    def __init__(self, agent: MalSimAgentState):
        self._agent = agent
        self._frozen = True

    def __setattr__(self, key: str, value: Any) -> None:
        if self._frozen:
            raise AttributeError("Cannot modify agent state view")
        super().__setattr__(key, value)

    def __delattr__(self, key: str) -> None:
        if self._frozen:
            raise AttributeError("Cannot modify agent state view")
        super().__delattr__(key)

    def __getattribute__(self, attr: str) -> Any:
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

    def __dir__(self) -> list[str]:
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

def prepare_attack_graph(
    attack_graph: AttackGraph,
    prune_unviable_unnecessary: bool = True
) -> AttackGraph:
    """
    Prepare the attack graph for running the simulation:
    - assign defense values to the defenses
    - calculate the viability and necessity of nodes
    - prune nodes if requested
    - in the future this should also handle initial TTC evaluations

    Args:
        attack_graph                -   The attack graph to modify
        prune_unviable_unnecessary  -   Prunes graph of unnecessary and
                                        unviable nodesif set to true
    """
    for node in attack_graph.nodes.values():
        # TODO: This and the rest of the defense_status logic in
        # MalSimDefenderState should be redone once the TTC calculations are
        # in place.
        if node.type == 'defense':
            node.defense_status = 1.0 if node.ttc and \
                node.ttc['name'] == 'Enabled' else 0.0
        node.is_viable = True
        node.is_necessary = True
        node.extras['viable'] = node.is_viable
        node.extras['necessary'] = node.is_necessary

    # Calculate viability and necessity and optionally prune graph
    calculate_viability_and_necessity(attack_graph)
    if prune_unviable_unnecessary:
        prune_unviable_and_unnecessary_nodes(attack_graph)

    return attack_graph

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
        max_iter: int = ITERATIONS_LIMIT,
    ):
        """
        Args:
            attack_graph                -   The attack graph to use
            max_iter                    -   Max iterations in simulation
            prune_unviable_unnecessary  -   Prunes graph if set to true
            sim_settings                -   Settings for simulator
        """
        logger.info("Creating Base MAL Simulator.")

        self.prune_unviable_unnecessary = prune_unviable_unnecessary
        prepare_attack_graph(attack_graph, prune_unviable_unnecessary)

        # Keep a backup attack graph to use when resetting
        self.attack_graph_backup = copy.deepcopy(attack_graph)

        # Initialize all values
        self.attack_graph = attack_graph

        self.sim_settings = sim_settings
        self.max_iter = max_iter  # Max iterations before stopping simulation
        self.cur_iter = 0        # Keep track on current iteration

        # All internal agent states (dead or alive)
        self._agent_states: dict[str, MalSimAttackerState | MalSimDefenderState] = {}

        # Keep track on all 'living' agents sorted by order to step in
        self._alive_agents: set[str] = set()


    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
    ) -> dict[str, MalSimAgentStateView]:
        """Reset attack graph, iteration and reinitialize agents"""

        logger.info("Resetting MAL Simulator.")
        # Reset attack graph
        self.attack_graph = copy.deepcopy(self.attack_graph_backup)

        #TODO: If we do a soft reset of the attack graph we do not need
        # prepare it again. The deepcopy loses the defense, viability, and
        # necessity values.
        prepare_attack_graph(
            self.attack_graph,
            self.prune_unviable_unnecessary
        )

        # Reset current iteration
        self.cur_iter = 0
        # Reset agents
        self._reset_agents()

        return self.agent_states

    def _create_attacker_state(
        self, name: str, entry_points: set[AttackGraphNode]
    ) -> MalSimAttackerState:
        """Create a new defender state, initialize values"""
        attacker_state = MalSimAttackerState(name)
        attacker_state.step_performed_nodes = set(entry_points)
        attacker_state.performed_nodes = set(entry_points)
        attacker_state.action_surface = attacker_state.calculate_attack_surface(
            skip_compromised = True
        )
        attacker_state.step_action_surface_additions = (
            set(attacker_state.action_surface)
        )
        attacker_state.step_action_surface_removals = set()
        attacker_state.entry_points = set(entry_points)
        attacker_state.reward = self._attacker_reward(attacker_state)
        return attacker_state

    def _create_defender_state(self, name: str) -> MalSimDefenderState:
        """Create a new defender state, initialize values"""

        defender_state = MalSimDefenderState(name)
        # Defender needs these for its initial state
        enabled_defenses = set(
            node for node in self.attack_graph.nodes.values()
            if defender_state.is_enabled_defense(node)
        )
        compromised_steps = set()
        for attacker_state in self._get_attacker_agents():
            compromised_steps |= attacker_state.performed_nodes
        defender_state.step_performed_nodes = enabled_defenses
        defender_state.performed_nodes = enabled_defenses
        defender_state.step_all_compromised_nodes = compromised_steps
        defender_state.action_surface = (
            defender_state.get_defense_surface(self.attack_graph)
        )
        defender_state.step_action_surface_additions = (
            set(defender_state.action_surface)
        )
        defender_state.step_action_surface_removals = set()
        defender_state.reward = self._defender_reward(defender_state)
        return defender_state

    def _update_attacker_state(
        self,
        attacker_state: MalSimAttackerState,
        step_agent_compromised_nodes: set[AttackGraphNode],
        step_nodes_made_unviable: set[AttackGraphNode]
    ) -> None:
        """
        Update a previous attacker state based on what the agent compromised
        and what nodes became unviable.
        """
        attacker_state.step_performed_nodes = step_agent_compromised_nodes
        attacker_state.performed_nodes |= step_agent_compromised_nodes
        attacker_state.step_unviable_nodes = step_nodes_made_unviable

        # Find what new steps attacker can reach this step
        action_surface_additions = (
            attacker_state.calculate_attack_surface(
                from_nodes=step_agent_compromised_nodes,
                skip_compromised=True
            ) # Exclude nodes already in the action surface
            - attacker_state.action_surface
        )
        # Remove nodes agent compromised and nodes made unviable
        action_surface_removals = (
            attacker_state.action_surface &
            (step_nodes_made_unviable | step_agent_compromised_nodes)
        )
        # New action surface is the old one plus the additions, minus removals
        new_action_surface = (
            (attacker_state.action_surface | action_surface_additions)
            - action_surface_removals
        )

        attacker_state.step_action_surface_additions = action_surface_additions
        attacker_state.step_action_surface_removals = action_surface_removals
        attacker_state.action_surface = new_action_surface
        attacker_state.reward = self._attacker_reward(attacker_state)
        attacker_state.truncated = self.cur_iter >= self.max_iter
        attacker_state.terminated = (
            self._attacker_is_terminated(attacker_state)
        )

    def _update_defender_state(
        self,
        defender_state: MalSimDefenderState,
        step_all_compromised_nodes: set[AttackGraphNode],
        step_enabled_defenses: set[AttackGraphNode],
        step_nodes_made_unviable: set[AttackGraphNode],
    ) -> None:
        """
        Update a previous defender state based on what steps
        were enabled/compromised during last step
        """
        defender_state.step_action_surface_additions = set()
        defender_state.step_action_surface_removals = step_enabled_defenses
        defender_state.action_surface -= step_enabled_defenses
        defender_state.step_all_compromised_nodes = step_all_compromised_nodes
        defender_state.step_unviable_nodes = step_nodes_made_unviable
        defender_state.step_performed_nodes = step_enabled_defenses
        defender_state.performed_nodes |= step_enabled_defenses
        defender_state.reward = self._defender_reward(defender_state)
        defender_state.truncated = self.cur_iter >= self.max_iter
        defender_state.terminated = self._defender_is_terminated(
            self._get_attacker_agents()
        )

    def _reset_agents(self) -> None:
        """Reset agent states to a fresh start"""

        # Revive all agents
        self._alive_agents = set(self._agent_states.keys())

        # Create new attacker agent states
        for attacker_state in self._get_attacker_agents():
            # TODO: Re-fetching the entry nodes is only need if we fully reset
            # the attack graph which should not be the case with the new
            # implementation.
            new_entry_point_nodes = {self.attack_graph.nodes[node.id]
                for node in attacker_state.entry_points}
            self._agent_states[attacker_state.name] = (
                self._create_attacker_state(
                    attacker_state.name,
                    new_entry_point_nodes
                )
            )

        # Create new defender agent states
        for defender_state in self._get_defender_agents():
            self._agent_states[defender_state.name] = (
                self._create_defender_state(defender_state.name)
            )

    def register_attacker(
        self, name: str, entry_points: set[AttackGraphNode]
    ) -> None:
        """Register a mal sim attacker agent"""
        assert name not in self._agent_states, \
            f"Duplicate agent named {name} not allowed"

        agent_state = self._create_attacker_state(name, entry_points)
        self._agent_states[name] = agent_state
        self._alive_agents.add(name)

    def register_defender(self, name: str) -> None:
        """Register a mal sim defender agent"""
        assert name not in self._agent_states, \
            f"Duplicate agent named {name} not allowed"

        agent_state = self._create_defender_state(name)
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
    ) -> None:
        """Uncompromise nodes for each attacker agent

        Go through the nodes in `attack_steps_to_uncompromise` for each
        attacker agent. If a node is compromised by the attacker agent:
            - Uncompromise the node and remove rewards for it.
        """
        for attacker_agent in self._get_attacker_agents():

            for unviable_node in attack_steps_to_uncompromise:
                if unviable_node in attacker_agent.performed_nodes:

                    # Reward is no longer present for attacker
                    node_reward = unviable_node.extras.get('reward', 0)
                    attacker_agent.reward -= node_reward

                    # Reward is no longer present for defenders
                    for defender_agent in self._get_defender_agents():
                        defender_agent.reward += node_reward

                    # Uncompromise node if requested
                    attacker_agent.performed_nodes.remove(unviable_node)

    def _attacker_step(
        self, agent: MalSimAttackerState, nodes: list[AttackGraphNode]
    ) -> set[AttackGraphNode]:
        """Compromise attack step nodes with attacker

        Args:
        agent - the agent to compromise nodes with
        nodes - the nodes to compromise

        Returns: set of nodes that were compromised.
        """

        compromised_nodes = set()

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
            if agent.is_node_traversable(node) \
                    and node in agent.action_surface:

                compromised_nodes.add(node)

                logger.info(
                    'Attacker agent "%s" compromised "%s"(%d).',
                    agent.name, node.full_name, node.id
                )
            else:
                logger.warning(
                    "Attacker could not compromise %s", node.full_name
                )

        return compromised_nodes

    def _defender_step(
        self, agent: MalSimDefenderState, nodes: list[AttackGraphNode]
    ) -> tuple[set[AttackGraphNode], set[AttackGraphNode]]:
        """Enable defense step nodes with defender.

        Args:
        agent - the agent to activate defense nodes with
        nodes - the defense step nodes to enable

        Returns a tuple of two sets, `enabled_defenses`
        and `attack_steps_made_unviable`.
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
            if agent.is_available_defense(node):
                node.defense_status = 1.0
                node.is_viable = False
                attack_steps_made_unviable |= \
                    propagate_viability_from_node(node)
                enabled_defenses.add(node)
                logger.info(
                    'Defender agent "%s" enabled "%s"(%d).',
                    agent.name, node.full_name, node.id
                )

        if self.sim_settings.uncompromise_untraversable_steps:
            self._uncompromise_attack_steps(attack_steps_made_unviable)

        return enabled_defenses, attack_steps_made_unviable

    @staticmethod
    def _attacker_reward(attacker_state: MalSimAttackerState) -> float:
        """
        Calculate current attacker reward by adding this steps
        compromised node rewards to the previous attacker reward.
        Can be overridden by subclass to implement custom reward function.

        Args:
        - attacker_state: the attacker state before nodes were compromised
        - step_agent_compromised_nodes: set of nodes compromised
          since last reward was calculated
        """
        # Attacker is rewarded for compromised nodes
        return attacker_state.reward + sum(
            float(n.extras.get("reward", 0))
            for n in attacker_state.step_performed_nodes
        )

    @staticmethod
    def _defender_reward(defender_state: MalSimDefenderState) -> float:
        """
        Calculate current defender reward by subtracting this steps
        compromised/enabled node rewards from the previous defender reward.
        Can be overridden by subclass to implement custom reward function.

        Args:
        - defender_state: the defender state before defenses were enabled
        - step_enabled_defenses: set of defenses enabled since last reward was
          calculated
        """
        # Defender is penalized for compromised nodes and enabled defenses
        step_enabled_defenses = defender_state.step_performed_nodes
        step_compromised_nodes = defender_state.step_all_compromised_nodes
        return defender_state.reward - sum(
            float(n.extras.get("reward", 0))
            for n in step_enabled_defenses | step_compromised_nodes
        )

    @staticmethod
    def _attacker_is_terminated(attacker_state: MalSimAttackerState) -> bool:
        """Check if attacker is terminated
        Can be overridden by subclass for custom termination condition.

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
        Can be overridden by subclass for custom termination condition.

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
        step_all_compromised_nodes: set[AttackGraphNode] = set()
        step_enabled_defenses: set[AttackGraphNode] = set()
        step_nodes_made_unviable: set[AttackGraphNode] = set()

        # Perform defender actions first
        for defender_state in self._get_defender_agents():
            enabled, unviable = self._defender_step(
                defender_state, actions.get(defender_state.name, [])
            )
            step_enabled_defenses |= enabled
            step_nodes_made_unviable |= unviable

        # Perform attacker actions afterwards
        for attacker_state in self._get_attacker_agents():
            agent_compromised = self._attacker_step(
                attacker_state, actions.get(attacker_state.name, [])
            )
            step_all_compromised_nodes |= agent_compromised

            # Update attacker state
            self._update_attacker_state(
                attacker_state, agent_compromised, step_nodes_made_unviable
            )

        # Update defender states and remove 'dead' agents of any type
        for agent_name in self._alive_agents.copy():
            agent_state = self._agent_states[agent_name]

            if isinstance(agent_state, MalSimDefenderState):
                self._update_defender_state(
                    agent_state,
                    step_all_compromised_nodes,
                    step_enabled_defenses,
                    step_nodes_made_unviable
                )

            # Remove agents that are terminated or truncated
            if agent_state.terminated or agent_state.truncated:
                logger.info("Removing agent %s", agent_state.name)
                self._alive_agents.remove(agent_state.name)

        self.cur_iter += 1
        return self.agent_states

    def render(self) -> None:
        pass
