from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import random
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional, TYPE_CHECKING

from maltoolbox.attackgraph import (
    AttackGraph,
    AttackGraphNode
)

from malsim.probs_utils import (
    calculate_prob,
    ProbCalculationMethod,
)

from malsim.graph_processing import (
    calculate_necessity,
    calculate_viability,
    make_node_unviable,
)

if TYPE_CHECKING:
    from .scenario import Scenario
    from malsim.agents import DecisionAgent

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
        self.num_attempts: dict[AttackGraphNode, int] = {}


class MalSimDefenderState(MalSimAgentState):
    """Stores the state of a defender in the simulator"""

    # Steps compromised successfully by any attacker in the last step
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

class TTCMode(Enum):
    """
    Describes how to use the probability distributions in the attack graph.
    """
    LIVE_SAMPLE = 1
    PRESAMPLE = 2
    EXPECTED = 3
    DISABLED = 4

class RewardMode(Enum):
    """Two different ways to generate rewards"""
    CUMULATIVE = 1  # Reward calculated on all previous steps actions
    ONE_OFF = 2     # Reward calculated only for current step actions

@dataclass
class MalSimulatorSettings():
    """Contains settings used in MalSimulator"""

    # uncompromise_untraversable_steps
    # - Uncompromise (evict attacker) from nodes/steps that are no longer
    #   traversable (often because a defense kicked in) if set to True
    # otherwise:
    # - Leave the node/step compromised even after it becomes untraversable
    uncompromise_untraversable_steps: bool = False
    # ttc_mode
    # - mode to sample TTCs on attack steps
    ttc_mode: TTCMode = TTCMode.DISABLED
    # seed
    # - optionally run deterministic simulations with seed
    seed: Optional[int] = None
    # attack_surface_skip_compromised
    # - if true do not add already compromised nodes to the attack surface
    attack_surface_skip_compromised: bool = True
    # attack_surface_skip_unviable
    # - if true do not add unviable nodes to the attack surface
    attack_surface_skip_unviable: bool = True
    # attack_surface_skip_unnecessary
    # - if true do not add unnecessary nodes to the attack surface
    attack_surface_skip_unnecessary: bool = True
    # run_defense_step_bernoullis
    # - if true, sample defenses bernoullis to decide their initial states
    run_defense_step_bernoullis: bool = True

    # Reward settings
    attacker_reward_mode: RewardMode = RewardMode.CUMULATIVE
    defender_reward_mode: RewardMode = RewardMode.CUMULATIVE

class MalSimulator():
    """A MAL Simulator that works on the AttackGraph

    Allows user to register agents (defender and attacker)
    and lets the agents perform actions step by step and updates
    the state of the attack graph based on the steps chosen.
    """

    def __init__(
        self,
        attack_graph: AttackGraph,
        node_rewards: Optional[dict[AttackGraphNode, float]] = None,
        sim_settings: MalSimulatorSettings = MalSimulatorSettings(),
        max_iter: int = ITERATIONS_LIMIT,
    ):
        """
        Args:
            attack_graph                -   The attack graph to use
            sim_settings                -   Settings for simulator
            max_iter                    -   Max iterations in simulation
        """
        logger.info("Creating Base MAL Simulator.")
        self.sim_settings = sim_settings
        random.seed(self.sim_settings.seed)

        # Initialize all values
        self.attack_graph = attack_graph

        self.max_iter = max_iter  # Max iterations before stopping simulation
        self.cur_iter = 0         # Keep track on current iteration

        # All internal agent states (dead or alive)
        self._agent_states: dict[str, MalSimAgentState] = {}

        # Store properties of each AttackGraphNode
        self._node_rewards: dict[AttackGraphNode, float] = node_rewards or {}
        self._enabled_defenses: set[AttackGraphNode] = set()
        self._calculated_bernoullis: dict[AttackGraphNode, float] = {}

        # Keep track on all 'living' agents sorted by order to step in
        self._alive_agents: set[str] = set()

        # Do initial calculations
        self._ttc_values = self._attack_step_ttcs()

        if self.sim_settings.run_defense_step_bernoullis:
            self._enabled_defenses = self._pre_enabled_defenses()

        self._viability_per_node = calculate_viability(
            self.attack_graph, self._enabled_defenses, self._ttc_values
        )
        self._necessity_per_node = calculate_necessity(
            self.attack_graph, self._enabled_defenses
        )

    def _attack_step_ttcs(self) -> dict[AttackGraphNode, float]:
        """Calculate and return attack steps TTCs"""
        ttc_values = {}
        for node in self.attack_graph.nodes.values():
            match(self.sim_settings.ttc_mode):
                case TTCMode.DISABLED:
                    ttc_value = 1.0
                case TTCMode.LIVE_SAMPLE | TTCMode.PRESAMPLE:
                    ttc_value = calculate_prob(
                        node,
                        node.ttc,
                        ProbCalculationMethod.SAMPLE,
                        self._calculated_bernoullis
                    )
                case TTCMode.EXPECTED:
                    ttc_value = calculate_prob(
                        node,
                        node.ttc,
                        ProbCalculationMethod.EXPECTED,
                        self._calculated_bernoullis
                    )
            if node.type in ['and', 'or']:
                ttc_values[node] = ttc_value

        return ttc_values

    def _pre_enabled_defenses(self) -> set[AttackGraphNode]:
        """Calculate and return pre enabled defenses"""
        pre_enabled_defenses = set()
        for node in self.attack_graph.nodes.values():
            if node.type == 'defense':
                ttc_value = calculate_prob(
                    node,
                    node.ttc,
                    ProbCalculationMethod.SAMPLE,
                    self._calculated_bernoullis
                )
                if ttc_value != math.inf:
                    pre_enabled_defenses.add(node)
        return pre_enabled_defenses

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        sim_settings: MalSimulatorSettings = MalSimulatorSettings(),
        max_iter: int = ITERATIONS_LIMIT,
        register_agents: bool = True,
        **kwargs: Any
    ) -> MalSimulator:
        """Create a MalSimulator object from a Scenario"""
        sim = cls(
            scenario.attack_graph,
            node_rewards=scenario.rewards,
            sim_settings=sim_settings,
            max_iter=max_iter,
            **kwargs
        )

        # Register agents
        if register_agents:
            for agent_info in scenario.agents:
                if agent_info['type'] == AgentType.ATTACKER:
                    sim.register_attacker(
                        agent_info['name'],
                        agent_info['entry_points']
                    )
                elif agent_info['type'] == AgentType.DEFENDER:
                    sim.register_defender(agent_info['name'])

        return sim

    def node_is_viable(self, node: AttackGraphNode) -> bool:
        """Get viability of a node"""
        return self._viability_per_node[node]

    def node_is_necessary(self, node: AttackGraphNode) -> bool:
        """Get necessity of a node"""
        return self._necessity_per_node[node]

    def node_is_enabled_defense(self, node: AttackGraphNode) -> bool:
        """Get a nodes defense status"""
        return node in self._enabled_defenses

    def node_reward(self, node: AttackGraphNode) -> float:
        """Get reward for a node"""
        return self._node_rewards.get(node, 0.0)

    def reset(
        self,
        options: Optional[dict[str, Any]] = None
    ) -> dict[str, MalSimAgentStateView]:
        """Reset attack graph, iteration and reinitialize agents"""

        random.seed(self.sim_settings.seed)
        logger.info("Resetting MAL Simulator.")

        # Reset nodes
        self._calculated_bernoullis.clear()
        self._enabled_defenses = set()

        self._ttc_values = self._attack_step_ttcs()

        if self.sim_settings.run_defense_step_bernoullis:
            self._enabled_defenses = self._pre_enabled_defenses()

        self._viability_per_node = calculate_viability(
            self.attack_graph, self._enabled_defenses, self._ttc_values
        )
        self._necessity_per_node = calculate_necessity(
            self.attack_graph, self._enabled_defenses
        )

        self.cur_iter = 0
        self._reset_agents()

        return self.agent_states

    def _get_defense_surface(self) -> set[AttackGraphNode]:
        """Get the defense surface.
        All non-suppressed defense steps that are not already enabled.

        Arguments:
        graph       - the attack graph
        """
        return {
            node for node in self.attack_graph.nodes.values()
            if self.node_is_viable(node)
            and node.type == 'defense'
            and 'suppress' not in node.tags
            and not self.node_is_enabled_defense(node)
        }

    def node_is_traversable(
            self,
            attacker_state: MalSimAttackerState,
            node: AttackGraphNode
        ) -> bool:
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

        if not self.node_is_viable(node):
            return False

        match(node.type):
            case 'or':
                traversable = any(
                    parent in attacker_state.performed_nodes
                    for parent in node.parents
                )
            case 'and':
                traversable = all(
                    parent in attacker_state.performed_nodes
                    or not self.node_is_necessary(parent)
                    for parent in node.parents
                )
            case 'exist' | 'notExist' | 'defense':
                traversable = False
            case _:
                raise TypeError(
                    f'Node "{node.full_name}"({node.id})'
                    f'has an unknown type "{node.type}".'
                )
        return traversable

    def _get_attack_surface(
            self,
            attacker_state: MalSimAttackerState,
            from_nodes: Optional[set[AttackGraphNode]] = None
    ) -> set[AttackGraphNode]:
        """
        Calculate the attack surface of the attacker.
        If from_nodes are provided only calculate the attack surface
        stemming from those nodes, otherwise use all nodes the attacker
        has compromised. If skip_compromised is true, exclude already
        compromised nodes from the returned attack surface.

        The attack surface includes all of the traversable children nodes.

        Arguments:
        from_nodes        - the nodes to calculate the attack surface from;
                            defaults to the attackers compromised nodes list
                            if omitted
        """
        logger.debug(
            'Get the attack surface for Attacker "%s".', attacker_state.name
        )
        attack_surface = set()
        frontier = (
            from_nodes if from_nodes is not None
            else attacker_state.performed_nodes
        )
        for attack_step in frontier:
            for child in attack_step.children:
                if (
                    self.sim_settings.attack_surface_skip_compromised
                    and child in attacker_state.performed_nodes
                ):
                    continue
                if (
                    self.sim_settings.attack_surface_skip_unviable
                    and not self.node_is_viable(child)
                ):
                    continue
                if (
                    self.sim_settings.attack_surface_skip_unnecessary
                    and not self.node_is_necessary(child)
                ):
                    continue
                if (
                    child not in attack_surface and
                    self.node_is_traversable(attacker_state, child)
                ):
                    attack_surface.add(child)

        return attack_surface

    def _create_attacker_state(
        self, name: str, entry_points: set[AttackGraphNode]
    ) -> MalSimAttackerState:
        """Create a new defender state, initialize values"""
        attacker_state = MalSimAttackerState(name)
        attacker_state.step_performed_nodes = set(entry_points)
        attacker_state.performed_nodes = set(entry_points)
        attacker_state.action_surface = (
            self._get_attack_surface(attacker_state)
        )
        attacker_state.step_action_surface_additions = (
            set(attacker_state.action_surface)
        )
        attacker_state.step_action_surface_removals = set()
        attacker_state.entry_points = set(entry_points)
        attacker_state.reward = self._attacker_reward(
            attacker_state, self.sim_settings.attacker_reward_mode
        )
        attacker_state.num_attempts = {
            n: 0 for n in self.attack_graph.nodes.values()
        }
        return attacker_state

    def _create_defender_state(self, name: str) -> MalSimDefenderState:
        """Create a new defender state, initialize values"""

        defender_state = MalSimDefenderState(name)
        # Defender needs these for its initial state
        compromised_steps = set()
        for attacker_state in self._get_attacker_agents():
            compromised_steps |= attacker_state.performed_nodes
        defender_state.step_performed_nodes = self._enabled_defenses
        defender_state.performed_nodes = self._enabled_defenses
        defender_state.step_all_compromised_nodes = compromised_steps
        defender_state.action_surface = self._get_defense_surface()
        defender_state.step_action_surface_additions = (
            set(defender_state.action_surface)
        )
        defender_state.step_action_surface_removals = set()
        defender_state.reward = self._defender_reward(
            defender_state, self.sim_settings.defender_reward_mode
        )
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

        # Find what nodes attacker can reach this step
        new_action_surface = self._get_attack_surface(attacker_state)
        action_surface_removals = (
            attacker_state.action_surface - new_action_surface
        )
        action_surface_additions = (
            new_action_surface - attacker_state.action_surface
        )
        attacker_state.step_action_surface_additions = action_surface_additions
        attacker_state.step_action_surface_removals = action_surface_removals
        attacker_state.action_surface = new_action_surface
        attacker_state.reward = self._attacker_reward(
            attacker_state, self.sim_settings.attacker_reward_mode
        )
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
        defender_state.reward = self._defender_reward(
            defender_state, self.sim_settings.defender_reward_mode
        )
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
            self._agent_states[attacker_state.name] = (
                self._create_attacker_state(
                    attacker_state.name,
                    attacker_state.entry_points
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

    def _get_attacker_agents(
            self, only_alive: bool = False
        ) -> list[MalSimAttackerState]:
        """Return list of mutable attacker agent states of attackers.
        If `only_alive` is set to True, only return the agents that are alive.
        """
        return [
            a for a in self._agent_states.values()
            if (a.name in self._alive_agents or not only_alive)
            and isinstance(a, MalSimAttackerState)
        ]

    def _get_defender_agents(
            self, only_alive: bool = False
        ) -> list[MalSimDefenderState]:
        """Return list of mutable defender agent states of defenders.
        If `only_alive` is set to True, only return the agents that are alive.
        """
        return [
            a for a in self._agent_states.values()
            if (a.name in self._alive_agents or not only_alive)
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
                    node_reward = self.node_reward(unviable_node)
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
            if self.node_is_traversable(agent, node):
                node_ttc_value = self._ttc_values[node]

                if self.sim_settings.ttc_mode == TTCMode.LIVE_SAMPLE:
                    node_ttc_value = calculate_prob(
                        node,
                        node.ttc,
                        ProbCalculationMethod.SAMPLE,
                        self._calculated_bernoullis
                    )

                agent.num_attempts[node] += 1
                if agent.num_attempts[node] >= node_ttc_value:
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
            if node in agent.action_surface:
                enabled_defenses.add(node)
                self._viability_per_node, made_unviable = make_node_unviable(
                    node, self._viability_per_node, self._ttc_values
                )
                attack_steps_made_unviable |= made_unviable
                logger.info(
                    'Defender agent "%s" enabled "%s"(%d).',
                    agent.name, node.full_name, node.id
                )

        if self.sim_settings.uncompromise_untraversable_steps:
            self._uncompromise_attack_steps(attack_steps_made_unviable)

        return enabled_defenses, attack_steps_made_unviable

    def _attacker_reward(
            self,
            attacker_state: MalSimAttackerState,
            reward_mode: RewardMode
        ) -> float:
        """
        Calculate current attacker reward either cumulative or one-off.
        If cumulative, sum previous and one-off reward, otherwise
        just return the one-off reward.

        Args:
        - attacker_state: the current attacker state
        - reward_mode: which way to calculate reward
        """

        # Attacker is rewarded for compromised nodes
        step_reward = sum(
            self.node_reward(n)
            for n in attacker_state.step_performed_nodes
        )

        if reward_mode == RewardMode.CUMULATIVE:
            # To make it cumulative, add previous step reward
            step_reward += attacker_state.reward

        return step_reward


    def _defender_reward(
            self,
            defender_state: MalSimDefenderState,
            reward_mode: RewardMode
        ) -> float:
        """
        Calculate current defender reward either cumulative or one-off.
        If cumulative, sum previous and one-off reward, otherwise
        just return the one-off reward.

        Args:
        - defender_state: the defender state before defenses were enabled
        - reward_mode: which way to calculate reward
        """
        step_enabled_defenses = defender_state.step_performed_nodes
        step_compromised_nodes = defender_state.step_all_compromised_nodes

        # Defender is penalized for compromised steps and enabled defenses
        step_reward = - sum(
            self.node_reward(n)
            for n in step_enabled_defenses | step_compromised_nodes
        )

        if reward_mode == RewardMode.CUMULATIVE:
            # To make it cumulative add previous step reward
            step_reward += defender_state.reward

        return step_reward

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

        if not self._alive_agents:
            logger.warning(
                "No agents are alive anymore, step will have no effect"
            )

        for agent_name in actions:
            if agent_name not in self._agent_states:
                raise KeyError(f"No agent has name '{agent_name}'")

        # Populate these from the results for all agents' actions.
        step_all_compromised_nodes: set[AttackGraphNode] = set()
        step_enabled_defenses: set[AttackGraphNode] = set()
        step_nodes_made_unviable: set[AttackGraphNode] = set()

        # Perform defender actions first
        for defender_state in self._get_defender_agents(only_alive=True):
            enabled, unviable = self._defender_step(
                defender_state, actions.get(defender_state.name, [])
            )
            step_enabled_defenses |= enabled
            step_nodes_made_unviable |= unviable

        # Perform attacker actions afterwards
        for attacker_state in self._get_attacker_agents(only_alive=True):
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
                logger.info(
                    "Removing agent %s since it is terminated or truncated",
                    agent_state.name
                )
                self._alive_agents.remove(agent_state.name)

        self.cur_iter += 1
        return self.agent_states

    def render(self) -> None:
        pass


def run_simulation(sim: MalSimulator, agents: list[dict[str, Any]]) -> None:
    """Run a simulation with agents"""

    states = sim.reset()
    total_rewards = {agent_dict['name']: 0.0 for agent_dict in agents}
    all_agents_term_or_trunc = False

    logger.info("Starting CLI env simulator.")

    i = 1
    while not all_agents_term_or_trunc:
        print(f"Iteration {i}")
        all_agents_term_or_trunc = True
        actions = {}

        # Select actions for each agent
        for agent_dict in agents:
            decision_agent: Optional[DecisionAgent] = agent_dict.get('agent')
            agent_name = agent_dict['name']
            if decision_agent is None:
                print(
                    f'Agent "{agent_name}" has no decision agent class '
                    'specified in scenario. Waiting.'
                )
                continue

            sim_agent_state = states[agent_name]
            agent_action = decision_agent.get_next_action(sim_agent_state)
            if agent_action:
                actions[agent_name] = [agent_action]
                print(
                    f'Agent {agent_name} chose action: '
                    f'{agent_action.full_name}'
                )

        # Perform next step of simulation
        states = sim.step(actions)

        for agent_dict in agents:
            agent_name = agent_dict['name']
            agent_state = states[agent_name]
            total_rewards[agent_name] += agent_state.reward
            if not agent_state.terminated and not agent_state.truncated:
                all_agents_term_or_trunc = False
        print("---")
        i += 1

    print(f"Game Over after {i} steps.")

    # Print total rewards
    for agent_dict in agents:
        agent_name = agent_dict['name']
        print(f'Total reward "{agent_name}"', total_rewards[agent_name])
