from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING
from collections.abc import Set
from types import MappingProxyType # For immutable dict

from numpy.random import default_rng

from maltoolbox.attackgraph import (
    AttackGraph,
    AttackGraphNode
)

from malsim.ttc_utils import (
    attempt_step_ttc,
    ttc_value_from_node,
    ProbCalculationMethod,
)

from malsim.graph_processing import (
    calculate_necessity,
    calculate_viability,
    make_node_unviable,
)

from malsim.scenario import AgentType, load_scenario

if TYPE_CHECKING:
    from .scenario import Scenario
    from malsim.agents import DecisionAgent

ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MalSimAgentState:
    """Stores the state of an agent in the simulator"""

    # Identifier of the agent, used in MalSimulator for lookup
    name: str

    # Reference to the simulator
    sim: MalSimulator

    # Contains possible actions for the agent in the next step
    action_surface: frozenset[AttackGraphNode]

    # Contains all nodes that this agent has performed successfully
    performed_nodes: frozenset[AttackGraphNode]

    # Contains the nodes performed successfully in the last step
    step_performed_nodes: frozenset[AttackGraphNode]

    # Contains possible nodes that became available in the last step
    step_action_surface_additions: frozenset[AttackGraphNode]

    # Contains nodes that became unavailable in the last step
    step_action_surface_removals: frozenset[AttackGraphNode]

    # Contains nodes that became unviable in the last step by defender actions
    step_unviable_nodes: frozenset[AttackGraphNode]

@dataclass(frozen=True)
class MalSimAttackerState(MalSimAgentState):
    """Stores the state of an attacker in the simulator"""

    # The starting points of an attacker agent
    entry_points: frozenset[AttackGraphNode]

    # Number of attempts to compromise a step (used for ttc caculations)
    num_attempts: MappingProxyType[AttackGraphNode, int]

@dataclass(frozen=True)
class MalSimDefenderState(MalSimAgentState):
    """Stores the state of a defender in the simulator"""

    # Contains all steps performed by any attacker
    compromised_nodes: frozenset[AttackGraphNode]

    # Contains steps successfully performed by any
    # attacker agent in the last step
    step_all_compromised_nodes: frozenset[AttackGraphNode]


class TTCMode(Enum):
    """
    Describes how to use the probability distributions in the attack graph.
    """
    EFFORT_BASED_PER_STEP_SAMPLE = 0
    PER_STEP_SAMPLE = 1
    PRE_SAMPLE = 2
    EXPECTED_VALUE = 3
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
    # Make an attack graph node unviable if it has TTC infinity
    infinity_ttc_attack_step_unviable: bool = True

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
        self.rng = default_rng(self.sim_settings.seed)

        # Initialize all values
        self.attack_graph = attack_graph

        self.max_iter = max_iter  # Max iterations before stopping simulation
        self.cur_iter = 0         # Keep track on current iteration
        self.recording: dict[int, dict[str, list[AttackGraphNode]]] = {}

        # All internal agent states (dead or alive)
        self._agent_states: dict[str, MalSimAgentState] = {}
        self._agent_rewards: dict[str, float] = {}

        # Store properties of each AttackGraphNode
        self._node_rewards: dict[AttackGraphNode, float] = node_rewards or {}
        self._enabled_defenses: set[AttackGraphNode] = set()
        self._impossible_attack_steps: set[AttackGraphNode] = set()

        # Keep track on all 'living' agents sorted by order to step in
        self._alive_agents: set[str] = set()

        self._calculated_bernoullis: dict[AttackGraphNode, float] = {}
        self._ttc_values = self._attack_step_ttcs()

        # Do initial calculations
        if self.sim_settings.run_defense_step_bernoullis:
            self._enabled_defenses = self._get_pre_enabled_defenses()

        if self.sim_settings.infinity_ttc_attack_step_unviable:
            self._impossible_attack_steps = (
                self._get_impossible_attack_steps()
            )

        self._viability_per_node = calculate_viability(
            self.attack_graph,
            self._enabled_defenses,
            self._impossible_attack_steps
        )
        self._necessity_per_node = calculate_necessity(
            self.attack_graph, self._enabled_defenses
        )

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario | str,
        sim_settings: MalSimulatorSettings = MalSimulatorSettings(),
        max_iter: int = ITERATIONS_LIMIT,
        register_agents: bool = True,
        **kwargs: Any
    ) -> MalSimulator:
        """Create a MalSimulator object from a Scenario"""

        if isinstance(scenario, str):
            # Load scenario if file was given
            scenario = load_scenario(scenario)

        sim = cls(
            scenario.attack_graph,
            node_rewards=scenario.rewards,
            sim_settings=sim_settings,
            max_iter=max_iter,
            **kwargs
        )

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

    def done(self) -> bool:
        """Return True if simulation run is done"""
        return len(self._alive_agents) == 0 or self.cur_iter > self.max_iter

    def node_ttc_value(self, node: AttackGraphNode) -> float:
        """Return ttc value of node if it has been sampled"""
        assert self.sim_settings.ttc_mode in (
            TTCMode.PRE_SAMPLE, TTCMode.EXPECTED_VALUE
        ), "TTC value only when TTCMode is PRE_SAMPLE or EXPECTED_VALUE"

        assert node in self._ttc_values, (
            f"Node {node.full_name} does not have a ttc value"
        )
        return self._ttc_values[node]

    def node_is_viable(self, node: AttackGraphNode) -> bool:
        """Get viability of a node"""
        return self._viability_per_node[node]

    def node_is_necessary(self, node: AttackGraphNode) -> bool:
        """Get necessity of a node"""
        return self._necessity_per_node[node]

    def node_is_enabled_defense(self, node: AttackGraphNode) -> bool:
        """Get a nodes defense status"""
        return node in self._enabled_defenses

    def node_is_compromised(self, node: AttackGraphNode) -> bool:
        """Return True if node is compromised by any attacker agent"""
        for attacker_agent in self._get_attacker_agents():
            if node in attacker_agent.performed_nodes:
                return True
        return False

    def node_is_traversable(
            self,
            performed_nodes: Set[AttackGraphNode],
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
                    parent in performed_nodes
                    for parent in node.parents
                )
            case 'and':
                traversable = all(
                    parent in performed_nodes
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

    def agent_reward(self, agent_name: str) -> float:
        """Get an agents current reward"""
        return self._agent_rewards.get(agent_name, 0)

    def node_reward(self, node: AttackGraphNode) -> float:
        """Get reward for a node"""
        return self._node_rewards.get(node, 0.0)

    def agent_is_terminated(self, agent_name: str) -> bool:
        """Return True if agent was terminated"""
        agent_state = self._agent_states[agent_name]
        if isinstance(agent_state, MalSimAttackerState):
            return self._attacker_is_terminated(agent_state)
        elif isinstance(agent_state, MalSimDefenderState):
            return self._defender_is_terminated()
        else:
            raise TypeError(f"Unknown agent state for {agent_name}")

    def reset(
        self, options: Optional[dict[str, Any]] = None
    ) -> dict[str, MalSimAgentState]:
        """Reset attack graph, iteration and reinitialize agents"""

        self.rng = default_rng(self.sim_settings.seed)
        logger.info("Resetting MAL Simulator.")

        # Reset nodes
        self._calculated_bernoullis.clear()
        self._enabled_defenses = set()
        self._agent_rewards = {}
        self._ttc_values = self._attack_step_ttcs()

        if self.sim_settings.run_defense_step_bernoullis:
            self._enabled_defenses = self._get_pre_enabled_defenses()

        if self.sim_settings.infinity_ttc_attack_step_unviable:
            self._impossible_attack_steps = (
                self._get_impossible_attack_steps()
            )

        self._viability_per_node = calculate_viability(
            self.attack_graph,
            self._enabled_defenses,
            self._impossible_attack_steps
        )
        self._necessity_per_node = calculate_necessity(
            self.attack_graph, self._enabled_defenses
        )

        self.cur_iter = 0
        self.recording = {}
        self._reset_agents()

        return self.agent_states

    def _attack_step_ttcs(self) -> dict[AttackGraphNode, float]:
        """
        Calculate and return attack steps TTCs if settings use
        pre sample or expected value
        """
        ttc_values = {}
        for node in self.attack_graph.nodes.values():

            if node.type not in ('or', 'and'):
                continue

            match(self.sim_settings.ttc_mode):
                case TTCMode.EXPECTED_VALUE:
                    ttc_values[node] = ttc_value_from_node(
                        node,
                        ProbCalculationMethod.EXPECTED,
                        self._calculated_bernoullis,
                        self.rng
                    )
                case TTCMode.PRE_SAMPLE:
                    # Otherwise sample
                    ttc_values[node] = ttc_value_from_node(
                        node,
                        ProbCalculationMethod.SAMPLE,
                        self._calculated_bernoullis,
                        self.rng
                    )

        return ttc_values

    def _get_pre_enabled_defenses(self) -> set[AttackGraphNode]:
        """
        Calculate and return pre defenses that got a non-infinite
        ttc value sample, which means they will be pre enabled
        """
        pre_enabled_defenses = set()
        for node in self.attack_graph.nodes.values():
            if node.type == 'defense':
                ttc_value = ttc_value_from_node(
                    node,
                    ProbCalculationMethod.SAMPLE,
                    self._calculated_bernoullis,
                    self.rng
                )
                if ttc_value != math.inf:
                    pre_enabled_defenses.add(node)
        return pre_enabled_defenses

    def _get_impossible_attack_steps(self) -> set[AttackGraphNode]:
        """
        Calculate and return attack steps with that got
        infintity TTC in sample which means they are impossible
        """
        impossible_attack_steps = set()

        for node in self.attack_graph.nodes.values():
            if node.type in ('or', 'and'):
                ttc_value = ttc_value_from_node(
                    node,
                    ProbCalculationMethod.SAMPLE,
                    self._calculated_bernoullis,
                    self.rng
                )
                if ttc_value == math.inf:
                    impossible_attack_steps.add(node)

        return impossible_attack_steps

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

    def _get_attack_surface(
            self,
            performed_nodes: Set[AttackGraphNode],
    ) -> frozenset[AttackGraphNode]:
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

        attack_surface = set()
        for attack_step in performed_nodes:
            for child in attack_step.children:
                if (
                    self.sim_settings.attack_surface_skip_compromised
                    and child in performed_nodes
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
                    self.node_is_traversable(performed_nodes, child)
                ):
                    attack_surface.add(child)

        return frozenset(attack_surface)

    def _create_attacker_state(
        self,
        name: str,
        entry_points: Set[AttackGraphNode]
    ) -> MalSimAttackerState:
        """Create a new defender state, initialize values"""
        attack_surface = self._get_attack_surface(entry_points)
        attacker_state = MalSimAttackerState(
            name,
            sim=self,
            entry_points = frozenset(entry_points),
            performed_nodes = frozenset(entry_points),
            action_surface = frozenset(attack_surface),
            step_action_surface_additions = frozenset(attack_surface),
            step_action_surface_removals = frozenset(),
            step_performed_nodes = frozenset(entry_points),
            step_unviable_nodes=frozenset(),
            num_attempts = MappingProxyType({
                n: 0 for n in self.attack_graph.nodes.values()
            }),
        )
        return attacker_state

    def _update_attacker_state(
        self,
        attacker_state: MalSimAttackerState,
        step_agent_compromised_nodes: Set[AttackGraphNode],
        step_agent_attempted_nodes: Set[AttackGraphNode],
        step_nodes_made_unviable: Set[AttackGraphNode]
    ) -> MalSimAttackerState:
        """
        Update a previous attacker state based on what the agent compromised
        and what nodes became unviable.
        """

        # Find what nodes attacker can reach this step
        new_action_surface = self._get_attack_surface(
            attacker_state.performed_nodes | step_agent_compromised_nodes
        )
        action_surface_removals = frozenset(
            attacker_state.action_surface - new_action_surface
        )
        action_surface_additions = frozenset(
            new_action_surface - attacker_state.action_surface
        )

        num_attempts = dict(attacker_state.num_attempts)
        for node in step_agent_attempted_nodes:
            num_attempts[node] += 1

        updated_attacker_state = MalSimAttackerState(
            attacker_state.name,
            sim = self,
            performed_nodes = (
                attacker_state.performed_nodes | step_agent_compromised_nodes
            ),
            action_surface = new_action_surface,
            step_action_surface_additions = action_surface_additions,
            step_action_surface_removals = action_surface_removals,
            step_performed_nodes = frozenset(step_agent_compromised_nodes),
            step_unviable_nodes = frozenset(step_nodes_made_unviable),
            entry_points = attacker_state.entry_points,
            num_attempts = MappingProxyType(num_attempts),
        )

        return updated_attacker_state

    def _create_defender_state(self, name: str) -> MalSimDefenderState:
        """Create a new defender state, initialize values"""

        compromised_steps: set[AttackGraphNode] = set()
        for attacker_state in self._get_attacker_agents():
            compromised_steps |= attacker_state.performed_nodes

        defense_surface = self._get_defense_surface()
        defender_state = MalSimDefenderState(
            name,
            sim = self,
            performed_nodes = frozenset(self._enabled_defenses),
            compromised_nodes = frozenset(compromised_steps),
            action_surface = frozenset(defense_surface),
            step_action_surface_additions = frozenset(defense_surface),
            step_action_surface_removals = frozenset(),
            step_performed_nodes = frozenset(self._enabled_defenses),
            step_all_compromised_nodes = frozenset(compromised_steps),
            step_unviable_nodes=frozenset()
        )

        return defender_state

    def _update_defender_state(
        self,
        defender_state: MalSimDefenderState,
        step_all_compromised_nodes: set[AttackGraphNode],
        step_enabled_defenses: set[AttackGraphNode],
        step_nodes_made_unviable: set[AttackGraphNode],
    ) -> MalSimDefenderState:
        """
        Update a previous defender state based on what steps
        were enabled/compromised during last step
        """

        updated_defender_state = MalSimDefenderState(
            defender_state.name,
            sim=self,
            performed_nodes = (
                defender_state.performed_nodes | step_enabled_defenses
            ),
            compromised_nodes = frozenset(
                defender_state.compromised_nodes | step_all_compromised_nodes
            ),
            step_action_surface_additions = frozenset(),
            step_action_surface_removals = frozenset(step_enabled_defenses),
            action_surface = frozenset(self._get_defense_surface()),
            step_performed_nodes = frozenset(step_enabled_defenses),
            step_all_compromised_nodes = frozenset(step_all_compromised_nodes),
            step_unviable_nodes = frozenset(step_nodes_made_unviable)
        )

        return updated_defender_state

    def _reset_agents(self) -> None:
        """Reset agent states to a fresh start"""

        # Revive all agents and reset reward
        self._alive_agents = set(self._agent_states.keys())
        self._agent_rewards = {}

        # Create new attacker agent states
        for attacker_state in self._get_attacker_agents():
            new_attacker_state = (
                self._create_attacker_state(
                    attacker_state.name,
                    attacker_state.entry_points
                )
            )
            self._agent_states[attacker_state.name] = new_attacker_state

            # Set to initial reward
            self._agent_rewards[attacker_state.name] = (
                self._attacker_step_reward(
                    new_attacker_state, self.sim_settings.attacker_reward_mode
                )
            )
        # Create new defender agent states
        for defender_state in self._get_defender_agents():
            new_defender_state = (
                self._create_defender_state(defender_state.name)
            )
            self._agent_states[defender_state.name] = new_defender_state

            # Set to initial reward
            self._agent_rewards[defender_state.name] = (
                self._defender_step_reward(
                    new_defender_state, self.sim_settings.defender_reward_mode
                )
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
        self._agent_rewards[name] = self._attacker_step_reward(
            agent_state, self.sim_settings.attacker_reward_mode
        )

        if len(self._get_defender_agents()) > 0:
            # Need to reset defender agents when attacker agent is added
            # Since the defender stores attackers performed steps/entrypoints
            self._reset_agents()

    def register_defender(self, name: str) -> None:
        """Register a mal sim defender agent"""
        assert name not in self._agent_states, \
            f"Duplicate agent named {name} not allowed"

        agent_state = self._create_defender_state(name)
        self._agent_states[name] = agent_state
        self._alive_agents.add(name)
        self._agent_rewards[name] = self._defender_step_reward(
            agent_state, self.sim_settings.defender_reward_mode
        )

    @property
    def agent_states(self) -> dict[str, MalSimAgentState]:
        """Return read only agent state for all dead and alive agents"""
        return self._agent_states

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

    def _attempt_attacker_step(
            self, agent: MalSimAttackerState, node: AttackGraphNode
        ) -> bool:
        """Attempt a step with a TTC distribution.

        Return True if the attempt was successful.
        """

        num_attempts = agent.num_attempts[node] + 1

        if self.sim_settings.ttc_mode == TTCMode.DISABLED:
            # Always suceed if disabled TTCs
            return True

        if self.sim_settings.ttc_mode == TTCMode.EFFORT_BASED_PER_STEP_SAMPLE:
            # Run trial to decide success if config says so (SANDOR mode)
            return attempt_step_ttc(
                node, num_attempts, self.rng
            )

        if self.sim_settings.ttc_mode == TTCMode.PER_STEP_SAMPLE:
            # Sample ttc value every time if config says so (ANDREI mode)
            node_ttc_value = ttc_value_from_node(
                node,
                ProbCalculationMethod.SAMPLE,
                self._calculated_bernoullis,
                self.rng
            )
            return node_ttc_value <= 1

        # Compare attempts to ttc expected value in EXPECTED_VALUE mode
        # or presampled ttcs in PRE_SAMPLE mode
        node_ttc_value = self._ttc_values.get(node, 0)
        return num_attempts + 1 >= node_ttc_value

    def _attacker_step(
        self, agent: MalSimAttackerState, nodes: list[AttackGraphNode]
    ) -> tuple[set[AttackGraphNode], set[AttackGraphNode]]:
        """Compromise attack step nodes with attacker

        Args:
        agent - the agent to compromise nodes with
        nodes - the nodes to compromise

        Returns: set of nodes that were compromised.
        """

        successful_compromises = set()
        attempted_compromises = set()

        for node in nodes:

            assert node == self.attack_graph.nodes[node.id], (
                f"{agent.name} tried to enable a node that is not part "
                "of this simulators attack_graph. Make sure the node "
                "comes from the agents action surface."
            )

            # Compromise node if possible
            if self.node_is_traversable(agent.performed_nodes, node):

                if self._attempt_attacker_step(agent, node):
                    successful_compromises.add(node)
                    logger.info(
                        'Attacker agent "%s" compromised "%s" (reward: %d).',
                        agent.name, node.full_name, self.node_reward(node)
                    )
                else:
                    logger.info(
                        'Attacker agent "%s" attempted "%s" (attempt %d).',
                        agent.name, node.full_name, agent.num_attempts[node]
                    )
                    attempted_compromises.add(node)

            else:
                logger.warning(
                    "Attacker could not compromise %s", node.full_name
                )

        return successful_compromises, attempted_compromises

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

            if node not in agent.action_surface:
                logger.warning(
                    'Defender agent "%s" tried to step through "%s"(%d), '
                    'which is not part of its defense surface. Defender '
                    'step will skip!', agent.name, node.full_name, node.id
                )
            else:
                enabled_defenses.add(node)
                self._viability_per_node, made_unviable = make_node_unviable(
                    node,
                    self._viability_per_node,
                    self._impossible_attack_steps
                )
                attack_steps_made_unviable |= made_unviable
                logger.info(
                    'Defender agent "%s" enabled "%s" (reward: %d).',
                    agent.name, node.full_name, self.node_reward(node)
                )

        return enabled_defenses, attack_steps_made_unviable

    def _attacker_step_reward(
            self,
            attacker_state: MalSimAttackerState,
            reward_mode: RewardMode,
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
            step_reward += self.agent_reward(attacker_state.name)

        return step_reward

    def _defender_step_reward(
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
            step_reward += self.agent_reward(defender_state.name)

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

    def _defender_is_terminated(self) -> bool:
        """Check if defender is terminated
        Can be overridden by subclass for custom termination condition.
        """
        # Defender is terminated if all attackers are terminated
        return all(
            self._attacker_is_terminated(a)
            for a in self._get_attacker_agents()
        )

    def step(
        self, actions: dict[str, list[AttackGraphNode]]
    ) -> dict[str, MalSimAgentState]:
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

        if self.done():
            msg = "Simulation is done, step has no effect"
            logger.warning(msg)
            print(msg)

        for agent_name in actions:
            if agent_name not in self._agent_states:
                raise KeyError(f"No agent has name '{agent_name}'")

        self.recording[self.cur_iter] = {}

        # Populate these from the results for all agents' actions.
        step_all_compromised_nodes: set[AttackGraphNode] = set()
        step_enabled_defenses: set[AttackGraphNode] = set()
        step_nodes_made_unviable: set[AttackGraphNode] = set()

        # Perform defender actions first
        for defender_state in self._get_defender_agents(only_alive=True):
            enabled, unviable = self._defender_step(
                defender_state, actions.get(defender_state.name, [])
            )
            self.recording[self.cur_iter][defender_state.name] = list(enabled)
            step_enabled_defenses |= enabled
            step_nodes_made_unviable |= unviable

        # Perform attacker actions afterwards
        for attacker_state in self._get_attacker_agents(only_alive=True):
            agent_compromised, agent_attempted = self._attacker_step(
                attacker_state, actions.get(attacker_state.name, [])
            )
            step_all_compromised_nodes |= agent_compromised
            self.recording[self.cur_iter][attacker_state.name] = (
                list(agent_compromised)
            )

            # Update attacker state
            updated_attacker_state = self._update_attacker_state(
                attacker_state,
                agent_compromised,
                agent_attempted,
                step_nodes_made_unviable
            )
            self._agent_states[attacker_state.name] = updated_attacker_state

            # Update attacker reward
            self._agent_rewards[attacker_state.name] = (
                self._attacker_step_reward(
                    updated_attacker_state,
                    self.sim_settings.attacker_reward_mode,
                )
            )

        # Update defender states and remove 'dead' agents of any type
        for agent_name in self._alive_agents.copy():
            agent_state = self._agent_states[agent_name]

            if isinstance(agent_state, MalSimDefenderState):
                # Update defender state
                updated_defender_state = self._update_defender_state(
                    agent_state,
                    step_all_compromised_nodes,
                    step_enabled_defenses,
                    step_nodes_made_unviable
                )
                self._agent_states[agent_name] = updated_defender_state

                # Update defender reward
                self._agent_rewards[agent_state.name] = (
                    self._defender_step_reward(
                        updated_defender_state,
                        self.sim_settings.defender_reward_mode,
                    )
                )

            # Remove agents that are terminated
            if self.agent_is_terminated(agent_state.name):
                logger.info(
                    "Agent %s terminated", agent_state.name
                )
                self._alive_agents.remove(agent_state.name)

        self.cur_iter += 1
        return self.agent_states

    def render(self) -> None:
        pass


def run_simulation(
        sim: MalSimulator, agents: list[dict[str, Any]]
    ) -> dict[str, list[AttackGraphNode]]:
    """Run a simulation with agents

    Return selected actions by each agent in each step
    """
    agent_actions: dict[str, list[AttackGraphNode]] = {}
    total_rewards = {agent_dict['name']: 0.0 for agent_dict in agents}

    logger.info("Starting CLI env simulator.")
    states = sim.reset()

    while not sim.done():
        print(f"Iteration {sim.cur_iter}")
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

            agent_state = states[agent_name]
            agent_action = decision_agent.get_next_action(agent_state)

            if agent_action:
                actions[agent_name] = [agent_action]
                print(
                    f'Agent {agent_name} chose action: '
                    f'{agent_action.full_name}'
                )

                # Store agent action
                agent_actions.setdefault(agent_name, []).append(agent_action)

        # Perform next step of simulation
        states = sim.step(actions)
        for agent_dict in agents:
            agent_name = agent_dict['name']
            total_rewards[agent_name] += sim.agent_reward(agent_name)

        print("---")

    print(f"Simulation over after {sim.cur_iter} steps.")

    # Print total rewards
    for agent_dict in agents:
        agent_name = agent_dict['name']
        print(f'Total reward "{agent_name}"', total_rewards[agent_name])

    return agent_actions
