from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING
from collections.abc import Set

from numpy.random import default_rng

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.scenario import (
    AttackerSettings,
    DefenderSettings,
    Scenario,
)
from malsim.mal_simulator.graph_processing import make_node_unviable
from malsim.mal_simulator.ttc_utils import (
    TTCDist,
    get_impossible_attack_steps,
    attack_step_ttc_values,
)
from malsim.mal_simulator.agent_state import (
    MalSimAttackerState,
    create_attacker_state,
    MalSimDefenderState,
    create_defender_state,
)
from malsim.mal_simulator.settings import MalSimulatorSettings, TTCMode, RewardMode
from malsim.mal_simulator.graph_state import compute_initial_graph_state
from malsim.mal_simulator.node_utils import (
    full_name_or_node_to_node,
    full_names_or_nodes_to_nodes,
    full_name_dict_to_node_dict,
)
from malsim.visualization.malsim_gui_client import MalSimGUIClient

if TYPE_CHECKING:
    from malsim.agents import DecisionAgent

ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)


class MalSimulator:
    """A MAL Simulator that works on the AttackGraph

    Allows user to register agents (defender and attacker)
    and lets the agents perform actions step by step and updates
    the state of the attack graph based on the steps chosen.
    """

    def __init__(
        self,
        attack_graph: AttackGraph,
        sim_settings: MalSimulatorSettings = MalSimulatorSettings(),
        max_iter: int = ITERATIONS_LIMIT,
        rewards: Optional[dict[str, float] | dict[AttackGraphNode, float]] = None,
        false_positive_rates: Optional[
            dict[str, float] | dict[AttackGraphNode, float]
        ] = None,
        false_negative_rates: Optional[
            dict[str, float] | dict[AttackGraphNode, float]
        ] = None,
        node_actionabilities: Optional[
            dict[str, bool] | dict[AttackGraphNode, bool]
        ] = None,
        node_observabilities: Optional[
            dict[str, bool] | dict[AttackGraphNode, bool]
        ] = None,
        send_to_api: bool = False,
    ):
        """
        Args:
            attack_graph                -   The attack graph to use
            sim_settings                -   Settings for simulator
            max_iter                    -   Max iterations in simulation
        """
        logger.info('Creating Base MAL Simulator.')
        self.sim_settings = sim_settings
        self.rng = default_rng(self.sim_settings.seed)

        # Initialize the REST API client
        self.rest_api_client = None
        if send_to_api:
            self.rest_api_client = MalSimGUIClient()

        # Initialize all values
        self.attack_graph = attack_graph
        self.max_iter = max_iter  # Max iterations before stopping simulation
        self.cur_iter = 0  # Keep track on current iteration
        self.recording: dict[int, dict[str, list[AttackGraphNode]]] = {}

        # Agent related data
        self._agent_settings: dict[str, AttackerSettings | DefenderSettings] = {}
        self._agent_states: dict[str, MalSimAttackerState | MalSimDefenderState] = {}
        self._alive_agents: set[str] = set()
        self._agent_rewards: dict[str, float] = {}

        # Store graph state based on probabilities
        self._graph_state = compute_initial_graph_state(
            attack_graph, sim_settings, self.rng
        )

        # Global settings (can be overriden by each agent)
        self._rewards: dict[AttackGraphNode, float] = full_name_dict_to_node_dict(
            self, rewards or {}
        )
        self._false_positive_rates: dict[AttackGraphNode, float] = (
            full_name_dict_to_node_dict(self, false_positive_rates or {})
        )
        self._false_negative_rates: dict[AttackGraphNode, float] = (
            full_name_dict_to_node_dict(self, false_negative_rates or {})
        )
        self._node_actionabilities: dict[AttackGraphNode, bool] = (
            full_name_dict_to_node_dict(self, node_actionabilities or {})
        )
        self._node_observabilities: dict[AttackGraphNode, bool] = (
            full_name_dict_to_node_dict(self, node_observabilities or {})
        )

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario | str,
        sim_settings: MalSimulatorSettings = MalSimulatorSettings(),
        max_iter: int = ITERATIONS_LIMIT,
        register_agents: bool = True,
        send_to_api: bool = False,
        **kwargs: Any,
    ) -> MalSimulator:
        """Create a MalSimulator object from a Scenario object or file

        Args:
            scenario - a Scenario object or a path to a scenario file
            sim_settings - settings to use in the simulator
            max_iter - max number of steps to run simulation for
            register_agents - whether to register the agents from the scenario or not
            send_to_api - whether to send data to GUI REST API or not
        """

        if isinstance(scenario, str):
            # Load scenario if file was given
            scenario = Scenario.load_from_file(scenario)

        sim = MalSimulator(
            scenario.attack_graph,
            sim_settings=sim_settings,
            max_iter=max_iter,
            send_to_api=send_to_api,
            rewards=(
                scenario.rewards.per_node(scenario.attack_graph)
                if scenario.rewards
                else None
            ),
            false_positive_rates=(
                scenario.false_positive_rates.per_node(scenario.attack_graph)
                if scenario.false_positive_rates
                else None
            ),
            false_negative_rates=(
                scenario.false_negative_rates.per_node(scenario.attack_graph)
                if scenario.false_negative_rates
                else None
            ),
            node_actionabilities=(
                scenario.is_actionable.per_node(scenario.attack_graph)
                if scenario.is_actionable
                else None
            ),
            node_observabilities=(
                scenario.is_observable.per_node(scenario.attack_graph)
                if scenario.is_observable
                else None
            ),
            **kwargs,
        )

        if register_agents:
            for agent_settings in scenario.agent_settings.values():
                if isinstance(agent_settings, AttackerSettings):
                    sim.register_attacker_settings(agent_settings)
                elif isinstance(agent_settings, DefenderSettings):
                    sim.register_defender_settings(agent_settings)
        return sim

    def done(self) -> bool:
        """Return True if simulation run is done"""
        return len(self._alive_agents) == 0 or self.cur_iter > self.max_iter

    def node_ttc_value(
        self, node: AttackGraphNode | str, agent_name: Optional[str] = None
    ) -> float:
        """Return ttc value of node if it has been sampled"""
        node = full_name_or_node_to_node(self, node)
        assert self.sim_settings.ttc_mode in (
            TTCMode.PRE_SAMPLE,
            TTCMode.EXPECTED_VALUE,
        ), 'TTC value only when TTCMode is PRE_SAMPLE or EXPECTED_VALUE'

        if agent_name:
            # If agent name is given and it overrides the global TTC values
            # Return that value instead of the global ttc value
            agent_state = self._agent_states[agent_name]
            if not isinstance(agent_state, MalSimAttackerState):
                raise ValueError(
                    f'Agent {agent_name} is not an attacker and has no TTC values'
                )
            if node in agent_state.ttc_value_overrides:
                return agent_state.ttc_value_overrides[node]

        assert node in self._graph_state.ttc_values, (
            f'Node {node.full_name} does not have a ttc value'
        )
        return self._graph_state.ttc_values[node]

    def node_is_actionable(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> bool:
        agent_settings = self._agent_settings[agent_name] if agent_name else None
        if agent_settings and agent_settings.actionable_steps:
            # Actionability from agent settings
            return bool(agent_settings.actionable_steps.value(node, False))
        if self._node_actionabilities:
            # Actionability from global settings
            return self._node_actionabilities.get(node, False)
        return True

    def node_reward(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> float:
        agent_settings = self._agent_settings[agent_name] if agent_name else None
        if agent_settings and agent_settings.rewards:
            # Node reward from agent settings
            return float(agent_settings.rewards.value(node, 0.0))
        if self._rewards:
            # Node reward from global settings
            return self._rewards.get(node, 0.0)
        return 0.0

    def node_is_observable(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> bool:
        agent_settings = self._agent_settings[agent_name] if agent_name else None
        if (
            isinstance(agent_settings, DefenderSettings)
            and agent_settings.observable_steps
        ):
            # Observability from agent settings
            return bool(agent_settings.observable_steps.value(node, False))
        if self._node_observabilities:
            # Observability from global settings
            return self._node_observabilities.get(node, False)
        return True

    def node_false_positive_rate(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> float:
        agent_settings = self._agent_settings[agent_name] if agent_name else None
        if (
            isinstance(agent_settings, DefenderSettings)
            and agent_settings.false_positive_rates
        ):
            # FPR from agent settings
            return float(agent_settings.false_positive_rates.value(node, 0.0))
        if self._false_positive_rates:
            # FPR from global settings
            return self._false_positive_rates.get(node, 0.0)
        return 0.0

    def node_false_negative_rate(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> float:
        agent_settings = self._agent_settings[agent_name] if agent_name else None
        if (
            isinstance(agent_settings, DefenderSettings)
            and agent_settings.false_negative_rates
        ):
            # FNR from agent settings
            return float(agent_settings.false_negative_rates.value(node, 0.0))
        if self._false_negative_rates:
            # FNR from global settings
            return self._false_negative_rates.get(node, 0.0)
        return 0.0

    def node_is_viable(self, node: AttackGraphNode | str) -> bool:
        """Get viability of a node"""
        node = full_name_or_node_to_node(self, node)
        return self._graph_state.viability_per_node[node]

    def node_is_necessary(self, node: AttackGraphNode | str) -> bool:
        """Get necessity of a node"""
        node = full_name_or_node_to_node(self, node)
        return self._graph_state.necessity_per_node[node]

    def node_is_enabled_defense(self, node: AttackGraphNode | str) -> bool:
        """Get a nodes defense status"""
        node = full_name_or_node_to_node(self, node)
        return any(
            node in attacker_agent.performed_nodes
            for attacker_agent in self._get_defender_agents()
        )

    def node_is_compromised(self, node: AttackGraphNode | str) -> bool:
        """Return True if node is compromised by any attacker agent"""
        node = full_name_or_node_to_node(self, node)
        return any(
            node in attacker_agent.performed_nodes
            for attacker_agent in self._get_attacker_agents()
        )

    @property
    def compromised_nodes(self) -> set[AttackGraphNode]:
        compromised: set[AttackGraphNode] = set()
        for attacker in self._get_attacker_agents():
            compromised |= attacker.performed_nodes
        return compromised

    def node_is_traversable(
        self, performed_nodes: Set[AttackGraphNode], node: AttackGraphNode
    ) -> bool:
        """
        Return True or False depending if the node specified is traversable
        for given the current attacker agent state.

        A node is traversable if it is viable and:
        - if it is of type 'or' and any of its parents have been compromised
        - if it is of type 'and' and all of its necessary parents have been
          compromised

        Arguments:
        performed_nodes - the nodes we assume are compromised in this evaluation
        node            - the node we wish to evalute traversability for
        """

        if not self.node_is_viable(node):
            return False

        if node.type in ('defense', 'exist', 'notExist'):
            # Only attack steps have traversability
            return False

        if not (node.parents & performed_nodes):
            # If no parent is reached, the node can not be traversable
            return False

        if node.type == 'or':
            traversable = any(parent in performed_nodes for parent in node.parents)
        elif node.type == 'and':
            traversable = all(
                parent in performed_nodes or not self.node_is_necessary(parent)
                for parent in node.parents
            )
        else:
            raise TypeError(
                f'Node "{node.full_name}"({node.id})has an unknown type "{node.type}".'
            )
        return traversable

    def get_node(
        self, full_name: Optional[str] = None, node_id: Optional[int] = None
    ) -> AttackGraphNode:
        """Get node from attack graph by either full name or id"""

        if full_name and not node_id:
            node = self.attack_graph.get_node_by_full_name(full_name)
            assert node, f'Node with full_name {full_name} does not exist'
            return node
        if node_id and not full_name:
            node = self.attack_graph.nodes[node_id]
            assert node, f'Node with id {node_id} does not exist'
            return node

        raise ValueError("Provide either full_name or node_id to 'get_node'")

    def agent_reward(self, agent_name: str) -> float:
        """Get an agents current reward"""
        return self._agent_rewards.get(agent_name, 0)

    def agent_is_terminated(self, agent_name: str) -> bool:
        """Return True if agent was terminated"""
        agent_state = self._agent_states[agent_name]
        if isinstance(agent_state, MalSimAttackerState):
            return self._attacker_is_terminated(agent_state)
        elif isinstance(agent_state, MalSimDefenderState):
            return self._defender_is_terminated()
        else:
            raise TypeError(f'Unknown agent state for {agent_name}')

    def reset(self) -> dict[str, MalSimAttackerState | MalSimDefenderState]:
        """Reset attack graph, iteration and reinitialize agents"""
        logger.info('Resetting MAL Simulator.')

        # Re-calculate initial graph state
        self._graph_state = compute_initial_graph_state(
            self.attack_graph, self.sim_settings, self.rng
        )
        self._agent_rewards = {}

        self.cur_iter = 0
        self.recording = {}
        self._reset_agents()
        # Upload initial state to the REST API
        if self.rest_api_client:
            self.rest_api_client.upload_initial_state(self.attack_graph)

        return self.agent_states

    def get_defense_surface(self, agent_name: str) -> set[AttackGraphNode]:
        """Get the defense surface.
        All non-suppressed defense steps that are not already enabled.

        Arguments:
        graph       - the attack graph
        """
        return {
            node
            for node in self.attack_graph.defense_steps
            if self.node_is_actionable(node, agent_name)
            and self.node_is_viable(node)
            and 'suppress' not in node.tags
            and not self.node_is_enabled_defense(node)
        }

    def get_attack_surface(
        self,
        agent_name: str,
        performed_nodes: Set[AttackGraphNode],
        from_nodes: Optional[Set[AttackGraphNode]] = None,
    ) -> frozenset[AttackGraphNode]:
        """
        Calculate the attack surface of the attacker.
        If from_nodes are provided only calculate the attack surface
        stemming from those nodes, otherwise use all performed_nodes.
        The attack surface includes all of the traversable children nodes.

        Arguments:
        agent_name      - the agent to get attack surface for
        performed_nodes - the nodes the agent has performed
        from_nodes      - the nodes to calculate the attack surface from

        """

        from_nodes = from_nodes if from_nodes is not None else performed_nodes
        attack_surface = set()

        skip_compromised = self.sim_settings.attack_surface_skip_compromised
        skip_unviable = self.sim_settings.attack_surface_skip_unviable
        skip_unnecessary = self.sim_settings.attack_surface_skip_unnecessary

        for parent in from_nodes:
            for child in parent.children:
                if skip_compromised and child in performed_nodes:
                    continue

                if skip_unviable and not self.node_is_viable(child):
                    continue

                if skip_unnecessary and not self.node_is_necessary(child):
                    continue

                if not self.node_is_actionable(child, agent_name):
                    continue

                if self.node_is_traversable(performed_nodes, child):
                    attack_surface.add(child)

        return frozenset(attack_surface)

    def _generate_false_negatives(
        self, agent_name: str, observed_nodes: Set[AttackGraphNode]
    ) -> set[AttackGraphNode]:
        """Return a set of false negative attack steps from observed nodes"""
        if self._false_negative_rates:
            return set(
                node
                for node in observed_nodes
                if self.rng.random() < self.node_false_negative_rate(node, agent_name)
            )
        else:
            return set()

    def _generate_false_positives(self, agent_name: str) -> set[AttackGraphNode]:
        """Return a set of false positive attack steps from attack graph"""
        if self._false_positive_rates:
            return set(
                node
                for node in self.attack_graph.attack_steps
                if self.rng.random() < self.node_false_positive_rate(node, agent_name)
            )
        else:
            return set()

    def _defender_observed_nodes(
        self, defender_name: str, compromised_nodes: Set[AttackGraphNode]
    ) -> set[AttackGraphNode]:
        """Generate set of observed compromised nodes
        From set of compromised nodes, generate observed nodes for a defender
        in regards to observability, false negatives and false positives.
        """
        observable_steps = set(
            n for n in compromised_nodes if self.node_is_observable(n, defender_name)
        )
        false_negatives = self._generate_false_negatives(
            defender_name, compromised_nodes
        )
        false_positives = self._generate_false_positives(defender_name)

        observed_nodes = (observable_steps - false_negatives) | false_positives
        return observed_nodes

    def _initial_attacker_state(
        self,
        attacker_settings: AttackerSettings,
    ) -> MalSimAttackerState:
        """Create an attacker state from attacker settings"""
        ttc_overrides, ttc_value_overrides, impossible_steps = (
            self._attacker_overriding_ttc_settings(attacker_settings)
        )
        return create_attacker_state(
            sim=self,
            name=attacker_settings.name,
            entry_points=set(
                full_names_or_nodes_to_nodes(self, attacker_settings.entry_points)
            ),
            goals=set(full_names_or_nodes_to_nodes(self, attacker_settings.goals)),
            ttc_overrides=ttc_overrides,
            ttc_value_overrides=ttc_value_overrides,
            impossible_step_overrides=impossible_steps,
        )

    def _initial_defender_state(
        self,
        defender_settings: DefenderSettings,
        pre_compromised_nodes: set[AttackGraphNode],
        pre_enabled_defenses: set[AttackGraphNode],
    ) -> MalSimDefenderState:
        """Create a defender state from defender settings"""
        return create_defender_state(
            sim=self,
            name=defender_settings.name,
            step_compromised_nodes=pre_compromised_nodes,
            step_enabled_defenses=pre_enabled_defenses,
        )

    def _reset_agents(self) -> None:
        """Reset agent states to a fresh start"""

        # Revive all agents and reset reward
        self._alive_agents = set(self._agent_settings.keys())
        self._agent_states = {}
        self._agent_rewards = {}
        pre_compromised_nodes: set[AttackGraphNode] = set()

        # Create new attacker agent states
        for attacker in self._agent_settings.values():
            if isinstance(attacker, AttackerSettings):
                # Get any overriding ttc settings from attacker settings
                new_attacker_state = self._initial_attacker_state(attacker)
                pre_compromised_nodes |= new_attacker_state.step_performed_nodes
                self._agent_states[attacker.name] = new_attacker_state
                self._agent_rewards[attacker.name] = self._attacker_step_reward(
                    new_attacker_state, self.sim_settings.attacker_reward_mode
                )

        # Create new defender agent states
        for defender in self._agent_settings.values():
            if isinstance(defender, DefenderSettings):
                new_defender_state = self._initial_defender_state(
                    defender,
                    pre_compromised_nodes,
                    self._graph_state.pre_enabled_defenses,
                )
                self._agent_states[defender.name] = new_defender_state
                self._agent_rewards[defender.name] = self._defender_step_reward(
                    new_defender_state, self.sim_settings.defender_reward_mode
                )

    def register_attacker(
        self,
        name: str,
        entry_points: set[str] | set[AttackGraphNode],
        goals: Optional[set[str] | set[AttackGraphNode]] = None,
    ) -> None:
        """Register a mal sim attacker agent without settings object"""
        attacker_settings = AttackerSettings(name, entry_points, goals or set())
        self.register_attacker_settings(attacker_settings)

    def register_attacker_settings(self, attacker_settings: AttackerSettings) -> None:
        """Register a mal sim attacker agent"""
        assert attacker_settings.name not in self._agent_settings, (
            f'Duplicate agent named {attacker_settings.name} not allowed'
        )
        self._alive_agents.add(attacker_settings.name)
        self._agent_settings[attacker_settings.name] = attacker_settings

        agent_state = self._initial_attacker_state(attacker_settings)
        self._agent_states[attacker_settings.name] = agent_state
        self._agent_rewards[attacker_settings.name] = self._attacker_step_reward(
            agent_state, self.sim_settings.attacker_reward_mode
        )

        if len(self._get_defender_agents()) > 0:
            # Need to reset defender agents when attacker agent is added
            # Since the defender stores attackers performed steps/entrypoints
            self._reset_agents()

    def _attacker_overriding_ttc_settings(
        self, attacker_settings: AttackerSettings
    ) -> tuple[
        dict[AttackGraphNode, TTCDist],
        dict[AttackGraphNode, float],
        set[AttackGraphNode],
    ]:
        """
        Get overriding TTC distributions, TTC values, and impossible attack steps
        from attacker settings if they exist.

        Returns three separate collections:
            - a dict of TTC distributions
            - a dict of TTC values
            - a set of impossible steps
        """

        if not attacker_settings.ttc_overrides:
            return {}, {}, set()

        ttc_overrides_names = attacker_settings.ttc_overrides.per_node(
            self.attack_graph
        )

        # Convert names to TTCDist objects and map from AttackGraphNode
        # objects instead of from full names
        ttc_overrides = {
            full_name_or_node_to_node(self, node): TTCDist.from_name(name)
            for node, name in ttc_overrides_names.items()
        }
        ttc_value_overrides = attack_step_ttc_values(
            ttc_overrides.keys(),
            self.sim_settings.ttc_mode,
            self.rng,
            ttc_dists=full_name_dict_to_node_dict(self, ttc_overrides),
        )
        impossible_step_overrides = get_impossible_attack_steps(
            ttc_overrides.keys(),
            self.rng,
            ttc_dists=full_name_dict_to_node_dict(self, ttc_overrides),
        )
        return ttc_overrides, ttc_value_overrides, impossible_step_overrides

    def register_defender(self, name: str) -> None:
        """Register a mal sim defender agent without setting object"""
        defender_settings = DefenderSettings(name)
        self.register_defender_settings(defender_settings)

    def register_defender_settings(self, defender_settings: DefenderSettings) -> None:
        """Register a mal sim defender agent"""

        if self._get_defender_agents():
            print(
                'WARNING: You have registered more than one defender agent. '
                'It does not make sense to have more than one, '
                'since all defender agents have the same state.'
            )
        assert defender_settings.name not in self._agent_settings, (
            f'Duplicate agent named {defender_settings.name} not allowed'
        )

        self._agent_settings[defender_settings.name] = defender_settings

        agent_state = self._initial_defender_state(
            defender_settings,
            self.compromised_nodes,
            self._graph_state.pre_enabled_defenses,
        )
        self._agent_states[defender_settings.name] = agent_state
        self._alive_agents.add(defender_settings.name)
        self._agent_rewards[defender_settings.name] = self._defender_step_reward(
            agent_state, self.sim_settings.defender_reward_mode
        )

    @property
    def agent_states(self) -> dict[str, MalSimAttackerState | MalSimDefenderState]:
        """Return read only agent state for all dead and alive agents"""
        return self._agent_states

    def _get_attacker_agents(
        self, only_alive: bool = False
    ) -> list[MalSimAttackerState]:
        """Return list of mutable attacker agent states of attackers.
        If `only_alive` is set to True, only return the agents that are alive.
        """
        return [
            a
            for a in self._agent_states.values()
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
            a
            for a in self._agent_states.values()
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

        if node in agent.ttc_overrides:
            # If this agent has custom ttc distribution set for this node, use it
            ttc_dist = agent.ttc_overrides[node]
        else:
            ttc_dist = TTCDist.from_node(node)

        if self.sim_settings.ttc_mode == TTCMode.DISABLED:
            # Always suceed if disabled TTCs
            return True

        elif self.sim_settings.ttc_mode == TTCMode.EFFORT_BASED_PER_STEP_SAMPLE:
            # Run trial to decide success if config says so (SANDOR mode)
            return ttc_dist.attempt_ttc_with_effort(num_attempts, self.rng)

        elif self.sim_settings.ttc_mode == TTCMode.PER_STEP_SAMPLE:
            # Sample ttc value every time if config says so (ANDREI mode)
            node_ttc_value = ttc_dist.sample_value(self.rng)
            return node_ttc_value <= 1

        # Compare attempts to ttc expected value in EXPECTED_VALUE mode
        # or presampled ttcs in PRE_SAMPLE mode
        elif self.sim_settings.ttc_mode in (TTCMode.EXPECTED_VALUE, TTCMode.PRE_SAMPLE):
            node_ttc_value = self.node_ttc_value(node, agent.name)
            return num_attempts + 1 >= node_ttc_value

        else:
            raise ValueError(f'Invalid TTC mode: {self.sim_settings.ttc_mode}')

    def _attacker_step(
        self, agent: MalSimAttackerState, nodes: list[AttackGraphNode]
    ) -> tuple[list[AttackGraphNode], list[AttackGraphNode]]:
        """Compromise attack step nodes with attacker

        Args:
        agent - the agent to compromise nodes with
        nodes - the nodes to compromise

        Returns: two lists with compromised, attempted nodes
        """

        successful_compromises = list()
        attempted_compromises = list()

        for node in nodes:
            assert node == self.attack_graph.nodes[node.id], (
                f'{agent.name} tried to enable a node that is not part '
                'of this simulators attack_graph. Make sure the node '
                'comes from the agents action surface.'
            )

            if node in agent.entry_points:
                # Entrypoints can be compromised as long as they are viable
                can_compromise = self.node_is_viable(node)
            else:
                # Otherwise it is limited by traversability
                can_compromise = self.node_is_traversable(agent.performed_nodes, node)

            if can_compromise:
                if self._attempt_attacker_step(agent, node):
                    successful_compromises.append(node)
                    logger.info(
                        'Attacker agent "%s" compromised "%s" (reward: %d).',
                        agent.name,
                        node.full_name,
                        self.node_reward(node, agent.name),
                    )
                else:
                    logger.info(
                        'Attacker agent "%s" attempted "%s" (attempt %d).',
                        agent.name,
                        node.full_name,
                        agent.num_attempts[node],
                    )
                    attempted_compromises.append(node)

            else:
                logger.warning('Attacker could not compromise %s', node.full_name)

        return successful_compromises, attempted_compromises

    def _defender_step(
        self, agent: MalSimDefenderState, nodes: list[AttackGraphNode]
    ) -> tuple[list[AttackGraphNode], set[AttackGraphNode]]:
        """Enable defense step nodes with defender.

        Args:
        agent - the agent to activate defense nodes with
        nodes - the defense step nodes to enable

        Returns a tuple of a list and a set, `enabled_defenses`
        and `attack_steps_made_unviable`.
        """

        enabled_defenses = list()
        attack_steps_made_unviable = set()

        for node in nodes:
            assert node == self.attack_graph.nodes[node.id], (
                f'{agent.name} tried to enable a node that is not part '
                'of this simulators attack_graph. Make sure the node '
                'comes from the agents action surface.'
            )

            if node not in agent.action_surface:
                logger.warning(
                    'Defender agent "%s" tried to step through "%s"(%d), '
                    'which is not part of its defense surface. Defender '
                    'step will skip!',
                    agent.name,
                    node.full_name,
                    node.id,
                )
            else:
                enabled_defenses.append(node)
                self._graph_state.viability_per_node, made_unviable = (
                    make_node_unviable(
                        node,
                        self._graph_state.viability_per_node,
                        self._graph_state.impossible_attack_steps,
                    )
                )
                attack_steps_made_unviable |= made_unviable
                logger.info(
                    'Defender agent "%s" enabled "%s" (reward: %d).',
                    agent.name,
                    node.full_name,
                    self.node_reward(node, agent.name),
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
            self.node_reward(n, attacker_state.name)
            for n in attacker_state.step_performed_nodes
        )

        if self.sim_settings.ttc_mode != TTCMode.DISABLED:
            # If TTC Mode is not disabled, attacker is penalized for each attempt
            step_reward -= len(attacker_state.step_attempted_nodes)
        elif self.sim_settings.ttc_mode == TTCMode.DISABLED:
            # If TTC Mode is disabled but reward mode uses TTCs, penalize with TTCs
            for node in attacker_state.step_performed_nodes:
                if reward_mode == RewardMode.EXPECTED_TTC:
                    step_reward -= (
                        TTCDist.from_node(node).expected_value if node.ttc else 0
                    )
                elif reward_mode == RewardMode.SAMPLE_TTC:
                    step_reward -= (
                        TTCDist.from_node(node).sample_value(self.rng)
                        if node.ttc
                        else 0
                    )

        # Cumulative reward mode for attacker makes no sense
        # If I hack someones computer, do I just keep getting rewarded for it?
        # Day after day I receive some kind of time payback that allows me
        # to keep hacking?
        if reward_mode == RewardMode.CUMULATIVE:
            # To make it cumulative, add previous step reward
            step_reward += self.agent_reward(attacker_state.name)

        return step_reward

    def _defender_step_reward(
        self, defender_state: MalSimDefenderState, reward_mode: RewardMode
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
        step_compromised_nodes = defender_state.step_compromised_nodes

        # Defender is penalized for compromised steps and enabled defenses
        step_reward = -sum(
            self.node_reward(n, defender_state.name)
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

        if len(attacker_state.action_surface) == 0:
            # Attacker is terminated if it has no more actions to take
            logger.info(
                'Attacker "%s" action surface is empty, terminate', attacker_state.name
            )
            return True
        if attacker_state.goals:
            # Attacker is terminated if it has goals and all goals are met
            return (
                attacker_state.goals & attacker_state.performed_nodes
                == attacker_state.goals
            )
        # Otherwise not terminated
        return False

    def _defender_is_terminated(self) -> bool:
        """Check if defender is terminated
        Can be overridden by subclass for custom termination condition.
        """
        # Defender is terminated if all attackers are terminated
        return all(self._attacker_is_terminated(a) for a in self._get_attacker_agents())

    def _pre_step_check(
        self, actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]]
    ) -> None:
        """Do some checks before performing step to inform the users"""
        if not self._agent_states:
            msg = (
                'No agents registered, register with `.register_attacker() `'
                'and .register_defender() before stepping'
            )
            logger.warning(msg)
            print(msg)

        if self.done():
            msg = 'Simulation is done, step has no effect'
            logger.warning(msg)
            print(msg)

        for agent_name in actions:
            if agent_name not in self._agent_states:
                raise KeyError(f"No agent has name '{agent_name}'")

    def step(
        self, actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]]
    ) -> dict[str, MalSimAttackerState | MalSimDefenderState]:
        """Take a step in the simulation

        Args:
        actions - a dict mapping agent name to agent actions which is a list
                  of AttackGraphNode or full names of the nodes to perform

        Returns:
        - A dictionary containing the agent state views keyed by agent names
        """

        self.cur_iter += 1
        logger.info('Stepping through iteration %d/%d', self.cur_iter, self.max_iter)

        self._pre_step_check(actions)
        self.recording[self.cur_iter] = {}

        # Populate these from the results for all agents' actions.
        step_compromised_nodes: list[AttackGraphNode] = list()
        step_enabled_defenses: list[AttackGraphNode] = list()
        step_nodes_made_unviable: set[AttackGraphNode] = set()

        # Perform defender actions first
        for defender_state in self._get_defender_agents(only_alive=True):
            agent_actions = list(
                full_names_or_nodes_to_nodes(self, actions.get(defender_state.name, []))
            )
            enabled, unviable = self._defender_step(defender_state, agent_actions)
            self.recording[self.cur_iter][defender_state.name] = list(enabled)
            step_enabled_defenses += enabled
            step_nodes_made_unviable |= unviable

        # Perform attacker actions afterwards
        for attacker_state in self._get_attacker_agents(only_alive=True):
            agent_actions = list(
                full_names_or_nodes_to_nodes(self, actions.get(attacker_state.name, []))
            )
            agent_compromised, agent_attempted = self._attacker_step(
                attacker_state, agent_actions
            )
            step_compromised_nodes += agent_compromised
            self.recording[self.cur_iter][attacker_state.name] = list(agent_compromised)

            # Update attacker state
            updated_attacker_state = create_attacker_state(
                sim=self,
                name=attacker_state.name,
                entry_points=attacker_state.entry_points,
                goals=attacker_state.goals,
                step_compromised_nodes=frozenset(agent_compromised),
                step_attempted_nodes=frozenset(agent_attempted),
                step_nodes_made_unviable=step_nodes_made_unviable,
                previous_state=attacker_state,
            )
            self._agent_states[attacker_state.name] = updated_attacker_state

            # Update attacker reward
            self._agent_rewards[attacker_state.name] = self._attacker_step_reward(
                updated_attacker_state,
                self.sim_settings.attacker_reward_mode,
            )

        # Update defender states and remove 'dead' agents of any type
        for agent_name in self._alive_agents.copy():
            agent_state = self._agent_states[agent_name]

            if isinstance(agent_state, MalSimDefenderState):
                # Update defender state
                updated_defender_state = create_defender_state(
                    sim=self,
                    name=agent_state.name,
                    step_compromised_nodes=set(step_compromised_nodes),
                    step_enabled_defenses=set(step_enabled_defenses),
                    step_nodes_made_unviable=step_nodes_made_unviable,
                    previous_state=agent_state,
                )
                self._agent_states[agent_name] = updated_defender_state

                # Update defender reward
                self._agent_rewards[agent_state.name] = self._defender_step_reward(
                    updated_defender_state,
                    self.sim_settings.defender_reward_mode,
                )

            # Remove agents that are terminated
            if self.agent_is_terminated(agent_state.name):
                logger.info('Agent %s terminated', agent_state.name)
                self._alive_agents.remove(agent_state.name)

        if self.rest_api_client:
            self.rest_api_client.upload_performed_nodes(
                step_compromised_nodes + step_enabled_defenses, self.cur_iter
            )

        return self.agent_states

    def render(self) -> None:
        pass


def run_simulation(
    sim: MalSimulator, agents: dict[str, AttackerSettings | DefenderSettings]
) -> dict[str, list[AttackGraphNode]]:
    """Run a simulation with agents

    Return selected actions by each agent in each step
    """
    agent_actions: dict[str, list[AttackGraphNode]] = {}
    total_rewards = {agent_name: 0.0 for agent_name in agents}

    logger.info('Starting CLI env simulator.')
    states = sim.reset()

    while not sim.done():
        print(f'Iteration {sim.cur_iter}')
        actions: dict[str, list[AttackGraphNode]] = {}

        # Select actions for each agent
        for agent_name, agent_config in agents.items():
            decision_agent: Optional[DecisionAgent] = agent_config.agent
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
                print(f'Agent {agent_name} chose action: {agent_action.full_name}')

                # Store agent action
                agent_actions.setdefault(agent_name, []).append(agent_action)

        # Perform next step of simulation
        states = sim.step(actions)
        for agent_name in agents:
            total_rewards[agent_name] += sim.agent_reward(agent_name)

        print('---')

    print(f'Simulation over after {sim.cur_iter} steps.')

    # Print total rewards
    for agent_name in agents:
        print(f'Total reward "{agent_name}"', total_rewards[agent_name])

    return agent_actions
