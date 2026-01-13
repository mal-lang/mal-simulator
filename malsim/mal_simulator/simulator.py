from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Optional, TYPE_CHECKING
from collections.abc import Callable, Mapping, Set

import numpy as np
from numpy.random import default_rng

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.mal_simulator.attacker import (
    attacker_is_terminated,
    attacker_overriding_ttc_settings,
    attacker_step,
    attacker_step_reward,
    attempt_attacker_step,
)
from malsim.mal_simulator.defender import (
    defender_is_terminated,
    defender_step,
    defender_step_reward,
)

from malsim.mal_simulator.attack_surface import get_attack_surface
from malsim.mal_simulator.defense_surface import get_defense_surface

from malsim.mal_simulator.false_alerts import (
    generate_false_negatives,
    generate_false_positives,
    node_false_negative_rate,
    node_false_positive_rate,
)
from malsim.scenario.scenario import (
    AttackerSettings,
    DefenderSettings,
    Scenario,
)
from malsim.mal_simulator.ttc_utils import (
    TTCDist,
)
from malsim.mal_simulator.agent_state import (
    AgentRewards,
    AgentSettings,
    AgentStates,
    MalSimAttackerState,
    MalSimDefenderState,
)
from malsim.mal_simulator.agent_state_utils import (
    create_attacker_state,
    create_defender_state,
    get_attacker_agents,
    get_defender_agents
)
from malsim.mal_simulator.simulator_state import (
    MalSimulatorState
)
from malsim.mal_simulator.settings import MalSimulatorSettings, TTCMode, RewardMode
from malsim.mal_simulator.graph_state import GraphState, compute_initial_graph_state
from malsim.mal_simulator.graph_utils import (
    full_names_or_nodes_to_nodes,
    full_name_dict_to_node_dict,
    get_node,
    node_is_actionable,
    node_is_necessary,
    node_is_observable,
    node_is_traversable,
    node_is_viable,
    node_reward,
)
from malsim.mal_simulator.state_query import (
    compromised_nodes,
    node_is_compromised,
    node_is_enabled_defense,
    node_ttc_value
)
from malsim.visualization.malsim_gui_client import MalSimGUIClient

if TYPE_CHECKING:
    from malsim.agents import DecisionAgent

ITERATIONS_LIMIT = int(1e9)
logger = logging.getLogger(__name__)


PERFORMED_ATTACKS_FUNCS: Mapping[
    RewardMode,
    Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
] = {
    RewardMode.CUMULATIVE: lambda ds: ds.performed_nodes,
    RewardMode.ONE_OFF: lambda ds: ds.step_performed_nodes,
    RewardMode.EXPECTED_TTC: lambda ds: ds.step_performed_nodes,
    RewardMode.SAMPLE_TTC: lambda ds: ds.step_performed_nodes,
}

ENABLED_DEFENSES_FUNCS: Mapping[
    RewardMode, Callable[[MalSimDefenderState], frozenset[AttackGraphNode]]
] = {
    # all enabled defenses
    RewardMode.CUMULATIVE: lambda ds: ds.performed_nodes,
    # only newly enabled defenses
    # (this means that the reward actually be defined
    # as a function of the state+action
    #  but whatever)
    RewardMode.ONE_OFF: lambda ds: ds.step_performed_nodes,
}

ENABLED_ATTACKS_FUNCS: Mapping[
    RewardMode, Callable[[MalSimDefenderState], frozenset[AttackGraphNode]]
] = {
    # all performed attacks
    RewardMode.CUMULATIVE: lambda ds: ds.compromised_nodes,
    # only newly performed attacks
    RewardMode.ONE_OFF: lambda ds: ds.step_compromised_nodes,
}


Recording = dict[int, dict[str, list[AttackGraphNode]]]


def initial_attacker_state(
    sim: MalSimulator,
    ttc_mode: TTCMode,
    rng: np.random.Generator,
    sim_state: MalSimulatorState,
    attacker_settings: AttackerSettings,
) -> MalSimAttackerState:
    """Create an attacker state from attacker settings"""
    ttc_overrides, ttc_value_overrides, impossible_steps = (
        attacker_overriding_ttc_settings(sim_state.attack_graph, attacker_settings, ttc_mode, rng)
    )
    return create_attacker_state(
        agent_settings=sim.agent_settings,
        sim_state=sim_state,
        name=attacker_settings.name,
        entry_points=set(
            full_names_or_nodes_to_nodes(sim_state.attack_graph, attacker_settings.entry_points)
        ),
        goals=set(full_names_or_nodes_to_nodes(sim_state.attack_graph, attacker_settings.goals)),
        ttc_overrides=ttc_overrides,
        ttc_value_overrides=ttc_value_overrides,
        impossible_step_overrides=impossible_steps,
    )


def register_attacker_settings(
    sim: MalSimulator,
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    sim_state: MalSimulatorState,
    alive_agents: set[str],
    agent_settings: AgentSettings,
    agent_states: AgentStates,
    agent_rewards: AgentRewards,
    node_rewards: dict[AttackGraphNode, float],
    sim_settings: MalSimulatorSettings,
    sim_rng: np.random.Generator,
    attacker_settings: AttackerSettings,
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
) -> tuple[AgentStates, set[str], AgentRewards, AgentSettings]:
    """Register a mal sim attacker agent"""
    assert attacker_settings.name not in agent_settings, (
        f'Duplicate agent named {attacker_settings.name} not allowed'
    )
    alive_agents.add(attacker_settings.name)
    agent_settings[attacker_settings.name] = attacker_settings

    agent_state = initial_attacker_state(
        sim, sim_settings.ttc_mode, sim_rng, sim_state, attacker_settings
    )
    agent_states[attacker_settings.name] = agent_state
    agent_rewards[attacker_settings.name] = attacker_step_reward(
        performed_attacks_func,
        agent_state,
        sim_rng,
        agent_settings,
        sim_settings.attacker_reward_mode,
        sim_settings.ttc_mode,
        node_rewards,
    )

    if len(get_defender_agents(agent_states, alive_agents)) > 0:
        # Need to reset defender agents when attacker agent is added
        # Since the defender stores attackers performed steps/entrypoints
        agent_states, alive_agents, agent_rewards = _reset_agents(
            sim,
            sim_rng,
            sim_state,
            agent_settings,
            sim_settings,
            performed_attacks_func,
            enabled_defenses_func,
            enabled_attacks_func,
            node_rewards,
        )
    return agent_states, alive_agents, agent_rewards, agent_settings


def register_attacker(
    sim: MalSimulator,
    sim_state: MalSimulatorState,
    name: str,
    node_rewards: dict[AttackGraphNode, float],
    alive_agents: set[str],
    agent_settings: AgentSettings,
    agent_states: AgentStates,
    agent_rewards: AgentRewards,
    sim_settings: MalSimulatorSettings,
    sim_rng: np.random.Generator,
    entry_points: set[str] | set[AttackGraphNode],
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    goals: Optional[set[str] | set[AttackGraphNode]] = None,
) -> tuple[AgentStates, set[str], AgentRewards, AgentSettings]:
    """Register a mal sim attacker agent without settings object"""
    attacker_settings = AttackerSettings(name, entry_points, goals or set())
    return register_attacker_settings(
        sim,
        performed_attacks_func,
        sim_state,
        alive_agents,
        agent_settings,
        agent_states,
        agent_rewards,
        node_rewards,
        sim_settings,
        sim_rng,
        attacker_settings,
        enabled_defenses_func,
        enabled_attacks_func,
    )


def initial_defender_state(
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    name: str,
    rng: np.random.Generator,
    pre_compromised_nodes: set[AttackGraphNode],
    pre_enabled_defenses: set[AttackGraphNode],
) -> MalSimDefenderState:
    """Create a defender state from defender settings"""
    return create_defender_state(
        sim_state=sim_state,
        agent_settings=agent_settings,
        name=name,
        step_compromised_nodes=pre_compromised_nodes,
        step_enabled_defenses=pre_enabled_defenses,
        rng=rng,
    )


def register_defender_settings(
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    sim_state: MalSimulatorState,
    agent_states: AgentStates,
    alive_agents: set[str],
    agent_rewards: AgentRewards,
    agent_settings: AgentSettings,
    defender_settings: DefenderSettings,
    rewards: dict[AttackGraphNode, float],
    compromised_nodes: set[AttackGraphNode],
    rng: np.random.Generator,
) -> tuple[AgentStates, set[str], AgentRewards, AgentSettings]:
    """Register a mal sim defender agent"""

    if get_defender_agents(agent_states, alive_agents):
        print(
            'WARNING: You have registered more than one defender agent. '
            'It does not make sense to have more than one, '
            'since all defender agents have the same state.'
        )
    assert defender_settings.name not in agent_settings, (
        f'Duplicate agent named {defender_settings.name} not allowed'
    )

    agent_settings[defender_settings.name] = defender_settings

    agent_state = initial_defender_state(
        sim_state,
        agent_settings,
        defender_settings.name,
        rng,
        compromised_nodes,
        sim_state.graph_state.pre_enabled_defenses,
    )
    agent_states[defender_settings.name] = agent_state
    alive_agents.add(defender_settings.name)
    agent_rewards[defender_settings.name] = defender_step_reward(
        agent_settings,
        enabled_defenses_func,
        enabled_attacks_func,
        agent_state,
        rewards,
    )
    return agent_states, alive_agents, agent_rewards, agent_settings


def register_defender(
    sim: MalSimulator,
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    sim_state: MalSimulatorState,
    agent_states: AgentStates,
    alive_agents: set[str],
    agent_rewards: AgentRewards,
    agent_settings: AgentSettings,
    rewards: dict[AttackGraphNode, float],
    _compromised_nodes: set[AttackGraphNode],
    name: str,
) -> tuple[AgentStates, set[str], AgentRewards, AgentSettings]:
    """Register a mal sim defender agent without setting object"""
    defender_settings = DefenderSettings(name)
    return register_defender_settings(
        enabled_defenses_func,
        enabled_attacks_func,
        sim_state,
        agent_states,
        alive_agents,
        agent_rewards,
        agent_settings,
        defender_settings,
        rewards,
        _compromised_nodes,
        sim.rng
    )


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
        """
        logger.info('Creating Base MAL Simulator.')
        self.sim_settings = sim_settings
        self.rng = default_rng(self.sim_settings.seed)

        # Initialize the REST API client
        self.rest_api_client = None
        if send_to_api:
            self.rest_api_client = MalSimGUIClient()

        # Initialize all values
        self.recording: Recording = defaultdict(dict)
        self._agent_settings: dict[str, AttackerSettings | DefenderSettings] = {}
        self._agent_states: AgentStates = {}
        self._alive_agents: set[str] = set()
        self._agent_rewards: AgentRewards = {}

        # Store graph state based on probabilities
        graph_state = compute_initial_graph_state(
            attack_graph, sim_settings, self.rng
        )

        self.sim_state = MalSimulatorState(
            attack_graph,
            sim_settings,
            graph_state,
            full_name_dict_to_node_dict(attack_graph, rewards or {}),
            full_name_dict_to_node_dict(attack_graph, false_positive_rates or {}),
            full_name_dict_to_node_dict(attack_graph, false_negative_rates or {}),
            full_name_dict_to_node_dict(attack_graph, node_actionabilities or {}),
            full_name_dict_to_node_dict(attack_graph, node_observabilities or {})
        )

        self.performed_attacks_func = PERFORMED_ATTACKS_FUNCS[
            sim_settings.attacker_reward_mode
        ]
        self.enabled_defenses_func = ENABLED_DEFENSES_FUNCS[
            sim_settings.defender_reward_mode
        ]
        self.enabled_attacks_func = ENABLED_ATTACKS_FUNCS[
            sim_settings.defender_reward_mode
        ]

    @property
    def rewards(self) -> dict[AttackGraphNode, float]:
        return self.sim_state.global_rewards

    @property
    def false_positive_rates(self) -> dict[AttackGraphNode, float]:
        return self.sim_state.global_false_positive_rates

    @property
    def false_negative_rates(self) -> dict[AttackGraphNode, float]:
        return self.sim_state.global_false_negative_rates

    @property
    def agent_settings(self) -> dict[str, AttackerSettings | DefenderSettings]:
        """Return read only agent settings for all registered agents"""
        return self._agent_settings

    @property
    def alive_agents(self) -> set[str]:
        """Return read only set of alive agents"""
        return self._alive_agents

    @alive_agents.setter
    def alive_agents(self, value: set[str]) -> None:
        """Set alive agents"""
        self._alive_agents = value

    @property
    def agent_rewards(self) -> dict[str, float]:
        """Return read only agent rewards"""
        return self._agent_rewards

    @agent_rewards.setter
    def agent_rewards(self, value: dict[str, float]) -> None:
        """Set agent rewards"""
        self._agent_rewards = value

    def __getstate__(self) -> dict[str, Any]:
        do_not_pickle = {
            'performed_attacks_func',
            'enabled_defenses_func',
            'enabled_attacks_func',
        }
        return {k: v for (k, v) in self.__dict__.items() if k not in do_not_pickle}

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario | str,
        sim_settings: MalSimulatorSettings = MalSimulatorSettings(),
        register_agents: bool = True,
        send_to_api: bool = False,
        **kwargs: Any,
    ) -> MalSimulator:
        """Create a MalSimulator object from a Scenario object or file

        Args:
            scenario - a Scenario object or a path to a scenario file
            sim_settings - settings to use in the simulator
            register_agents - whether to register the agents from the scenario or not
            send_to_api - whether to send data to GUI REST API or not
        """

        if isinstance(scenario, str):
            # Load scenario if file was given
            scenario = Scenario.load_from_file(scenario)

        sim = MalSimulator(
            scenario.attack_graph,
            sim_settings=sim_settings,
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
        return done(self._alive_agents)

    def node_ttc_value(
        self, node: AttackGraphNode | str, agent_name: Optional[str] = None
    ) -> float:
        return node_ttc_value(
            self.sim_state,
            self.sim_settings.ttc_mode,
            node,
            self._agent_states[agent_name] if agent_name else None,
        )

    def node_is_actionable(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> bool:
        return node_is_actionable(
            self._agent_settings, self.sim_state.global_actionability, node, agent_name
        )

    def node_reward(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> float:
        return node_reward(self._agent_settings, self.sim_state.global_rewards, node, agent_name)

    def node_is_observable(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> bool:
        return node_is_observable(
            self._agent_settings, self.sim_state.global_observability, node, agent_name
        )

    def node_false_positive_rate(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> float:
        return node_false_positive_rate(
            self._agent_settings, self.sim_state.global_false_positive_rates, node, agent_name
        )

    def node_false_negative_rate(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> float:
        return node_false_negative_rate(
            self._agent_settings, self.sim_state.global_false_negative_rates, node, agent_name
        )

    def node_is_viable(self, node: AttackGraphNode | str) -> bool:
        return node_is_viable(self.sim_state, node)

    def node_is_necessary(self, node: AttackGraphNode | str) -> bool:
        return node_is_necessary(self.sim_state, node)

    def node_is_enabled_defense(self, node: AttackGraphNode | str) -> bool:
        return node_is_enabled_defense(
            self.sim_state.attack_graph,
            self._agent_states,
            self._alive_agents,
            node,
        )

    def node_is_compromised(self, node: AttackGraphNode | str) -> bool:
        return node_is_compromised(
            self.sim_state.attack_graph,
            self._agent_states,
            self._alive_agents,
            node,
        )

    @property
    def compromised_nodes(self) -> set[AttackGraphNode]:
        return compromised_nodes(
            self._agent_states,
            self._alive_agents,
        )

    def node_is_traversable(
        self, performed_nodes: Set[AttackGraphNode], node: AttackGraphNode
    ) -> bool:
        return node_is_traversable(self.sim_state, performed_nodes, node)

    def get_node(
        self, full_name: Optional[str] = None, node_id: Optional[int] = None
    ) -> AttackGraphNode:
        return get_node(self.sim_state.attack_graph, full_name, node_id)

    def agent_reward(self, agent_name: str) -> float:
        return agent_reward(self._agent_rewards, agent_name)

    def agent_is_terminated(self, agent_name: str) -> bool:
        return agent_is_terminated(self._agent_states, self._alive_agents, agent_name)

    def reset(self) -> dict[str, MalSimAttackerState | MalSimDefenderState]:
        agent_states, alive_agents, graph_state, agent_rewards, recording = reset(
            self,
            self.sim_state,
            self.sim_settings,
            self._agent_settings,
            self.rng,
            self.rest_api_client,
            self.performed_attacks_func,
            self.enabled_defenses_func,
            self.enabled_attacks_func,
            self.sim_state.global_rewards,
        )
        self._agent_states = agent_states
        self._alive_agents = alive_agents
        self.sim_state.graph_state = graph_state
        self._agent_rewards = agent_rewards
        self.recording = recording
        return self._agent_states

    def get_defense_surface(self, agent_name: str) -> set[AttackGraphNode]:
        return get_defense_surface(
            self._agent_settings,
            self.sim_state,
            self.sim_state.global_actionability,
            agent_name,
        )

    def get_attack_surface(
        self,
        agent_name: str,
        performed_nodes: Set[AttackGraphNode],
        from_nodes: Optional[Set[AttackGraphNode]] = None,
    ) -> frozenset[AttackGraphNode]:
        return get_attack_surface(
            self.sim_settings,
            self.sim_state,
            self._agent_settings,
            self._agent_states,
            self.sim_state.global_actionability,
            agent_name,
            performed_nodes,
            from_nodes,
        )

    def _generate_false_negatives(
        self, agent_name: str, observed_nodes: Set[AttackGraphNode]
    ) -> set[AttackGraphNode]:
        return generate_false_negatives(
            self._agent_settings,
            self.sim_state.global_false_negative_rates,
            self.rng,
            agent_name,
            observed_nodes,
        )

    def _generate_false_positives(self, agent_name: str) -> set[AttackGraphNode]:
        return generate_false_positives(
            self.sim_state.global_false_positive_rates,
            self._agent_settings,
            self.sim_state.attack_graph,
            agent_name,
            self.rng,
        )

    def _initial_attacker_state(
        self,
        attacker_settings: AttackerSettings,
    ) -> MalSimAttackerState:
        return initial_attacker_state(
            self,
            self.sim_settings.ttc_mode,
            self.rng,
            self.sim_state,
            attacker_settings,
        )

    def _initial_defender_state(
        self,
        defender_settings: DefenderSettings,
        pre_compromised_nodes: set[AttackGraphNode],
        pre_enabled_defenses: set[AttackGraphNode],
    ) -> MalSimDefenderState:
        return initial_defender_state(
            self.sim_state,
            defender_settings,
            pre_compromised_nodes,
            pre_enabled_defenses,
        )

    def _reset_agents(self) -> None:
        agent_states, alive_agents, agent_rewards = _reset_agents(
            self,
            self.rng,
            self.sim_state,
            self.agent_settings,
            self.sim_settings,
            self.performed_attacks_func,
            self.enabled_defenses_func,
            self.enabled_attacks_func,
            self.sim_state.global_rewards,
        )
        self._agent_states = agent_states
        self._alive_agents = alive_agents
        self._agent_rewards = agent_rewards

    def register_attacker(
        self,
        name: str,
        entry_points: set[str] | set[AttackGraphNode],
        goals: Optional[set[str] | set[AttackGraphNode]] = None,
    ) -> None:
        agent_states, alive_agents, agent_rewards, agent_settings = register_attacker(
            self,
            self.sim_state,
            name,
            self.sim_state.global_rewards,
            self._alive_agents,
            self._agent_settings,
            self._agent_states,
            self._agent_rewards,
            self.sim_settings,
            self.rng,
            entry_points,
            self.performed_attacks_func,
            self.enabled_defenses_func,
            self.enabled_attacks_func,
            goals,
        )
        self._agent_states = agent_states
        self._alive_agents = alive_agents
        self._agent_rewards = agent_rewards
        self._agent_settings = agent_settings

    def register_attacker_settings(self, attacker_settings: AttackerSettings) -> None:
        agent_states, alive_agents, agent_rewards, agent_settings = (
            register_attacker_settings(
                self,
                self.performed_attacks_func,
                self.sim_state,
                self._alive_agents,
                self._agent_settings,
                self._agent_states,
                self._agent_rewards,
                self.sim_state.global_rewards,
                self.sim_settings,
                self.rng,
                attacker_settings,
                self.enabled_defenses_func,
                self.enabled_attacks_func,
            )
        )
        self._agent_states = agent_states
        self._alive_agents = alive_agents
        self._agent_rewards = agent_rewards
        self._agent_settings = agent_settings

    def _attacker_overriding_ttc_settings(
        self, attacker_settings: AttackerSettings
    ) -> tuple[
        dict[AttackGraphNode, TTCDist],
        dict[AttackGraphNode, float],
        set[AttackGraphNode],
    ]:
        return attacker_overriding_ttc_settings(
            self.sim_state.attack_graph, attacker_settings, self.sim_settings.ttc_mode, self.rng
        )

    def register_defender(self, name: str) -> None:
        agent_states, alive_agents, agent_rewards, agent_settings = register_defender(
            self,
            self.enabled_defenses_func,
            self.enabled_attacks_func,
            self.sim_state,
            self._agent_states,
            self._alive_agents,
            self._agent_rewards,
            self._agent_settings,
            self.sim_state.global_rewards,
            self.compromised_nodes,
            name,
        )
        self._agent_states = agent_states
        self._alive_agents = alive_agents
        self._agent_rewards = agent_rewards
        self._agent_settings = agent_settings

    def register_defender_settings(self, defender_settings: DefenderSettings) -> None:
        agent_states, alive_agents, agent_rewards, agent_settings = (
            register_defender_settings(
                self.enabled_defenses_func,
                self.enabled_attacks_func,
                self.sim_state,
                self._agent_states,
                self._alive_agents,
                self._agent_rewards,
                self._agent_settings,
                defender_settings,
                self.sim_state.global_rewards,
                self.compromised_nodes,
                self.rng
            )
        )
        self._agent_states = agent_states
        self._alive_agents = alive_agents
        self._agent_rewards = agent_rewards
        self._agent_settings = agent_settings

    @property
    def agent_states(self) -> dict[str, MalSimAttackerState | MalSimDefenderState]:
        """Return read only agent state for all dead and alive agents"""
        return self._agent_states

    def _get_attacker_agents(
        self, only_alive: bool = False
    ) -> list[MalSimAttackerState]:
        return get_attacker_agents(
            self._agent_states,
            self._alive_agents,
            only_alive,
        )

    def _get_defender_agents(
        self, only_alive: bool = False
    ) -> list[MalSimDefenderState]:
        return get_defender_agents(
            self._agent_states,
            self._alive_agents,
            only_alive,
        )

    def _attempt_attacker_step(
        self, agent: MalSimAttackerState, node: AttackGraphNode
    ) -> bool:
        return attempt_attacker_step(
            self._agent_states,
            self.sim_state,
            self.rng,
            self.sim_settings.ttc_mode,
            agent,
            node,
        )

    def _attacker_step(
        self, agent: MalSimAttackerState, nodes: list[AttackGraphNode]
    ) -> tuple[list[AttackGraphNode], list[AttackGraphNode]]:
        return attacker_step(
            self._agent_states,
            self.rng,
            self.sim_settings.ttc_mode,
            self._agent_settings,
            self.rewards,
            self.sim_state,
            agent,
            nodes,
        )

    def _defender_step(
        self, agent: MalSimDefenderState, nodes: list[AttackGraphNode]
    ) -> tuple[list[AttackGraphNode], set[AttackGraphNode]]:
        return defender_step(
            self.sim_state.graph_state,
            agent,
            self.rewards,
            self._agent_settings,
            nodes,
            self.sim_state.attack_graph,
        )

    def _attacker_step_reward(
        self,
        attacker_state: MalSimAttackerState,
        reward_mode: RewardMode,
    ) -> float:
        return attacker_step_reward(
            self.performed_attacks_func,
            attacker_state,
            self.rng,
            self._agent_settings,
            reward_mode,
            self.sim_settings.ttc_mode,
            self.sim_state.global_rewards,
        )

    def _defender_step_reward(
        self, defender_state: MalSimDefenderState, reward_mode: RewardMode
    ) -> float:
        return defender_step_reward(
            self._agent_settings,
            self.enabled_defenses_func,
            self.enabled_attacks_func,
            defender_state,
            self.sim_state.global_rewards,
        )

    @staticmethod
    def _attacker_is_terminated(attacker_state: MalSimAttackerState) -> bool:
        return attacker_is_terminated(attacker_state)

    def _defender_is_terminated(self) -> bool:
        return defender_is_terminated(self._agent_states, self._alive_agents)

    def _pre_step_check(
        self, actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]]
    ) -> None:
        return _pre_step_check(self._agent_states, self._alive_agents, actions)

    def step(
        self, actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]]
    ) -> dict[str, MalSimAttackerState | MalSimDefenderState]:
        agent_states, recording, graph_state, agent_rewards, live_agents = step(
            self,
            self.recording,
            self.sim_state,
            self._agent_states,
            self._alive_agents,
            self._agent_settings,
            self.sim_settings,
            self.rng,
            self._alive_agents,
            self._agent_rewards,
            self.sim_state.global_rewards,
            self.performed_attacks_func,
            self.enabled_defenses_func,
            self.enabled_attacks_func,
            actions,
            self.rest_api_client,
        )
        self._agent_states = agent_states
        self.recording = recording
        self.sim_state.graph_state = graph_state
        self._agent_rewards = agent_rewards
        self._alive_agents = live_agents
        return self._agent_states

    def render(self) -> None:
        pass

    @property
    def node_actionabilities(self) -> dict[AttackGraphNode, bool]:
        return self.sim_state.global_actionability

    @property
    def node_observabilities(self) -> dict[AttackGraphNode, bool]:
        return self.sim_state.global_observability

    @property
    def graph_state(self) -> GraphState:
        return self.sim_state.graph_state

    @graph_state.setter
    def graph_state(self, value: GraphState) -> None:
        """Set graph state"""
        self.sim_state.graph_state = value


def done(alive_agents: set[str]) -> bool:
    """Return True if simulation run is done"""
    return len(alive_agents) == 0


def agent_reward(agent_rewards: dict[str, float], agent_name: str) -> float:
    """Get an agents current reward"""
    return agent_rewards.get(agent_name, 0)


def agent_is_terminated(
    agent_states: AgentStates, live_agents: set[str], agent_name: str
) -> bool:
    """Return True if agent was terminated"""
    agent_state = agent_states[agent_name]
    if isinstance(agent_state, MalSimAttackerState):
        return attacker_is_terminated(agent_state)
    elif isinstance(agent_state, MalSimDefenderState):
        return defender_is_terminated(agent_states, live_agents)
    else:
        raise TypeError(f'Unknown agent state for {agent_name}')


def reset(
    sim: MalSimulator,
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    agent_settings: AgentSettings,
    rng: np.random.Generator,
    rest_api_client: Optional[MalSimGUIClient],
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    node_rewards: dict[AttackGraphNode, float],
) -> tuple[
    AgentStates,
    set[str],
    GraphState,
    dict[str, float],
    Recording,
]:
    """Reset attack graph and reinitialize agents"""
    logger.info('Resetting MAL Simulator.')

    # Re-calculate initial graph state
    graph_state = compute_initial_graph_state(sim_state.attack_graph, sim_settings, rng)
    agent_rewards: dict[str, float] = {}

    recording: Recording = defaultdict(dict)
    agent_states, alive_agents, agent_rewards = _reset_agents(
        sim,
        rng,
        sim_state,
        agent_settings,
        sim_settings,
        performed_attacks_func,
        enabled_defenses_func,
        enabled_attacks_func,
        node_rewards,
    )
    # Upload initial state to the REST API
    if rest_api_client:
        rest_api_client.upload_initial_state(sim_state.attack_graph)

    return agent_states, alive_agents, graph_state, agent_rewards, recording


def _reset_agents(
    sim: MalSimulator,
    rng: np.random.Generator,
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    sim_settings: MalSimulatorSettings,
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    node_rewards: dict[AttackGraphNode, float],
) -> tuple[AgentStates, set[str], AgentRewards]:
    """Reset agent states to a fresh start"""

    # Revive all agents and reset reward
    alive_agents = set(agent_settings.keys())
    agent_states: AgentStates = {}
    agent_rewards: AgentRewards = {}
    pre_compromised_nodes: set[AttackGraphNode] = set()

    # Create new attacker agent states
    for attacker in agent_settings.values():
        if isinstance(attacker, AttackerSettings):
            # Get any overriding ttc settings from attacker settings
            new_attacker_state = initial_attacker_state(
                sim,
                sim_settings.ttc_mode,
                rng,
                sim_state,
                attacker,
            )
            pre_compromised_nodes |= new_attacker_state.step_performed_nodes
            agent_states[attacker.name] = new_attacker_state
            agent_rewards[attacker.name] = attacker_step_reward(
                performed_attacks_func=performed_attacks_func,
                attacker_state=new_attacker_state,
                rng=rng,
                agent_settings=agent_settings,
                reward_mode=sim_settings.attacker_reward_mode,
                ttc_mode=sim_settings.ttc_mode,
                node_rewards=node_rewards,
            )

    # Create new defender agent states
    for defender in agent_settings.values():
        if isinstance(defender, DefenderSettings):
            new_defender_state = initial_defender_state(
                sim_state,
                agent_settings,
                defender.name,
                rng,
                pre_compromised_nodes,
                sim_state.graph_state.pre_enabled_defenses,
            )
            agent_states[defender.name] = new_defender_state
            agent_rewards[defender.name] = defender_step_reward(
                agent_settings,
                enabled_defenses_func,
                enabled_attacks_func,
                new_defender_state,
                node_rewards,
            )

    return agent_states, alive_agents, agent_rewards


def _pre_step_check(
    agent_states: AgentStates,
    alive_agents: set[str],
    actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]],
) -> None:
    """Do some checks before performing step to inform the users"""
    if not agent_states:
        msg = (
            'No agents registered, register with `.register_attacker() `'
            'and .register_defender() before stepping'
        )
        logger.warning(msg)
        print(msg)

    if done(alive_agents):
        msg = 'Simulation is done, step has no effect'
        logger.warning(msg)
        print(msg)

    for agent_name in actions:
        if agent_name not in agent_states:
            raise KeyError(f"No agent has name '{agent_name}'")


def step(
    sim: MalSimulator,
    recording: Recording,
    sim_state: MalSimulatorState,
    agent_states: AgentStates,
    alive_agents: set[str],
    agent_settings: AgentSettings,
    sim_settings: MalSimulatorSettings,
    rng: np.random.Generator,
    live_agents: set[str],
    agent_rewards: dict[str, float],
    rewards: dict[AttackGraphNode, float],
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]],
    rest_api_client: Optional[MalSimGUIClient] = None,
) -> tuple[
    AgentStates,
    Recording,
    GraphState,
    dict[str, float],
    set[str],
]:
    """Take a step in the simulation

    Args:
    actions - a dict mapping agent name to agent actions which is a list
                of AttackGraphNode or full names of the nodes to perform

    Returns:
    - A dictionary containing the agent state views keyed by agent names
    """

    _pre_step_check(agent_states, live_agents, actions)

    # Populate these from the results for all agents' actions.
    step_compromised_nodes: list[AttackGraphNode] = list()
    step_enabled_defenses: list[AttackGraphNode] = list()
    step_nodes_made_unviable: set[AttackGraphNode] = set()
    current_iteration = 0

    # Perform defender actions first
    for defender_state in get_defender_agents(
        agent_states, live_agents, only_alive=True
    ):
        agent_actions = list(
            full_names_or_nodes_to_nodes(
                sim_state.attack_graph, actions.get(defender_state.name, [])
            )
        )
        enabled, unviable = defender_step(
            sim_state.graph_state,
            defender_state,
            rewards,
            agent_settings,
            agent_actions,
            sim_state.attack_graph,
        )
        current_iteration = defender_state.iteration
        logger.info(
            'Stepping through iteration %d for %s',
            current_iteration,
            defender_state.name,
        )

        recording[current_iteration][defender_state.name] = list(enabled)
        step_enabled_defenses += enabled
        step_nodes_made_unviable |= unviable

    # Perform attacker actions afterwards
    for attacker_state in get_attacker_agents(
        agent_states, live_agents, only_alive=True
    ):
        agent_actions = list(
            full_names_or_nodes_to_nodes(
                sim_state.attack_graph, actions.get(attacker_state.name, [])
            )
        )
        agent_compromised, agent_attempted = attacker_step(
            agent_states,
            rng,
            sim_settings.ttc_mode,
            agent_settings,
            rewards,
            sim_state,
            attacker_state,
            agent_actions,
        )
        current_iteration = attacker_state.iteration
        step_compromised_nodes += agent_compromised
        recording[current_iteration][attacker_state.name] = list(agent_compromised)

        # Update attacker state
        updated_attacker_state = create_attacker_state(
            sim_state=sim_state,
            agent_settings=agent_settings,
            name=attacker_state.name,
            entry_points=attacker_state.entry_points,
            goals=attacker_state.goals,
            step_compromised_nodes=frozenset(agent_compromised),
            step_attempted_nodes=frozenset(agent_attempted),
            step_nodes_made_unviable=step_nodes_made_unviable,
            previous_state=attacker_state,
        )
        agent_states[attacker_state.name] = updated_attacker_state

        # Update attacker reward
        agent_rewards[attacker_state.name] = attacker_step_reward(
            performed_attacks_func,
            updated_attacker_state,
            rng,
            agent_settings,
            sim_settings.attacker_reward_mode,
            sim_settings.ttc_mode,
            rewards,
        )

    # Update defender states and remove 'dead' agents of any type
    for agent_name in live_agents.copy():
        agent_state = agent_states[agent_name]

        if isinstance(agent_state, MalSimDefenderState):
            # Update defender state
            updated_defender_state = create_defender_state(
                sim_state=sim_state,
                agent_settings=agent_settings,
                name=agent_state.name,
                rng=rng,
                step_compromised_nodes=set(step_compromised_nodes),
                step_enabled_defenses=set(step_enabled_defenses),
                step_nodes_made_unviable=step_nodes_made_unviable,
                previous_state=agent_state,
            )
            agent_states[agent_name] = updated_defender_state

            # Update defender reward
            agent_rewards[agent_state.name] = defender_step_reward(
                agent_settings,
                enabled_defenses_func,
                enabled_attacks_func,
                updated_defender_state,
                rewards,
            )

        # Remove agents that are terminated
        if agent_is_terminated(agent_states, alive_agents, agent_state.name):
            logger.info('Agent %s terminated', agent_state.name)
            live_agents.remove(agent_state.name)

    # the way current_iteration is used here is flawed.
    if rest_api_client:
        rest_api_client.upload_performed_nodes(
            step_compromised_nodes + step_enabled_defenses,
            current_iteration,
        )

    return agent_states, recording, sim_state.graph_state, agent_rewards, live_agents


def render(sim: MalSimulator) -> None:
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
    iteration = 0
    while not sim.done():
        print(f'Iteration {iteration}')
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
        iteration += 1
        print('---')

    print(f'Simulation over after {iteration} steps.')

    # Print total rewards
    for agent_name in agents:
        print(f'Total reward "{agent_name}"', total_rewards[agent_name])

    return agent_actions
