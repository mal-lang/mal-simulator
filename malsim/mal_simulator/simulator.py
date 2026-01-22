from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Optional
from collections.abc import Callable, Mapping, Set

import numpy as np
from numpy.random import default_rng

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.mal_simulator.attacker_state import MalSimAttackerState, get_attacker_agents
from malsim.mal_simulator.attacker_step import (
    attacker_is_terminated,
    attacker_step,
)
from malsim.mal_simulator.defender_state import MalSimDefenderState, get_defender_agents
from malsim.mal_simulator.defender_step import (
    defender_is_terminated,
    defender_step,
)
from malsim.mal_simulator.node_getters import (
    full_names_or_nodes_to_nodes,
    get_node,
)
from malsim.mal_simulator.observability import node_is_observable
from malsim.mal_simulator.register_agent import (
    register_attacker,
    register_attacker_settings,
    register_defender,
    register_defender_settings,
)
from malsim.mal_simulator.reset_agent import reset_agents
from malsim.mal_simulator.rewards import defender_step_reward, attacker_step_reward
from malsim.mal_simulator.false_alerts import (
    node_false_negative_rate,
    node_false_positive_rate,
)
from malsim.config.agent_settings import AttackerSettings, DefenderSettings
from malsim.types import (
    AgentRewards,
    AgentStates,
    AgentSettings,
    Recording,
)
from malsim.scenario.scenario import Scenario
from malsim.mal_simulator.attacker_state_factories import create_attacker_state
from malsim.mal_simulator.defender_state_factories import create_defender_state
from malsim.mal_simulator.simulator_state import (
    MalSimulatorState,
    create_simulator_state,
)
from malsim.config.sim_settings import MalSimulatorSettings, RewardMode
from malsim.mal_simulator.graph_utils import (
    node_is_actionable,
    node_is_necessary,
    node_is_traversable,
    node_is_viable,
    node_reward,
)
from malsim.mal_simulator.state_query import (
    compromised_nodes,
    node_is_compromised,
    node_is_enabled_defense,
    node_ttc_value,
)
from malsim.visualization.malsim_gui_client import MalSimGUIClient

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

BASE_SETTINGS = MalSimulatorSettings()


class MalSimulator:
    """A MAL Simulator that works on the AttackGraph

    Allows user to register agents (defender and attacker)
    and lets the agents perform actions step by step and updates
    the state of the attack graph based on the steps chosen.
    """

    def __init__(
        self,
        attack_graph: AttackGraph,
        sim_settings: MalSimulatorSettings = BASE_SETTINGS,
        agent_settings: Optional[AgentSettings] = None,
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
            attack_graph           - The attack graph to use
            sim_settings           - Settings for simulator
            agent_settings         - The agents to pre-register
            rewards                - Global rewards per node
            false_positive_rates   - global fpr per node
            false_negative_rates   - global fnr per node
            node_actionabilities   - global actionabilities per node
            node_observabilities   - global obserabilities per node
            send_to_api            - Enable to send data to malsim-gui rest api
        """
        logger.info('Creating Base MAL Simulator.')
        self.sim_settings = sim_settings
        self.rng = default_rng(self.sim_settings.seed)

        # Initialize the REST API client
        self.rest_api_client = None
        if send_to_api:
            self.rest_api_client = MalSimGUIClient()

        self.recording: Recording = defaultdict(dict)
        self.sim_state = create_simulator_state(
            attack_graph,
            sim_settings,
            self.rng,
            rewards,
            false_positive_rates,
            false_negative_rates,
            node_actionabilities,
            node_observabilities,
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

        self._agent_settings: dict[str, AttackerSettings | DefenderSettings] = (
            agent_settings or {}
        )
        self._agent_states: AgentStates = {}
        self._alive_agents: set[str] = set()
        self._agent_rewards: AgentRewards = {}

        if self._agent_settings:
            # register agents if they were given
            self._agent_states, self._alive_agents, self._agent_rewards = reset_agents(
                self.sim_state,
                self._agent_settings,
                self.performed_attacks_func,
                self.enabled_defenses_func,
                self.enabled_attacks_func,
                self.rng,
            )

    def __getstate__(self) -> dict[str, Any]:
        """This just ensures a pickled simulator doesn't contain some data structures"""
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
        sim_settings: MalSimulatorSettings = BASE_SETTINGS,
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
        return create_simulator_from_scenario(
            scenario, sim_settings, register_agents, send_to_api, **kwargs
        )

    def done(self) -> bool:
        return done(self._alive_agents)

    def node_ttc_value(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> float:
        if agent_name:
            agent = self._agent_states[agent_name]
            assert isinstance(agent, MalSimAttackerState), (
                'TTC values only apply to attackers'
            )
            return node_ttc_value(agent, node)
        else:
            return self.sim_state.graph_state.ttc_values[node]

    def node_is_actionable(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> bool:
        agent_actionability = None
        if agent_name:
            agent_actionability = self._agent_states[agent_name].actionability_rule
        return node_is_actionable(
            agent_actionability, self.sim_state.global_actionability, node
        )

    def node_reward(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> float:
        if not agent_name:
            return self.sim_state.global_rewards.get(node, 0)
        else:
            return node_reward(self._agent_states[agent_name], node)

    def node_is_observable(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> bool:
        agent_observability = None
        if agent_name:
            agent = self._agent_states[agent_name]
            assert isinstance(agent, MalSimDefenderState), (
                'Observability only apply to defenders'
            )
            agent_observability = agent.observability_rule

        return node_is_observable(
            agent_observability, self.sim_state.global_observability, node
        )

    def node_false_positive_rate(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> float:
        false_positive_rates_rule = None
        if agent_name:
            agent = self._agent_states[agent_name]
            assert isinstance(agent, MalSimDefenderState), (
                'False positives only apply to defenders'
            )
            false_positive_rates_rule = agent.false_positive_rates_rule
        return node_false_positive_rate(
            false_positive_rates_rule, self.sim_state.global_false_positive_rates, node
        )

    def node_false_negative_rate(
        self, node: AttackGraphNode, agent_name: Optional[str] = None
    ) -> float:
        false_negative_rates_rule = None
        if agent_name:
            agent = self._agent_states[agent_name]
            assert isinstance(agent, MalSimDefenderState), (
                'False negatives only apply to defenders'
            )
            false_negative_rates_rule = agent.false_negative_rates_rule
        return node_false_negative_rate(
            false_negative_rates_rule, self.sim_state.global_false_negative_rates, node
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
        agent_states, alive_agents, sim_state, agent_rewards, recording = reset(
            self.sim_state,
            self._agent_settings,
            self.rng,
            self.rest_api_client,
            self.performed_attacks_func,
            self.enabled_defenses_func,
            self.enabled_attacks_func,
        )
        self._agent_states = agent_states
        self._alive_agents = alive_agents
        self._agent_rewards = agent_rewards
        self.recording = recording
        self.sim_state = sim_state
        return self._agent_states

    def register_attacker(
        self,
        name: str,
        entry_points: set[str] | set[AttackGraphNode],
        goals: Optional[set[str] | set[AttackGraphNode]] = None,
    ) -> None:
        agent_states, alive_agents, agent_rewards, agent_settings = register_attacker(
            self.sim_state,
            name,
            self._alive_agents,
            self._agent_settings,
            self._agent_states,
            self._agent_rewards,
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
                self.performed_attacks_func,
                self.sim_state,
                self._alive_agents,
                self._agent_settings,
                self._agent_states,
                self._agent_rewards,
                self.sim_state.global_rewards,
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

    def register_defender(self, name: str) -> None:
        agent_states, alive_agents, agent_rewards, agent_settings = register_defender(
            self.enabled_defenses_func,
            self.enabled_attacks_func,
            self.sim_state,
            self._agent_states,
            self._alive_agents,
            self._agent_rewards,
            self._agent_settings,
            self.compromised_nodes,
            name,
            self.rng,
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
                self.compromised_nodes,
                self.rng,
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

    def _defender_is_terminated(self) -> bool:
        return defender_is_terminated(self._agent_states, self._alive_agents)

    def step(
        self, actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]]
    ) -> dict[str, MalSimAttackerState | MalSimDefenderState]:
        agent_states, recording, sim_state, agent_rewards, live_agents = step(
            self.recording,
            self.sim_state,
            self._agent_states,
            self._alive_agents,
            self.rng,
            self._alive_agents,
            self._agent_rewards,
            self.performed_attacks_func,
            self.enabled_defenses_func,
            self.enabled_attacks_func,
            actions,
            self.rest_api_client,
        )
        self._agent_states = agent_states
        self.recording = recording
        self.sim_state = sim_state
        self._agent_rewards = agent_rewards
        self._alive_agents = live_agents
        return self._agent_states


def create_simulator_from_scenario(
    scenario: str | Scenario,
    sim_settings: MalSimulatorSettings,
    register_agents: bool = True,
    send_to_api: bool = False,
    **kwargs: Any,
) -> MalSimulator:
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
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    rng: np.random.Generator,
    rest_api_client: Optional[MalSimGUIClient],
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
) -> tuple[
    AgentStates,
    set[str],
    MalSimulatorState,
    dict[str, float],
    Recording,
]:
    """Reset attack graph and reinitialize agents"""
    logger.info('Resetting MAL Simulator.')

    # Re-calculate initial simulator state
    sim_state = create_simulator_state(
        sim_state.attack_graph,
        sim_state.settings,
        rng,
        sim_state.global_rewards,
        sim_state.global_false_positive_rates,
        sim_state.global_false_negative_rates,
        sim_state.global_actionability,
        sim_state.global_observability,
    )

    # Reset rewards and recording
    agent_rewards: dict[str, float] = {}
    recording: Recording = defaultdict(dict)

    agent_states, alive_agents, agent_rewards = reset_agents(
        sim_state,
        agent_settings,
        performed_attacks_func,
        enabled_defenses_func,
        enabled_attacks_func,
        rng,
    )
    # Upload initial state to the REST API
    if rest_api_client:
        rest_api_client.upload_initial_state(sim_state.attack_graph)

    return agent_states, alive_agents, sim_state, agent_rewards, recording


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
    recording: Recording,
    sim_state: MalSimulatorState,
    agent_states: AgentStates,
    alive_agents: set[str],
    rng: np.random.Generator,
    live_agents: set[str],
    agent_rewards: dict[str, float],
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]],
    rest_api_client: Optional[MalSimGUIClient] = None,
) -> tuple[AgentStates, Recording, MalSimulatorState, dict[str, float], set[str]]:
    """Take a step in the simulation

    Args:
    actions - a dict mapping agent name to agent actions which is a list
                of AttackGraphNode or full names of the nodes to perform

    Returns:
    - A dictionary containing the agent state views keyed by agent names
    """

    _pre_step_check(agent_states, live_agents, actions)

    # Populate these from the results for all agents' actions.
    step_compromised_nodes: list[AttackGraphNode] = []
    step_enabled_defenses: list[AttackGraphNode] = []
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
        enabled, unviable = defender_step(sim_state, defender_state, agent_actions)
        current_iteration = defender_state.iteration

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
            sim_state, attacker_state, agent_actions, rng
        )
        current_iteration = attacker_state.iteration
        step_compromised_nodes += agent_compromised
        recording[current_iteration][attacker_state.name] = list(agent_compromised)

        # Update attacker state
        updated_attacker_state = create_attacker_state(
            sim_state=sim_state,
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
            sim_state.settings.attacker_reward_mode,
            sim_state.settings.ttc_mode,
        )

    # Update defender states and remove 'dead' agents of any type
    for agent_name in live_agents.copy():
        agent_state = agent_states[agent_name]

        if isinstance(agent_state, MalSimDefenderState):
            current_iteration = agent_state.iteration
            # Update defender state
            updated_defender_state = create_defender_state(
                sim_state=sim_state,
                name=agent_name,
                step_compromised_nodes=set(step_compromised_nodes),
                step_enabled_defenses=set(step_enabled_defenses),
                step_nodes_made_unviable=step_nodes_made_unviable,
                previous_state=agent_state,
                rng=rng,
            )
            agent_states[agent_name] = updated_defender_state

            # Update defender reward
            agent_rewards[agent_state.name] = defender_step_reward(
                enabled_defenses_func,
                enabled_attacks_func,
                updated_defender_state,
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

    return agent_states, recording, sim_state, agent_rewards, live_agents
