from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, NamedTuple
from collections.abc import Callable, Iterable, Mapping, Set
import numpy as np
from numpy.random import default_rng

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim.config.agent_settings import defender_settings
from malsim.config.agent_settings import attacker_settings
from malsim.mal_simulator.agent_states import (
    AgentStates,
    attacker_states,
    defender_states,
)
from malsim.mal_simulator.attacker_state import AttackerState
from malsim.mal_simulator.attacker_step import (
    attacker_is_terminated,
    attacker_step,
)
from malsim.mal_simulator.defender_state import DefenderState
from malsim.mal_simulator.defender_step import (
    defender_is_terminated,
    defender_step,
)
from malsim.mal_simulator.graph_state import compute_initial_graph_state
from malsim.mal_simulator.node_getters import (
    full_names_or_nodes_to_nodes,
    get_node,
)
from malsim.mal_simulator.observability import node_is_observable

from malsim.mal_simulator.reset_agent import reset_agents
from malsim.mal_simulator.rewards import (
    attacker_step_reward_fn,
    defender_step_reward_fn,
)
from malsim.mal_simulator.false_alerts import (
    node_false_negative_rate,
    node_false_positive_rate,
)
from malsim.config.agent_settings import (
    AgentSettings,
    AttackerSettings,
    DefenderSettings,
)
from malsim.types import (
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
    Callable[[AttackerState], Set[AttackGraphNode]],
] = {
    RewardMode.CUMULATIVE: lambda ds: ds.performed_nodes,
    RewardMode.ONE_OFF: lambda ds: ds.step_performed_nodes,
    RewardMode.EXPECTED_TTC: lambda ds: ds.step_performed_nodes,
    RewardMode.SAMPLE_TTC: lambda ds: ds.step_performed_nodes,
}

ENABLED_DEFENSES_FUNCS: Mapping[
    RewardMode, Callable[[DefenderState], Set[AttackGraphNode]]
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
    RewardMode, Callable[[DefenderState], Set[AttackGraphNode]]
] = {
    # all performed attacks
    RewardMode.CUMULATIVE: lambda ds: ds.compromised_nodes,
    # only newly performed attacks
    RewardMode.ONE_OFF: lambda ds: ds.step_compromised_nodes,
}

BASE_SETTINGS = MalSimulatorSettings()


class MALSimulatorStaticData(NamedTuple):
    attack_graph: AttackGraph
    sim_settings: MalSimulatorSettings


class MalSimulator:
    """A MAL Simulator that works on the AttackGraph

    Allows user to register agents (defender and attacker)
    and lets the agents perform actions step by step and updates
    the state of the attack graph based on the steps chosen.
    """

    def __init__(
        self,
        attack_graph: AttackGraph,
        agents: Iterable[AttackerSettings[AttackGraphNode | str] | DefenderSettings],
        sim_settings: MalSimulatorSettings = BASE_SETTINGS,
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
        rng = default_rng(sim_settings.seed)
        rest_api_client = MalSimGUIClient() if send_to_api else None

        static_sim_data = MALSimulatorStaticData(
            attack_graph,
            sim_settings,
        )

        _attacker_settings = [a for a in agents if isinstance(a, AttackerSettings)]
        _defender_settings = [a for a in agents if isinstance(a, DefenderSettings)]

        attacker_settings_with_nodes = [
            a.convert_to_attack_graph_nodes(attack_graph) for a in _attacker_settings
        ]

        _agent_settings: AgentSettings = {
            a.name: a for a in (_defender_settings + attacker_settings_with_nodes)
        } or {}

        agent_states, sim_state, recording = reset(
            static_sim_data,
            _agent_settings,
            rng,
            rest_api_client,
        )

        defender_reward_fns = {
            agent_id: defender_step_reward_fn(
                ENABLED_DEFENSES_FUNCS[agent_settings.reward_mode],
                ENABLED_ATTACKS_FUNCS[agent_settings.reward_mode],
                agent_settings,
            )
            for agent_id, agent_settings in defender_settings(_agent_settings).items()
        }
        attacker_reward_fns = {
            agent_id: attacker_step_reward_fn(
                PERFORMED_ATTACKS_FUNCS[agent_setting.reward_mode],
                sim_settings.ttc_mode,
                agent_setting,
                rng,
            )
            for agent_id, agent_setting in attacker_settings(_agent_settings).items()
        }

        # Set all instance variables
        self.rng = rng
        self._agent_states = agent_states
        self.sim_state = sim_state
        self.recording = recording
        self.sim_settings = sim_settings
        self.agent_settings = _agent_settings
        self.rest_api_client = rest_api_client
        self._static_data = static_sim_data
        self._defender_reward_fns = defender_reward_fns
        self._attacker_reward_fns = attacker_reward_fns

    def __getstate__(self) -> dict[str, Any]:
        """This just ensures a pickled simulator doesn't contain some data structures"""
        do_not_pickle = {
            '_defender_reward_fns',
            '_attacker_reward_fns',
        }
        return {k: v for (k, v) in self.__dict__.items() if k not in do_not_pickle}

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario | str,
        send_to_api: bool = False,
    ) -> MalSimulator:
        """Create a MalSimulator object from a Scenario object or file

        Args:
            scenario - a Scenario object or a path to a scenario file
            sim_settings - settings to use in the simulator
            send_to_api - whether to send data to GUI REST API or not
        """
        return create_simulator_from_scenario(scenario, send_to_api)

    def done(self) -> bool:
        return done(self.alive_agents)

    @property
    def alive_agents(self) -> Set[str]:
        return alive_agents(self._agent_states)

    def node_ttc_value(
        self, node: AttackGraphNode, agent_name: str | None = None
    ) -> float:
        if agent_name:
            agent = self._agent_states[agent_name]
            assert isinstance(agent, AttackerState), (
                'TTC values only apply to attackers'
            )
            return node_ttc_value(agent, node)
        else:
            return self.sim_state.graph_state.ttc_values[node]

    def node_is_actionable(
        self, node: AttackGraphNode, agent_name: str | None = None
    ) -> bool:
        agent_actionability = (
            self.agent_settings[agent_name].actionable_steps if agent_name else None
        )
        return node_is_actionable(agent_actionability, node)

    def node_reward(self, node: AttackGraphNode, agent_name: str) -> float:
        return node_reward(node, reward_rule=self.agent_settings[agent_name].rewards)

    def node_is_observable(
        self, node: AttackGraphNode, agent_name: str | None = None
    ) -> bool:
        agent_observability = None
        if agent_name:
            agent = defender_settings(self.agent_settings)[agent_name]
            agent_observability = agent.observable_steps

        return (
            node_is_observable(agent_observability, node)
            if agent_observability
            else True
        )

    def node_false_positive_rate(
        self, node: AttackGraphNode, agent_name: str | None = None
    ) -> float:
        false_positive_rates_rule = None
        if agent_name:
            agent = defender_settings(self.agent_settings)[agent_name]
            false_positive_rates_rule = agent.false_positive_rates
        return node_false_positive_rate(node, false_positive_rates_rule)

    def node_false_negative_rate(
        self, node: AttackGraphNode, agent_name: str | None = None
    ) -> float:
        false_negative_rates_rule = None
        if agent_name:
            agent = defender_settings(self.agent_settings)[agent_name]
            false_negative_rates_rule = agent.false_negative_rates
        return node_false_negative_rate(node, false_negative_rates_rule)

    def node_is_viable(self, node: AttackGraphNode | str) -> bool:
        return node_is_viable(self.sim_state, node)

    def node_is_necessary(self, node: AttackGraphNode | str) -> bool:
        return node_is_necessary(self.sim_state, node)

    def node_is_enabled_defense(self, node: AttackGraphNode | str) -> bool:
        return node_is_enabled_defense(
            self.sim_state.attack_graph,
            self._agent_states,
            node,
        )

    def node_is_compromised(self, node: AttackGraphNode | str) -> bool:
        return node_is_compromised(
            self.sim_state.attack_graph,
            self._agent_states,
            node,
        )

    @property
    def compromised_nodes(self) -> Set[AttackGraphNode]:
        return compromised_nodes(
            self._agent_states,
        )

    def node_is_traversable(
        self, performed_nodes: Set[AttackGraphNode], node: AttackGraphNode
    ) -> bool:
        return node_is_traversable(self.sim_state, performed_nodes, node)

    def get_node(
        self, full_name: str | None = None, node_id: int | None = None
    ) -> AttackGraphNode:
        return get_node(self.sim_state.attack_graph, full_name, node_id)

    def agent_reward_by_name(self, agent_name: str) -> float:
        return self.agent_reward(agent_state=self.agent_states[agent_name])

    def agent_reward(self, agent_state: AttackerState | DefenderState) -> float:
        return self._agent_reward_from_state(agent_state) or 0.0

    def agent_is_terminated(self, agent_name: str) -> bool:
        return agent_is_terminated(self._agent_states, agent_name)

    def reset(self) -> dict[str, AttackerState | DefenderState]:
        (
            self._agent_states,
            self.sim_state,
            self.recording,
        ) = reset(
            self._static_data,
            self.agent_settings,
            self.rng,
            self.rest_api_client,
        )
        return self._agent_states

    @property
    def agent_states(self) -> dict[str, AttackerState | DefenderState]:
        """Return read only agent state for all dead and alive agents"""
        return self._agent_states

    def _defender_is_terminated(self) -> bool:
        return defender_is_terminated(self._agent_states)

    def _agent_reward_from_state(
        self, state: AttackerState | DefenderState
    ) -> float | None:
        return (
            (
                self._defender_reward_fns[state.name](state)
                if state.name in self._defender_reward_fns
                else None
            )
            if isinstance(state, DefenderState)
            else (
                self._attacker_reward_fns[state.name](state)
                if state.name in self._attacker_reward_fns
                else None
            )
        )

    def step(
        self, actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]]
    ) -> dict[str, AttackerState | DefenderState]:
        agent_states, recording, sim_state = step(
            self.recording,
            self.sim_state,
            self._agent_states,
            self.rng,
            actions,
            self.rest_api_client,
        )
        self._agent_states = agent_states
        self.recording = recording
        self.sim_state = sim_state

        return self._agent_states


def create_simulator_from_scenario(
    scenario: str | Scenario,
    send_to_api: bool = False,
) -> MalSimulator:
    if isinstance(scenario, str):
        # Load scenario if file was given
        scenario = Scenario.load_from_file(scenario)

    return MalSimulator(
        scenario.attack_graph,
        sim_settings=scenario.sim_settings,
        send_to_api=send_to_api,
        agents=scenario.agent_settings,
    )


def done(alive_agents: Set[str]) -> bool:
    """Return True if simulation run is done"""
    return len(alive_agents) == 0


def agent_is_terminated(agent_states: AgentStates, agent_name: str) -> bool:
    """Return True if agent was terminated"""
    agent_state = agent_states[agent_name]
    if isinstance(agent_state, AttackerState):
        return attacker_is_terminated(agent_state)
    elif isinstance(agent_state, DefenderState):
        return defender_is_terminated(agent_states)
    else:
        raise TypeError(f'Unknown agent state for {agent_name}')


def reset(
    static_data: MALSimulatorStaticData,
    agent_settings: AgentSettings,
    rng: np.random.Generator,
    rest_api_client: MalSimGUIClient | None,
) -> tuple[
    AgentStates,
    MalSimulatorState,
    Recording,
]:
    """Reset attack graph and reinitialize agents"""
    logger.info('Resetting MAL Simulator.')
    attack_graph = static_data.attack_graph
    settings = static_data.sim_settings

    # Re-calculate initial simulator state
    graph_state = compute_initial_graph_state(attack_graph, settings, rng)
    sim_state = create_simulator_state(attack_graph, graph_state, settings)

    agent_states = reset_agents(
        sim_state,
        settings,
        agent_settings,
        rng,
    )
    # Upload initial state to the REST API
    if rest_api_client:
        rest_api_client.upload_initial_state(attack_graph)

    return agent_states, sim_state, defaultdict(dict)


def _pre_step_check(
    agent_states: AgentStates,
    alive_agents: Set[str],
    actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]],
) -> None:
    """Do some checks before performing step to inform the users"""
    if not agent_states:
        msg = 'No agents registered'
        logger.warning(msg)

    if done(alive_agents):
        msg = 'Simulation is done but you can still step'
        logger.warning(msg)

    for agent_name in actions:
        if agent_name not in agent_states:
            raise KeyError(f"No agent has name '{agent_name}'")


def step(
    recording: Recording,
    sim_state: MalSimulatorState,
    agent_states: AgentStates,
    rng: np.random.Generator,
    actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]],
    rest_api_client: MalSimGUIClient | None = None,
) -> tuple[AgentStates, Recording, MalSimulatorState]:
    """Take a step in the simulation

    Args:
    actions - a dict mapping agent name to agent actions which is a list
                of AttackGraphNode or full names of the nodes to perform

    Returns:
    - A dictionary containing the agent state views keyed by agent names
    """

    _pre_step_check(agent_states, alive_agents(agent_states), actions)

    # Populate these from the results for all agents' actions.
    step_compromised_nodes: list[AttackGraphNode] = []
    step_enabled_defenses: list[AttackGraphNode] = []
    step_nodes_made_unviable: Set[AttackGraphNode] = set()
    current_iteration = 0

    # Perform defender actions first
    for defender_state in defender_states(agent_states).values():
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
    for attacker_state in attacker_states(agent_states).values():
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
        agent_states[attacker_state.name] = create_attacker_state(
            sim_state=sim_state,
            attack_surface_settings=sim_state.settings.attack_surface,
            attacker_settings=attacker_state.settings,
            name=attacker_state.name,
            step_compromised_nodes=frozenset(agent_compromised),
            step_attempted_nodes=frozenset(agent_attempted),
            step_nodes_made_unviable=step_nodes_made_unviable,
            previous_state=attacker_state,
            ttc_values=attacker_state.ttc_values,
            impossible_steps=attacker_state.impossible_steps,
        )

    # Update defender states and rewards
    for defender_state in defender_states(agent_states).values():
        current_iteration = defender_state.iteration
        # Update defender state
        agent_states[defender_state.name] = create_defender_state(
            sim_state=sim_state,
            name=defender_state.name,
            step_compromised_nodes=set(step_compromised_nodes),
            step_enabled_defenses=set(step_enabled_defenses),
            step_nodes_made_unviable=step_nodes_made_unviable,
            defender_settings=defender_state.settings,
            previous_state=defender_state,
            rng=rng,
        )

    # the way current_iteration is used here is flawed.
    if rest_api_client:
        rest_api_client.upload_performed_nodes(
            step_compromised_nodes + step_enabled_defenses,
            current_iteration,
        )

    return agent_states, recording, sim_state


def alive_agents(agent_states: AgentStates) -> Set[str]:
    """Return a set of alive agents"""
    return {
        agent_name
        for agent_name in agent_states
        if not agent_is_terminated(agent_states, agent_name)
    }
