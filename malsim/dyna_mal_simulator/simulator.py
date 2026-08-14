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
from malsim.dyna_mal_simulator.model_state import (
    reset_model_effects
)
from malsim.mal_simulator.agent_states import (
    AgentStates,
    attacker_states,
    defender_states,
)
from malsim.mal_simulator.attacker_state import AttackerState
from malsim.dyna_mal_simulator.attacker_step import dyna_attacker_step
from malsim.mal_simulator.defender_state import DefenderState
from malsim.dyna_mal_simulator.defender_step import dyna_defender_step
from malsim.mal_simulator.graph_state import compute_initial_graph_state
from malsim.mal_simulator.node_getters import (
    full_names_or_nodes_to_nodes,
)

from malsim.mal_simulator.reset_agent import reset_agents
from malsim.mal_simulator.rewards import (
    attacker_step_reward_fn,
    defender_step_reward_fn,
)
from malsim.config.agent_settings import (
    AgentSettings,
    AttackerSettings,
    DefenderSettings,
)
from malsim.mal_simulator.simulator import _pre_step_check, alive_agents
from malsim.types import (
    Recording,
)
from malsim.scenario.scenario import Scenario
from malsim.mal_simulator.attacker_state_factories import create_attacker_state
from malsim.mal_simulator.defender_state_factories import create_defender_state
from malsim.dyna_mal_simulator.simulator_state import (
    DynaMalSimulatorState,
    create_simulator_state,
    update_simulator_state,
)
from malsim.config.sim_settings import MalSimulatorSettings, RewardMode
from malsim.visualization.malsim_gui_client import MalSimGUIClient
from malsim import MalSimulator

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


class DynaMalSimulator(MalSimulator):
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

        if attack_graph.model is None:
            raise ValueError(
                'AttackGraph model is not set. DynaMalSimulator requires a model '
                'to be set.'
            )

        _attacker_settings = [a for a in agents if isinstance(a, AttackerSettings)]
        _defender_settings = [a for a in agents if isinstance(a, DefenderSettings)]

        attacker_settings_with_nodes = [
            a.convert_to_attack_graph_nodes(attack_graph) for a in _attacker_settings
        ]

        _agent_settings: AgentSettings = {
            a.name: a for a in (_defender_settings + attacker_settings_with_nodes)
        } or {}

        model_snapshot = attack_graph.model.to_dict()

        agent_states, sim_state, recording = dyna_reset(
            model_snapshot=model_snapshot,
            attack_graph=attack_graph,
            settings=sim_settings,
            agent_settings=_agent_settings,
            rng=rng,
            rest_api_client=rest_api_client,
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
        self.sim_state: DynaMalSimulatorState = sim_state
        self.recording = recording
        self.sim_settings = sim_settings
        self.agent_settings = _agent_settings
        self.rest_api_client = rest_api_client
        self._attack_graph = attack_graph
        self._sim_settings = sim_settings
        self._defender_reward_fns = defender_reward_fns
        self._attacker_reward_fns = attacker_reward_fns
        self._model_snapshot = model_snapshot

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario | str,
        send_to_api: bool = False,
        sim_settings: MalSimulatorSettings | None = None,
    ) -> DynaMalSimulator:
        """Create a DynaMalSimulator object from a Scenario object or file

        Args:
            scenario - a Scenario object or a path to a scenario file
            send_to_api - whether to send data to GUI REST API or not
        """
        return dyna_create_simulator_from_scenario(scenario, send_to_api, sim_settings)

    def reset(
        self, seed: int | None = None
    ) -> dict[str, AttackerState | DefenderState]:
        """
        Reset the simulator to the initial state.
        Optionally, a seed can be provided to re-seed the random number generator.
        """
        (
            self._agent_states,
            self.sim_state,
            self.recording,
        ) = dyna_reset(
            model_snapshot=self._model_snapshot,
            attack_graph=self._attack_graph,
            settings=self.sim_settings,
            agent_settings=self.agent_settings,
            rng=self.rng,
            rest_api_client=self.rest_api_client,
        )

        if seed is not None:
            self.rng = default_rng(seed)

        return self._agent_states

    def step(
        self, actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]]
    ) -> dict[str, AttackerState | DefenderState]:
        agent_states, recording, sim_state = dyna_step(
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


def dyna_create_simulator_from_scenario(
    scenario: str | Scenario,
    send_to_api: bool = False,
    sim_settings: MalSimulatorSettings | None = None,
) -> DynaMalSimulator:
    if isinstance(scenario, str):
        # Load scenario if file was given
        scenario = Scenario.load_from_file(scenario)

    return DynaMalSimulator(
        scenario.attack_graph,
        sim_settings=sim_settings or scenario.sim_settings,
        send_to_api=send_to_api,
        agents=scenario.agent_settings,
    )


def dyna_reset(
    model_snapshot: dict[str, Any],
    attack_graph: AttackGraph,
    settings: MalSimulatorSettings,
    agent_settings: AgentSettings,
    rng: np.random.Generator,
    rest_api_client: MalSimGUIClient | None,
) -> tuple[
    AgentStates,
    DynaMalSimulatorState,
    Recording,
]:
    """Reset attack graph and reinitialize agents"""
    logger.info('Resetting Dyna MAL Simulator.')

    # Restore the instance model and regenerate the attack graph to match
    reset_model_effects(attack_graph, model_snapshot)

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


def dyna_step(
    recording: Recording,
    sim_state: DynaMalSimulatorState,
    agent_states: AgentStates,
    rng: np.random.Generator,
    actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]],
    rest_api_client: MalSimGUIClient | None = None,
) -> tuple[AgentStates, Recording, DynaMalSimulatorState]:
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
    current_iteration = 0

    # Perform defender actions first
    for defender_state in defender_states(agent_states).values():
        agent_actions = list(
            full_names_or_nodes_to_nodes(
                sim_state.attack_graph, actions.get(defender_state.name, [])
            )
        )
        enabled, sim_state = dyna_defender_step(
            sim_state, defender_state, agent_actions, rng
        )
        current_iteration = defender_state.iteration

        recording[current_iteration][defender_state.name] = list(enabled)
        step_enabled_defenses += enabled
        sim_state = update_simulator_state(sim_state, set(step_enabled_defenses), [])

    # Perform attacker actions afterwards
    for attacker_state in attacker_states(agent_states).values():
        agent_actions = list(
            full_names_or_nodes_to_nodes(
                sim_state.attack_graph, actions.get(attacker_state.name, [])
            )
        )
        agent_compromised, agent_attempted, sim_state = dyna_attacker_step(
            sim_state, attacker_state, agent_actions, rng
        )
        current_iteration = attacker_state.iteration
        step_compromised_nodes += agent_compromised
        recording[current_iteration][attacker_state.name] = list(agent_compromised)
        sim_state = update_simulator_state(sim_state, set(), [])

        # Update attacker state
        agent_states[attacker_state.name] = create_attacker_state(
            sim_state=sim_state,
            attack_surface_settings=sim_state.settings.attack_surface,
            attacker_settings=attacker_state.settings,
            name=attacker_state.name,
            entry_points=attacker_state.entry_points,
            new_performed_nodes=frozenset(agent_compromised),
            new_attempted_nodes=frozenset(agent_attempted),
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
            defender_settings=defender_state.settings,
            new_compromised_nodes=set(step_compromised_nodes),
            new_enabled_defenses=set(step_enabled_defenses),
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
