from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Iterable, Optional, TYPE_CHECKING
from collections.abc import Callable, Mapping, Set

from numpy.random import default_rng

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode


from malsim.mal_simulator.agent_state import get_defender_agents, get_attacker_agents
from malsim.mal_simulator.sim_data import SimData
from malsim.mal_simulator.attacker import (
    attacker_is_terminated,
    attacker_step,
    attacker_step_reward,
    register_attacker_settings,
)
from malsim.mal_simulator.defender import (
    defender_is_terminated,
    defender_step,
    defender_step_reward,
    initial_defender_state,
    defender_is_terminated,
    register_defender_settings,
)
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
    AgentData,
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


def done(agent_data: AgentData, cur_iter: int, max_iter: int) -> bool:
    """Return True if simulation run is done"""
    return len(agent_data.alive_agents) == 0 or cur_iter > max_iter


def agent_is_terminated(agent_data: AgentData, agent_name: str) -> bool:
    """Return True if agent was terminated"""
    agent_state = agent_data.agent_states[agent_name]
    if isinstance(agent_state, MalSimAttackerState):
        return attacker_is_terminated(self, agent_state)
    elif isinstance(agent_state, MalSimDefenderState):
        return defender_is_terminated(self)
    else:
        raise TypeError(f'Unknown agent state for {agent_name}')


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
        self.attack_graph = attack_graph
        self.recording: dict[int, dict[str, list[AttackGraphNode]]] = defaultdict(dict)

        # Agent related data
        self.agent_data = AgentData()

        # Store graph state based on probabilities
        self._graph_state = compute_initial_graph_state(
            attack_graph, sim_settings, self.rng
        )

        # Global settings (can be overriden by each agent)
        self.sim_data = SimData(
            rewards=full_name_dict_to_node_dict(self.attack_graph, rewards or {}),
            false_positive_rates=(
                full_name_dict_to_node_dict(
                    self.attack_graph, false_positive_rates or {}
                )
            ),
            false_negative_rates=(
                full_name_dict_to_node_dict(
                    self.attack_graph, false_negative_rates or {}
                )
            ),
            node_actionabilities=(
                full_name_dict_to_node_dict(
                    self.attack_graph, node_actionabilities or {}
                )
            ),
            node_observabilities=(
                full_name_dict_to_node_dict(
                    self.attack_graph, node_observabilities or {}
                )
            ),
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
                    register_attacker_settings(sim, agent_settings)
                elif isinstance(agent_settings, DefenderSettings):
                    register_defender_settings(sim, agent_settings)
        return sim

    @property
    def compromised_nodes(self) -> set[AttackGraphNode]:
        compromised: set[AttackGraphNode] = set()
        for attacker in get_attacker_agents(self.agent_data):
            compromised |= attacker.performed_nodes
        return compromised

    def agent_reward(self, agent_name: str) -> float:
        """Get an agents current reward"""
        return self.agent_data.agent_rewards.get(agent_name, 0)

    def reset(self) -> dict[str, MalSimAttackerState | MalSimDefenderState]:
        """Reset attack graph and reinitialize agents"""
        logger.info('Resetting MAL Simulator.')

        # Re-calculate initial graph state
        self._graph_state = compute_initial_graph_state(
            self.attack_graph, self.sim_settings, self.rng
        )
        self._agent_rewards = {}

        self.recording = defaultdict(dict)
        self._reset_agents()
        # Upload initial state to the REST API
        if self.rest_api_client:
            self.rest_api_client.upload_initial_state(self.attack_graph)

        return self.agent_states

    def _reset_agents(self) -> None:
        """Reset agent states to a fresh start"""

        # Revive all agents and reset reward
        _alive_agents = set(self.agent_data.agent_settings.keys())
        _agent_states = {}
        _agent_rewards = {}
        pre_compromised_nodes: set[AttackGraphNode] = set()

        # Create new attacker agent states
        for attacker in self._agent_settings.values():
            if isinstance(attacker, AttackerSettings):
                # Get any overriding ttc settings from attacker settings
                new_attacker_state = initial_attacker_state(self, attacker)
                pre_compromised_nodes |= new_attacker_state.step_performed_nodes
                self._agent_states[attacker.name] = new_attacker_state
                self._agent_rewards[attacker.name] = attacker_step_reward(
                    self, new_attacker_state, self.sim_settings.attacker_reward_mode
                )

        # Create new defender agent states
        for defender in self._agent_settings.values():
            if isinstance(defender, DefenderSettings):
                new_defender_state = initial_defender_state(
                    self,
                    defender,
                    pre_compromised_nodes,
                    self._graph_state.pre_enabled_defenses,
                )
                self._agent_states[defender.name] = new_defender_state
                self._agent_rewards[defender.name] = _defender_step_reward(
                    self, new_defender_state, self.sim_settings.defender_reward_mode
                )

    @property
    def agent_states(self) -> dict[str, MalSimAttackerState | MalSimDefenderState]:
        """Return read only agent state for all dead and alive agents"""
        return self._agent_states

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

        self._pre_step_check(actions)

        # Populate these from the results for all agents' actions.
        step_compromised_nodes: list[AttackGraphNode] = list()
        step_enabled_defenses: list[AttackGraphNode] = list()
        step_nodes_made_unviable: set[AttackGraphNode] = set()
        current_iteration = 0
        recording = self.recording
        agent_states = self._agent_states
        agent_rewards = self._agent_rewards
        live_agents = self._alive_agents
        sim_settings = self.sim_settings
        rest_api_client = self.rest_api_client

        # Perform defender actions first
        for defender_state in _get_defender_agents(self, only_alive=True):
            agent_actions = list(
                full_names_or_nodes_to_nodes(
                    self.attack_graph, actions.get(defender_state.name, [])
                )
            )
            enabled, unviable = _defender_step(self, defender_state, agent_actions)
            self.recording[self.cur_iter][defender_state.name] = list(enabled)
            step_enabled_defenses += enabled
            step_nodes_made_unviable |= unviable

        # Perform attacker actions afterwards
        for attacker_state in _get_attacker_agents(self, only_alive=True):
            agent_actions = list(
                full_names_or_nodes_to_nodes(
                    self.attack_graph, actions.get(attacker_state.name, [])
                )
            )
            agent_compromised, agent_attempted = _attacker_step(
                self, attacker_state, agent_actions
            )
            current_iteration = attacker_state.iteration
            step_compromised_nodes += agent_compromised
            recording[current_iteration][attacker_state.name] = list(agent_compromised)

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
            agent_states[attacker_state.name] = updated_attacker_state

            # Update attacker reward
            self._agent_rewards[attacker_state.name] = _attacker_step_reward(
                self,
                updated_attacker_state,
                sim_settings.attacker_reward_mode,
            )

        # Update defender states and remove 'dead' agents of any type
        for agent_name in self.agent_data.alive_agents.copy():
            agent_state = self.agent_data.agent_states[agent_name]

            if isinstance(agent_state, MalSimDefenderState):
                # Update defender state
                updated_defender_state = create_defender_state(
                    self.sim_data,
                    self.attack_graph,
                    self._graph_state,
                    self.agent_data.agent_settings[agent_name],
                    step_compromised_nodes=set(step_compromised_nodes),
                    step_enabled_defenses=set(step_enabled_defenses),
                    step_nodes_made_unviable=step_nodes_made_unviable,
                    previous_state=agent_state,
                )
                agent_states[agent_name] = updated_defender_state

                # Update defender reward
                self._agent_rewards[agent_state.name] = defender_step_reward(
                    self.sim_data,
                    updated_defender_state,
                    sim_settings.defender_reward_mode,
                )

            # Remove agents that are terminated
            if agent_is_terminated(self.agent_data, agent_state.name):
                logger.info('Agent %s terminated', agent_state.name)
                live_agents.remove(agent_state.name)

        # the way current_iteration is used here is flawed.
        if rest_api_client:
            rest_api_client.upload_performed_nodes(
                step_compromised_nodes + step_enabled_defenses,
                current_iteration,
            )

        self._agent_states = agent_states
        self.recording = recording
        self._agent_rewards = agent_rewards
        self._alive_agents = live_agents
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

    while not done(sim.agent_data, sim.cur_iter, sim.max_iter):
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
        iteration += 1
        print('---')

    print(f'Simulation over after {iteration} steps.')

    # Print total rewards
    for agent_name in agents:
        print(f'Total reward "{agent_name}"', total_rewards[agent_name])

    return agent_actions
