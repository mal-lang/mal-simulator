from collections.abc import MutableSet
from typing import Optional
from collections.abc import Set

from maltoolbox.attackgraph import AttackGraphNode
import numpy as np

from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.attacker_state_factories import initial_attacker_state
from malsim.mal_simulator.defender_state import get_defender_agents
from malsim.mal_simulator.defender_state_factories import initial_defender_state
from malsim.mal_simulator.reset_agent import reset_agents
from malsim.mal_simulator.simulator_state import MalSimulatorState
from malsim.config.agent_settings import AttackerSettings, DefenderSettings
from malsim.types import AgentStates, AgentSettings


def register_attacker_settings(
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    alive_agents: MutableSet[str],
    agent_settings: AgentSettings,
    agent_states: AgentStates,
    sim_rng: np.random.Generator,
    attacker_settings: AttackerSettings,
) -> tuple[AgentStates, Set[str], AgentSettings]:
    """Register a mal sim attacker agent"""
    assert attacker_settings.name not in agent_settings, (
        f'Duplicate agent named {attacker_settings.name} not allowed'
    )
    alive_agents.add(attacker_settings.name)
    agent_settings[attacker_settings.name] = attacker_settings

    agent_state = initial_attacker_state(
        sim_state, sim_settings, attacker_settings, sim_rng
    )
    agent_states[attacker_settings.name] = agent_state

    if len(get_defender_agents(agent_states, alive_agents)) > 0:
        # Need to reset defender agents when attacker agent is added
        # Since the defender stores attackers performed steps/entrypoints
        agent_states, _alive_agents = reset_agents(
            sim_state,
            sim_settings,
            agent_settings,
            sim_rng,
        )
    else:
        _alive_agents = alive_agents
    return agent_states, _alive_agents, agent_settings


def register_attacker(
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    name: str,
    alive_agents: MutableSet[str],
    agent_settings: AgentSettings,
    agent_states: AgentStates,
    sim_rng: np.random.Generator,
    entry_points: Set[str] | Set[AttackGraphNode],
    goals: Optional[Set[str] | Set[AttackGraphNode]] = None,
) -> tuple[AgentStates, Set[str], AgentSettings]:
    """Register a mal sim attacker agent without settings object"""
    attacker_settings = AttackerSettings(name, entry_points, goals or set())
    return register_attacker_settings(
        sim_state,
        sim_settings,
        alive_agents,
        agent_settings,
        agent_states,
        sim_rng,
        attacker_settings,
    )


def register_defender_settings(
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    agent_states: AgentStates,
    alive_agents: MutableSet[str],
    agent_settings: AgentSettings,
    defender_settings: DefenderSettings,
    compromised_nodes: Set[AttackGraphNode],
    rng: np.random.Generator,
) -> tuple[AgentStates, Set[str], AgentSettings]:
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
        sim_settings=sim_settings,
        defender_settings=defender_settings,
        pre_compromised_nodes=compromised_nodes,
        pre_enabled_defenses=sim_state.graph_state.pre_enabled_defenses,
        rng=rng,
    )
    agent_states[defender_settings.name] = agent_state
    alive_agents.add(defender_settings.name)
    return agent_states, alive_agents, agent_settings


def register_defender(
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    agent_states: AgentStates,
    alive_agents: MutableSet[str],
    agent_settings: AgentSettings,
    _compromised_nodes: Set[AttackGraphNode],
    name: str,
    rng: np.random.Generator,
) -> tuple[AgentStates, Set[str], AgentSettings]:
    """Register a mal sim defender agent without setting object"""
    defender_settings = DefenderSettings(name)
    return register_defender_settings(
        sim_state,
        sim_settings,
        agent_states,
        alive_agents,
        agent_settings,
        defender_settings,
        _compromised_nodes,
        rng,
    )
