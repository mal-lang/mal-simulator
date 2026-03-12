from collections.abc import Callable, Set

import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.sim_settings import MalSimulatorSettings
from malsim.mal_simulator.attacker_state_factories import initial_attacker_state
from malsim.mal_simulator.defender_state_factories import initial_defender_state
from malsim.mal_simulator.simulator_state import MalSimulatorState
from malsim.config.agent_settings import (
    AgentSettings,
    get_defender_settings,
    get_attacker_settings,
)
from malsim.mal_simulator.agent_states import AgentStates


def reset_attackers(
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    agent_settings: AgentSettings,
    rng: np.random.Generator,
) -> tuple[AgentStates, Set[str], Set[AttackGraphNode]]:
    """Recreate all attacker agent states"""

    attacker_states: AgentStates = {}
    alive_attackers: Set[str] = set()
    pre_compromised_nodes: Set[AttackGraphNode] = set()

    for attacker_settings in get_attacker_settings(agent_settings).values():
        # Get any overriding ttc settings from attacker settings
        new_attacker_state = initial_attacker_state(
            sim_state,
            sim_settings=sim_settings,
            attacker_settings=attacker_settings,
            rng=rng,
        )
        pre_compromised_nodes |= new_attacker_state.step_performed_nodes
        alive_attackers.add(attacker_settings.name)
        attacker_states[attacker_settings.name] = new_attacker_state

    return attacker_states, alive_attackers, pre_compromised_nodes


def reset_defenders(
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    pre_compromised_nodes: Set[AttackGraphNode],
    rng: np.random.Generator,
) -> tuple[AgentStates, Set[str]]:
    """Recreate all defender agent states"""
    defender_states: AgentStates = {}
    alive_defenders: Set[str] = set()

    for defender_settings in get_defender_settings(agent_settings).values():
        new_defender_state = initial_defender_state(
            sim_state,
            defender_settings,
            pre_compromised_nodes,
            sim_state.graph_state.pre_enabled_defenses,
            rng,
        )
        alive_defenders.add(defender_settings.name)
        defender_states[defender_settings.name] = new_defender_state

    return defender_states, alive_defenders


def reset_agents(
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    agent_settings: AgentSettings,
    rng: np.random.Generator,
) -> tuple[AgentStates, Set[str]]:
    """Reset agent states to a fresh start"""

    attacker_states, alive_attackers, pre_compromised_nodes = reset_attackers(
        sim_state, sim_settings, agent_settings, rng
    )

    defender_states, alive_defenders = reset_defenders(
        sim_state,
        agent_settings,
        pre_compromised_nodes,
        rng,
    )

    return (
        attacker_states | defender_states,
        alive_attackers | alive_defenders,
    )
