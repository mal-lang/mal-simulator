from collections.abc import Mapping, Set


import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.sim_settings import MalSimulatorSettings
from .attacker_state import AttackerState
from malsim.mal_simulator.attacker_state_factories import initial_attacker_state
from .defender_state import DefenderState
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
) -> Mapping[str, AttackerState]:
    """Recreate all attacker agent states"""

    return {
        name: initial_attacker_state(
            sim_state,
            sim_settings=sim_settings,
            attacker_settings=attacker_settings,
            rng=rng,
        )
        for name, attacker_settings in get_attacker_settings(agent_settings).items()
    }


def reset_defenders(
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    pre_compromised_nodes: Set[AttackGraphNode],
    rng: np.random.Generator,
) -> Mapping[str, DefenderState]:
    """Recreate all defender agent states"""
    return {
        name: initial_defender_state(
            sim_state,
            defender_settings,
            pre_compromised_nodes,
            sim_state.graph_state.pre_enabled_defenses,
            rng,
        )
        for name, defender_settings in get_defender_settings(agent_settings).items()
    }


def reset_agents(
    sim_state: MalSimulatorState,
    sim_settings: MalSimulatorSettings,
    agent_settings: AgentSettings,
    rng: np.random.Generator,
) -> AgentStates:
    """Reset agent states to a fresh start"""

    attacker_states = reset_attackers(sim_state, sim_settings, agent_settings, rng)

    pre_compromised_nodes: Set[AttackGraphNode] = {
        node
        for state in attacker_states.values()
        for node in state.step_performed_nodes
    }

    defender_states = reset_defenders(
        sim_state, agent_settings, pre_compromised_nodes, rng
    )

    return {**attacker_states, **defender_states}
