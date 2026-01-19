from collections.abc import Callable
import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from malsim.mal_simulator.attacker_state import MalSimAttackerState
from malsim.mal_simulator.attacker_state_factories import initial_attacker_state
from malsim.mal_simulator.defender_state import MalSimDefenderState
from malsim.mal_simulator.defender_state_factories import initial_defender_state
from malsim.mal_simulator.rewards import attacker_step_reward, defender_step_reward
from malsim.mal_simulator.simulator_state import MalSimulatorState
from malsim.config.agent_settings import get_defender_settings, get_attacker_settings
from malsim.types import AgentRewards, AgentStates, AgentSettings


def reset_attackers(
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    rng: np.random.Generator,
) -> tuple[AgentStates, set[str], AgentRewards, set[AttackGraphNode]]:
    """Recreate all attacker agent states"""

    attacker_states: AgentStates = {}
    alive_attackers: set[str] = set()
    attacker_rewards: AgentRewards = {}
    pre_compromised_nodes: set[AttackGraphNode] = set()

    for attacker_settings in get_attacker_settings(agent_settings).values():
        # Get any overriding ttc settings from attacker settings
        new_attacker_state = initial_attacker_state(
            sim_state, attacker_settings, sim_state.settings.ttc_mode, rng
        )
        pre_compromised_nodes |= new_attacker_state.step_performed_nodes
        alive_attackers.add(attacker_settings.name)
        attacker_states[attacker_settings.name] = new_attacker_state
        attacker_rewards[attacker_settings.name] = attacker_step_reward(
            performed_attacks_func=performed_attacks_func,
            attacker_state=new_attacker_state,
            rng=rng,
            reward_mode=sim_state.settings.attacker_reward_mode,
            ttc_mode=sim_state.settings.ttc_mode,
        )

    return attacker_states, alive_attackers, attacker_rewards, pre_compromised_nodes


def reset_defenders(
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    pre_compromised_nodes: set[AttackGraphNode],
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    rng: np.random.Generator,
) -> tuple[AgentStates, set[str], AgentRewards]:
    """Recreate all defender agent states"""
    defender_states: AgentStates = {}
    alive_defenders: set[str] = set()
    defender_rewards: AgentRewards = {}

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
        defender_rewards[defender_settings.name] = defender_step_reward(
            enabled_defenses_func, enabled_attacks_func, new_defender_state
        )

    return defender_states, alive_defenders, defender_rewards


def reset_agents(
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    rng: np.random.Generator,
) -> tuple[AgentStates, set[str], AgentRewards]:
    """Reset agent states to a fresh start"""

    attacker_states, alive_attackers, attacker_rewards, pre_compromised_nodes = (
        reset_attackers(sim_state, agent_settings, performed_attacks_func, rng)
    )

    defender_states, alive_defenders, defender_rewards = reset_defenders(
        sim_state,
        agent_settings,
        pre_compromised_nodes,
        enabled_defenses_func,
        enabled_attacks_func,
        rng,
    )

    return (
        attacker_states | defender_states,
        alive_attackers | alive_defenders,
        attacker_rewards | defender_rewards,
    )
