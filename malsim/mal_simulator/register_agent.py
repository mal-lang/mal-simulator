from collections.abc import Callable
from typing import Optional

from maltoolbox.attackgraph import AttackGraphNode
import numpy as np

from malsim.mal_simulator.attacker_state import MalSimAttackerState
from malsim.mal_simulator.attacker_state_factories import initial_attacker_state
from malsim.mal_simulator.defender_state import MalSimDefenderState, get_defender_agents
from malsim.mal_simulator.defender_state_factories import initial_defender_state
from malsim.mal_simulator.reset_agent import reset_agents
from malsim.mal_simulator.rewards import attacker_step_reward, defender_step_reward
from malsim.mal_simulator.simulator_state import MalSimulatorState
from malsim.config.agent_settings import AttackerSettings, DefenderSettings
from malsim.mal_simulator.types import AgentRewards, AgentStates, AgentSettings


def register_attacker_settings(
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    sim_state: MalSimulatorState,
    alive_agents: set[str],
    agent_settings: AgentSettings,
    agent_states: AgentStates,
    agent_rewards: AgentRewards,
    node_rewards: dict[AttackGraphNode, float],
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
        sim_state, attacker_settings, sim_state.settings.ttc_mode, sim_rng
    )
    agent_states[attacker_settings.name] = agent_state
    agent_rewards[attacker_settings.name] = attacker_step_reward(
        performed_attacks_func,
        agent_state,
        sim_rng,
        sim_state.settings.attacker_reward_mode,
        sim_state.settings.ttc_mode,
    )

    if len(get_defender_agents(agent_states, alive_agents)) > 0:
        # Need to reset defender agents when attacker agent is added
        # Since the defender stores attackers performed steps/entrypoints
        agent_states, alive_agents, agent_rewards = reset_agents(
            sim_rng,
            sim_state,
            agent_settings,
            sim_state.settings,
            performed_attacks_func,
            enabled_defenses_func,
            enabled_attacks_func,
            node_rewards,
        )
    return agent_states, alive_agents, agent_rewards, agent_settings


def register_attacker(
    sim_state: MalSimulatorState,
    name: str,
    alive_agents: set[str],
    agent_settings: AgentSettings,
    agent_states: AgentStates,
    agent_rewards: AgentRewards,
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
        performed_attacks_func,
        sim_state,
        alive_agents,
        agent_settings,
        agent_states,
        agent_rewards,
        sim_state.global_rewards,
        sim_rng,
        attacker_settings,
        enabled_defenses_func,
        enabled_attacks_func,
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
        defender_settings,
        compromised_nodes,
        sim_state.graph_state.pre_enabled_defenses,
        rng,
    )
    agent_states[defender_settings.name] = agent_state
    alive_agents.add(defender_settings.name)
    agent_rewards[defender_settings.name] = defender_step_reward(
        enabled_defenses_func,
        enabled_attacks_func,
        agent_state,
    )
    return agent_states, alive_agents, agent_rewards, agent_settings


def register_defender(
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
    rng: np.random.Generator,
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
        rng,
    )
