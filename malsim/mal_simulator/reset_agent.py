from collections.abc import Callable
import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from malsim.mal_simulator.agent_state import (
    AgentRewards,
    AgentStates,
    MalSimAttackerState,
    MalSimDefenderState,
)
from malsim.mal_simulator.agent_state_factories import (
    initial_attacker_state,
    initial_defender_state,
)
from malsim.mal_simulator.rewards import attacker_step_reward, defender_step_reward
from malsim.mal_simulator.settings import MalSimulatorSettings
from malsim.mal_simulator.simulator_state import MalSimulatorState
from malsim.config.agent_settings import (
    AgentSettings,
    AttackerSettings,
    DefenderSettings,
)


def reset_agents(
    rng: np.random.Generator,
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    sim_settings: MalSimulatorSettings,
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    global_rewards: dict[AttackGraphNode, float],
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
                sim_state, attacker, sim_settings.ttc_mode, rng
            )
            pre_compromised_nodes |= new_attacker_state.step_performed_nodes
            agent_states[attacker.name] = new_attacker_state
            agent_rewards[attacker.name] = attacker_step_reward(
                performed_attacks_func=performed_attacks_func,
                attacker_state=new_attacker_state,
                rng=rng,
                reward_mode=sim_settings.attacker_reward_mode,
                ttc_mode=sim_settings.ttc_mode,
            )

    # Create new defender agent states
    for defender_settings in agent_settings.values():
        if isinstance(defender_settings, DefenderSettings):
            new_defender_state = initial_defender_state(
                sim_state,
                defender_settings,
                pre_compromised_nodes,
                sim_state.graph_state.pre_enabled_defenses,
                rng,
            )
            agent_states[defender_settings.name] = new_defender_state
            agent_rewards[defender_settings.name] = defender_step_reward(
                enabled_defenses_func,
                enabled_attacks_func,
                new_defender_state,
            )

    return agent_states, alive_agents, agent_rewards
