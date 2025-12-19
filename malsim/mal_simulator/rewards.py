from __future__ import annotations


from typing import TypeVar
from collections.abc import Callable, Generator
from numpy.random import Generator


from maltoolbox.attackgraph import AttackGraphNode


from malsim.mal_simulator.ttc_utils import (
    TTCDist,
)
from malsim.mal_simulator.agent_state import (
    MalSimAgentState,
    MalSimAttackerState,
    MalSimDefenderState,
)
from malsim.mal_simulator.settings import TTCMode, RewardMode


DefenderRewardFunction = Callable[[MalSimDefenderState], float]
AttackerRewardFunction = Callable[[MalSimAttackerState], float]


def defender_step_reward(
    defender_state: MalSimDefenderState,
    node_reward_for_agent: Callable[[AttackGraphNode, str], float],
) -> float:
    """
    Calculate current defender reward.

    Args:
    - defender_state: the defender state before defenses were enabled
    """

    # Defender is penalized for compromised steps and enabled defenses
    return -sum(
        node_reward_for_agent(n, defender_state.name)
        for n in defender_state.performed_nodes | defender_state.compromised_nodes
    )


def attacker_step_reward(
    attacker_state: MalSimAttackerState,
    reward_mode: RewardMode,
    ttc_mode: TTCMode,
    node_reward_for_agent: Callable[[AttackGraphNode, str], float],
    rng: Generator,
) -> float:
    """
    Calculate current attacker reward either cumulative or one-off.
    If cumulative, sum previous and one-off reward, otherwise
    just return the one-off reward.

    Args:
    - attacker_state: the current attacker state
    - reward_mode: which way to calculate reward
    """

    # Attacker is rewarded for compromised nodes
    step_reward = sum(
        node_reward_for_agent(n, attacker_state.name)
        for n in attacker_state.step_performed_nodes
    )

    if ttc_mode != TTCMode.DISABLED:
        # If TTC Mode is not disabled, attacker is penalized for each attempt
        step_reward -= len(attacker_state.step_attempted_nodes)
    elif ttc_mode == TTCMode.DISABLED:
        # If TTC Mode is disabled but reward mode uses TTCs, penalize with TTCs
        for node in attacker_state.step_performed_nodes:
            if reward_mode == RewardMode.EXPECTED_TTC:
                step_reward -= TTCDist.from_node(node).expected_value if node.ttc else 0
            elif reward_mode == RewardMode.SAMPLE_TTC:
                step_reward -= (
                    TTCDist.from_node(node).sample_value(rng) if node.ttc else 0
                )

    return step_reward


V = TypeVar('V', bound=MalSimAgentState)


def cumulative_reward_func(
    f: Callable[[V], float],
    agent_return: Callable[[str], float],
) -> Callable[[V], float]:
    def g(s: V) -> float:
        return (
            f(s)  # current step reward
            + agent_return(s.name)  # previous rewards
        )

    return g
