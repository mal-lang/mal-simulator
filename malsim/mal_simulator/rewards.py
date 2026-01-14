from collections.abc import Callable

from maltoolbox.attackgraph import AttackGraphNode
import numpy as np

from malsim.mal_simulator.agent_state import MalSimAttackerState, MalSimDefenderState
from malsim.mal_simulator.graph_utils import node_reward
from malsim.mal_simulator.settings import RewardMode, TTCMode
from malsim.mal_simulator.ttc_utils import TTCDist
from malsim.config.agent_settings import AgentSettings


def defender_step_reward(
    agent_settings: AgentSettings,
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    defender_state: MalSimDefenderState,
    rewards: dict[AttackGraphNode, float],
) -> float:
    """
    Calculate current defender reward either cumulative or one-off.
    If cumulative, sum previous and one-off reward, otherwise
    just return the one-off reward.

    Args:
    - defender_state: the defender state before defenses were enabled
    - reward_mode: which way to calculate reward
    """
    enabled_defenses = enabled_defenses_func(defender_state)
    compromised_nodes = enabled_attacks_func(defender_state)

    # Defender is penalized for compromised steps and enabled defenses
    step_reward = -sum(
        node_reward(defender_state, n)
        for n in enabled_defenses | compromised_nodes
    )

    return step_reward


def attacker_step_reward(
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    attacker_state: MalSimAttackerState,
    rng: np.random.Generator,
    reward_mode: RewardMode,
    ttc_mode: TTCMode,
) -> float:
    """
    Calculate current attacker reward either cumulative or one-off.
    If cumulative, sum previous and one-off reward, otherwise
    just return the one-off reward.

    Args:
    - attacker_state: the current attacker state
    - reward_mode: which way to calculate reward
    """

    performed_steps = performed_attacks_func(attacker_state)
    action = attacker_state.step_attempted_nodes

    # Attacker is rewarded for compromised nodes
    step_reward = sum(
        node_reward(attacker_state, n)
        for n in performed_steps
    )

    if ttc_mode != TTCMode.DISABLED:
        # If TTC Mode is not disabled, attacker is penalized for each attempt
        step_reward -= len(action)
    elif ttc_mode == TTCMode.DISABLED:
        # If TTC Mode is disabled but reward mode uses TTCs, penalize with TTCs
        for node in performed_steps:
            if reward_mode == RewardMode.EXPECTED_TTC:
                step_reward -= TTCDist.from_node(node).expected_value if node.ttc else 0
            elif reward_mode == RewardMode.SAMPLE_TTC:
                step_reward -= (
                    TTCDist.from_node(node).sample_value(rng) if node.ttc else 0
                )

    return step_reward
