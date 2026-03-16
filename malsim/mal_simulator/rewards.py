from collections.abc import Callable, Set

from maltoolbox.attackgraph import AttackGraphNode
import numpy as np

from malsim.config.agent_settings import AttackerSettings, DefenderSettings
from malsim.mal_simulator.defender_state import MalSimDefenderState
from malsim.mal_simulator.attacker_state import MalSimAttackerState
from malsim.mal_simulator.graph_utils import node_reward
from malsim.config.sim_settings import RewardMode, TTCMode
from malsim.mal_simulator.ttc_utils import TTCDist


def defender_step_reward_fn(
    enabled_defenses_func: Callable[[MalSimDefenderState], Set[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], Set[AttackGraphNode]],
    defender_settings: DefenderSettings,
) -> Callable[[MalSimDefenderState], float]:
    """
    Reward function creator for defender agent.
    Functions for determining enabled defenses and compromised steps are passed in,
    as well as defender settings which includes rewards and reward mode.
    Args:
        - enabled_defenses_func: function that takes in defender state
        and returns enabled defenses.
        - enabled_attacks_func: function that takes in defender state
        and returns compromised steps.
        - defender_settings: settings for the defender,
        including rewards and reward mode
    """

    def defender_step_reward(
        defender_state: MalSimDefenderState,
    ) -> float:
        """
        Reward function for the defender. Penalizes for enabled defenses and compromised steps.
        Args:
        - defender_state: the defender state before defenses were enabled
        """
        enabled_defenses = enabled_defenses_func(defender_state)
        compromised_nodes = enabled_attacks_func(defender_state)

        # Defender is penalized for compromised steps and enabled defenses
        step_reward = -sum(
            node_reward(n, defender_settings.rewards)
            for n in enabled_defenses | compromised_nodes
        )

        return step_reward

    return defender_step_reward


def attacker_step_reward_fn(
    performed_attacks_func: Callable[[MalSimAttackerState], Set[AttackGraphNode]],
    ttc_mode: TTCMode,
    attacker_settings: AttackerSettings[AttackGraphNode],
    rng: np.random.Generator,
) -> Callable[[MalSimAttackerState], float]:
    def attacker_step_reward(
        attacker_state: MalSimAttackerState,
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
        reward_mode = attacker_settings.reward_mode
        # Attacker is rewarded for compromised nodes
        step_reward = sum(
            node_reward(n, attacker_settings.rewards) for n in performed_steps
        )

        if ttc_mode != TTCMode.DISABLED:
            # If TTC Mode is not disabled, attacker is penalized for each attempt
            step_reward -= len(action)
        elif ttc_mode == TTCMode.DISABLED:
            # If TTC Mode is disabled but reward mode uses TTCs, penalize with TTCs
            for node in performed_steps:
                if reward_mode == RewardMode.EXPECTED_TTC:
                    step_reward -= (
                        TTCDist.from_node(node).expected_value if node.ttc else 0
                    )
                elif reward_mode == RewardMode.SAMPLE_TTC:
                    step_reward -= (
                        TTCDist.from_node(node).sample_value(rng) if node.ttc else 0
                    )

        return step_reward

    return attacker_step_reward


AgentRewards = dict[str, float]
