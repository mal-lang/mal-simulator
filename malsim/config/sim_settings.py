from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TTCMode(Enum):
    """
    Describes how to use the probability distributions in the attack graph.
    """

    EFFORT_BASED_PER_STEP_SAMPLE = 0
    PER_STEP_SAMPLE = 1
    PRE_SAMPLE = 2
    EXPECTED_VALUE = 3
    DISABLED = 4


class RewardMode(Enum):
    """Two different ways to generate rewards"""

    CUMULATIVE = 1  # Reward calculated on all previous steps actions
    ONE_OFF = 2  # Reward calculated only for current step actions
    EXPECTED_TTC = 3  # Penalty calculated based on expected TTC value
    SAMPLE_TTC = 4  # Penalty calculated based on sampled TTC value


@dataclass
class MalSimulatorSettings:
    """Contains settings used in MalSimulator"""

    # uncompromise_untraversable_steps
    # - Uncompromise (evict attacker) from nodes/steps that are no longer
    #   traversable (often because a defense kicked in) if set to True
    # otherwise:
    # - Leave the node/step compromised even after it becomes untraversable
    uncompromise_untraversable_steps: bool = False

    # ttc_mode
    # - mode to sample TTCs on attack steps
    ttc_mode: TTCMode = TTCMode.DISABLED

    # seed
    # - optionally run deterministic simulations with seed
    seed: Optional[int] = None
    # attack_surface_skip_compromised
    # - if true do not add already compromised nodes to the attack surface
    attack_surface_skip_compromised: bool = True
    # attack_surface_skip_unviable
    # - if true do not add unviable nodes to the attack surface
    attack_surface_skip_unviable: bool = True
    # attack_surface_skip_unnecessary
    # - if true do not add unnecessary nodes to the attack surface
    attack_surface_skip_unnecessary: bool = True
    # If set to True, each attacker compromises/performs their
    # entry point nodes at the start of the simulation
    compromise_entrypoints_at_start: bool = True

    # run_defense_step_bernoullis
    # - if true, sample defense step bernoullis to decide their initial states
    run_defense_step_bernoullis: bool = True
    # run_attack_step_bernoullis
    # - if true, sample attack step bernoullis to decide if they are impossible/exists
    run_attack_step_bernoullis: bool = True

    # Reward settings
    attacker_reward_mode: RewardMode = RewardMode.CUMULATIVE
    defender_reward_mode: RewardMode = RewardMode.CUMULATIVE

    def __post_init__(self) -> None:
        """Allow ttc/reward mode to be given as strings - convert to enums"""
        if isinstance(self.ttc_mode, str):
            self.ttc_mode = TTCMode[self.ttc_mode]
        if isinstance(self.attacker_reward_mode, str):
            self.attacker_reward_mode = RewardMode[self.attacker_reward_mode]
        if isinstance(self.defender_reward_mode, str):
            self.defender_reward_mode = RewardMode[self.defender_reward_mode]
