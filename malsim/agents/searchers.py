import logging

from collections import deque
from typing import Any, Deque, Dict, List, Set, Union

import numpy as np

logger = logging.getLogger(__name__)


def get_new_targets(
    observation: dict, discovered_targets: Set[int], mask: tuple
) -> List[int]:
    attack_surface = mask[1]
    surface_indexes = list(np.flatnonzero(attack_surface))
    new_targets = [idx for idx in surface_indexes if idx not in discovered_targets]
    return new_targets, surface_indexes


class PassiveAttacker:
    def compute_action_from_dict(self, observation, mask):
        return (0, None)

class BreadthFirstAttacker:
    def __init__(self, agent_config: dict) -> None:
        self.targets: Deque[int] = deque([])
        self.current_target: int = None
        seed = (
            agent_config["seed"]
            if agent_config.get("seed", None)
            else np.random.SeedSequence().entropy
        )
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get("randomize", False)
            else None
        )

    def compute_action_from_dict(self, observation: Dict[str, Any], mask: tuple):
        new_targets, surface_indexes = get_new_targets(observation, self.targets, mask)

        # Add new targets to the back of the queue
        # if desired, shuffle the new targets to make the attacker more unpredictable
        if self.rng:
            self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.appendleft(c)

        self.current_target, done = self.select_next_target(
            self.current_target, self.targets, surface_indexes
        )

        self.current_target = None if done else self.current_target
        action = 0 if done else 1
        if action == 0:
            logger.debug(
                "Attacker Breadth First agent does not have "
                "any valid targets it will terminate"
            )

        return (action, self.current_target)

    @staticmethod
    def select_next_target(
        current_target: int,
        targets: Union[List[int], Deque[int]],
        attack_surface: Set[int],
    ) -> int:
        # If the current target was not compromised, put it
        # back, but on the bottom of the stack.
        if current_target in attack_surface:
            targets.appendleft(current_target)
            current_target = targets.pop()

        while current_target not in attack_surface:
            if len(targets) == 0:
                return None, True

            current_target = targets.pop()

        return current_target, False


class DepthFirstAttacker:
    def __init__(self, agent_config: dict) -> None:
        self.current_target = -1
        self.targets: List[int] = []
        seed = (
            agent_config["seed"]
            if agent_config.get("seed", None)
            else np.random.SeedSequence().entropy
        )
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get("randomize", False)
            else None
        )

    def compute_action_from_dict(self, observation: Dict[str, Any], mask: tuple):
        new_targets, surface_indexes = get_new_targets(observation, self.targets, mask)

        # Add new targets to the top of the stack
        if self.rng:
            self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.append(c)

        self.current_target, done = self.select_next_target(
            self.current_target, self.targets, surface_indexes
        )

        self.current_target = None if done else self.current_target
        action = 0 if done else 1
        return (action, self.current_target)

    @staticmethod
    def select_next_target(
        current_target: int,
        targets: Union[List[int], Deque[int]],
        attack_surface: Set[int],
    ) -> int:
        if current_target in attack_surface:
            return current_target, False

        while current_target not in attack_surface:
            if len(targets) == 0:
                return None, True

            current_target = targets.pop()

        return current_target, False


AGENTS = {
    BreadthFirstAttacker.__name__: BreadthFirstAttacker,
    DepthFirstAttacker.__name__: DepthFirstAttacker,
}
