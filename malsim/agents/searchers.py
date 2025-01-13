import json
import logging
import re

from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PassiveAttacker:
    def __init__(self, agent_config: dict) -> None:
        pass

    def compute_action_from_dict(self, observation, mask):
        return (0, None)


class BreadthFirstAttacker:
    _insert_head: Optional[int] = 0
    name = ' '.join(re.findall(r'[A-Z][^A-Z]*', __qualname__))

    def __init__(self, agent_config: dict) -> None:
        self.targets: list[int] = []
        self.current_target: Optional[int] = None
        self.rng = None

        if agent_config.get('randomize', False):
            self.rng = np.random.default_rng(
                agent_config.get('seed', np.random.SeedSequence().entropy)
            )

    def compute_action_from_dict(
        self, observation: dict[str, Any], mask: tuple
    ) -> tuple[int, Optional[int]]:
        # mask[1] has 1s for actionable steps the agent has not compromised yet.
        attack_surface = list(np.flatnonzero(mask[1]))

        new_targets = [step for step in attack_surface if step not in self.targets]

        if self.rng:
            self.rng.shuffle(new_targets)

        for c in new_targets:
            if self._insert_head is None:
                self.targets.append(c)
            else:
                self.targets.insert(self._insert_head, c)

        # If self.current_target is not yet compromised, e.g. due to TTCs, keep
        # using that as the target, else choose a new target.
        if self.current_target not in attack_surface:
            self.current_target, done = self._select_next_target()

        if done:
            logger.debug(
                '%s agent does not have any valid targets it will terminate', self.name
            )

        return (int(not done), self.current_target)

    def _select_next_target(self) -> tuple[Optional[int], bool]:
        try:
            return self.targets.pop(), False
        except IndexError:
            return None, True


class DepthFirstAttacker(BreadthFirstAttacker):
    _insert_head = None
