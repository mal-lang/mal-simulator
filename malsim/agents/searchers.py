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

    default_settings = {
        'randomize': False,
        'seed': None,
    }

    def __init__(self, agent_config: dict) -> None:
        self.targets: list[int] = []
        self.current_target: Optional[int] = None

        self.attack_graph = agent_config.pop('attack_graph')
        self.settings = self.default_settings | agent_config

        self.rng = np.random.default_rng(
            self.settings['seed'] or np.random.SeedSequence()
        )

    def compute_action_from_dict(
        self, observation: dict[str, Any], mask: tuple
    ) -> tuple[int, Optional[int]]:
        # mask[1] has 1s for actionable steps the agent has not compromised yet.
        attack_surface = list(np.flatnonzero(mask[1]))

        new_targets = [step for step in attack_surface if step not in self.targets]

        if self.settings['randomize']:
            self.rng.shuffle(new_targets)

        for c in new_targets:
            if self._insert_head is None:
                self.targets.append(c)
            else:
                self.targets.insert(self._insert_head, c)

        self.current_target, done = self._select_next_target()

        if done:
            logger.debug(
                '%s agent does not have any valid targets it will terminate', self.name
            )

        return (int(not done), self.current_target)

    def _select_next_target(self) -> tuple[Optional[int], bool]:
        if self.current_target in self.targets:
            # If self.current_target is not yet compromised, e.g. due to TTCs,
            # keep using that as the target.
            self.targets.remove(self.current_target)
            self.targets.append(self.current_target)

        try:
            return self.targets.pop(), False
        except IndexError:
            return None, True


class DepthFirstAttacker(BreadthFirstAttacker):
    _insert_head = None
