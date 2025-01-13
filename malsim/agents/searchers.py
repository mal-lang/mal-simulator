import json
import logging
import pprint
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
        'wait_factor': 0,
    }

    def __init__(self, agent_config: dict) -> None:
        self.targets: list[int] = []
        self.current_target: Optional[int] = None

        self.logs: list[dict] = []

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

        act, self.current_target = self._select_next_target()

        if act:
            self._collect_logs(observation)
        else:
            logger.debug(
                '%s agent does not have any valid targets it will terminate', self.name
            )

        return (int(act), self.current_target)

    def _select_next_target(self) -> tuple[bool, Optional[int]]:
        if self.current_target in self.targets:
            # If self.current_target is not yet compromised, e.g. due to TTCs,
            # keep using that as the target.
            self.targets.remove(self.current_target)
            self.targets.append(self.current_target)

        act = np.random.choice(
            [True, False],
            p=[1 - self.settings['wait_factor'], self.settings['wait_factor']],
        )

        try:
            return act, self.targets.pop()
        except IndexError:
            return False, None

    def _collect_logs(self, observation):
        for _, detector in self.attack_graph.nodes[
            self.current_target
        ].detectors.items():
            attack_step = self.attack_graph.nodes[self.current_target]
            log = {
                'timestamp': observation['timestamp'],
                '_detector': detector.name,
                'asset': str(attack_step.asset.name),
                'attack_step': attack_step.name,
                'agent': self.__class__.__name__,
                #'context': {},
            }

            for label, lgasset in detector.context.items():
                *_, asset = (
                    step.asset
                    for step in self.attack_graph.attackers[0].reached_attack_steps
                    if step.asset.type
                    in [subasset.name for subasset in lgasset.sub_assets]
                )

                log[label] = str(asset.name)

            self.logs.append(log)

            logger.info('Detector triggered on %s', attack_step.full_name)
            logger.info(pprint.pformat(log))

    def terminate(self):
        self._write_logs()

    def _write_logs(self):
        with open('logs.json', 'w') as f:
            json.dump(self.logs, f, indent=2)
            self.logs = []


class DepthFirstAttacker(BreadthFirstAttacker):
    _insert_head = None
