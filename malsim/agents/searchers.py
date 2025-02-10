from __future__ import annotations

import json
import logging
import pprint
import re

from collections import deque
from typing import Optional, TYPE_CHECKING

import numpy as np

from .decision_agent import DecisionAgent
from ..sims import MalSimAgentStateView

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode

logger = logging.getLogger(__name__)


class BreadthFirstAttacker(DecisionAgent):
    """A Breadth-First agent, with possible randomization at each level."""

    _extend_method = 'extendleft'
    # Controls where newly discovered steps will be appended to the list of
    # available actions. Currently used to differentiate between BFS and DFS
    # agents.

    name = ' '.join(re.findall(r'[A-Z][^A-Z]*', __qualname__))
    # A human-friendly name for the agent.

    default_settings = {
        'randomize': False,
        # Whether to randomize next target selection, still respecting the
        # policy of the agent (e.g. BFS or DFS).
        'seed': None,
        # The random seed to initialize the randomness engine with.
        'wait_factor': 0,
    }

    def __init__(self, agent_config: dict) -> None:
        """Initialize a BFS agent.

        Args:
            agent_config: Dict with settings to override defaults
        """
        self.targets: deque[AttackGraphNode] = deque()
        self.current_target: Optional[AttackGraphNode] = None
        self.logs: list[dict] = []
        self.attack_graph = agent_config.pop('attack_graph')
        self.settings = self.default_settings | agent_config
        self.rng = np.random.default_rng(
            self.settings['seed'] or np.random.SeedSequence()
        )

    def get_next_action(
        self, agent: MalSimAgentStateView, **kwargs
    ) -> Optional[AttackGraphNode]:
        self._update_targets(agent.action_surface)
        self._select_next_target()

        if self.current_target:
            self._collect_logs(agent)

        # TODO fix this
        act = np.random.choice(
            [True, False],
            p=[1 - self.settings['wait_factor'], self.settings['wait_factor']],
        )

        return self.current_target

    def _update_targets(self, action_surface: list[AttackGraphNode]):
        # action surface does not have a guaranteed order,
        # so for the agent to be deterministic we need to sort
        action_surface.sort(key=lambda n: n.id)

        new_targets = [
            step
            for step in action_surface
            if step not in self.targets and not step.is_compromised()
        ]

        if self.settings['randomize']:
            self.rng.shuffle(new_targets)

        if self.current_target in new_targets:
            # If self.current_target is not yet compromised, e.g. due to TTCs,
            # keep using that as the target.
            new_targets.remove(self.current_target)
            new_targets.append(self.current_target)

        # Enabled defenses may remove previously possible attack steps.
        self.targets = deque(filter(lambda n: n.is_viable, self.targets))

        getattr(self.targets, self._extend_method)(new_targets)

    def _select_next_target(self) -> None:
        """
        Implement the actual next target selection logic.
        """
        try:
            self.current_target = self.targets.pop()
        except IndexError:
            self.current_target = None

    def _collect_logs(self, state):
        attack_step = self.attack_graph.nodes[self.current_target.id]
        for _, detector in attack_step.detectors.items():
            log = {
                'timestamp': state.timestamp,
                '_detector': detector.name,
                'asset': str(attack_step.asset.name),
                'attack_step': attack_step.name,
                'agent': self.__class__.__name__,
                #'context': {},
            }

            for label, lgasset in detector.context.items():
                ret = (
                    step.asset
                    for step in self.attack_graph.attackers[0].reached_attack_steps
                    if step.asset.type
                    in [subasset.name for subasset in lgasset.sub_assets]
                )
                try:
                    *_, asset = (
                        step.asset
                        for step in self.attack_graph.attackers[0].reached_attack_steps
                        if step.asset.type
                        in [subasset.name for subasset in lgasset.sub_assets]
                    )
                except ValueError:
                    msg = (
                        f'Context {detector.context} cannot be satisfied '
                        f'for step {attack_step.full_name}.'
                    )
                    raise ValueError(msg)

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
    _extend_method = 'extend'
