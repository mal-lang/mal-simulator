from __future__ import annotations
import logging
import random
import re

from collections import deque
from collections.abc import Iterable
from typing import Optional, TYPE_CHECKING

from .decision_agent import DecisionAgent
from ..mal_simulator import MalSimAgentStateView

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
        # The random seed to initialize the randomness engine with. If set, the
        # simulation will be deterministic.
    }

    def __init__(self, agent_config: dict) -> None:
        """Initialize a BFS agent.

        Args:
            agent_config: Dict with settings to override defaults
        """
        self.targets: deque[AttackGraphNode] = deque()
        self.current_target: Optional[AttackGraphNode] = None

        self.settings = self.default_settings | agent_config

        self.rng = random.Random(self.settings.get('seed'))

    def get_next_action(
        self, agent_state: MalSimAgentStateView, **kwargs
    ) -> Optional[AttackGraphNode]:
        self._update_targets(agent_state.action_surface)
        self._select_next_target()

        return self.current_target

    def _update_targets(self, action_surface: Iterable[AttackGraphNode]):
        if self.settings['seed']:
            # If a seed is set, we assume the user wants determinism in the
            # simulation. Thus, we sort to an ordered list to make sure the
            # non-deterministic ordering of the action_surface set does not
            # break simulation determinism.
            action_surface = sorted(list(action_surface), key=lambda n: n.id)

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


class DepthFirstAttacker(BreadthFirstAttacker):
    _extend_method = 'extend'
