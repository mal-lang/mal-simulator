from __future__ import annotations
import logging
import random
import re

from collections import deque
from typing import Optional, TYPE_CHECKING, Any

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

    default_settings: dict[str, Any] = {
        'randomize': False,
        # Whether to randomize next target selection, still respecting the
        # policy of the agent (e.g. BFS or DFS).
        'seed': None,
        # The random seed to initialize the randomness engine with.
        # If set, the simulation will be deterministic.
    }

    def __init__(self, agent_config: dict) -> None:
        """Initialize a BFS agent.

        Args:
            agent_config: Dict with settings to override defaults
        """

        # A deque with target nodes in bfs order
        self.targets: deque[AttackGraphNode] = deque()

        # The last target chosen
        self.current_target: Optional[AttackGraphNode] = None

        # Agent random/seed settings
        self.settings = self.default_settings | agent_config

        self.rng = random.Random(self.settings.get('seed'))
        self.started = False

    def get_next_action(
        self, agent_state: MalSimAgentStateView, **kwargs
    ) -> Optional[AttackGraphNode]:

        # Add and remove nodes from self.targets
        self._update_targets(
            new_targets=(
                agent_state.step_action_surface_additions
                if self.started else agent_state.action_surface
            ),
            disabled_nodes=agent_state.step_action_surface_removals,
        )
        self._select_next_target()

        self.started = True
        return self.current_target

    def _update_targets(
        self,
        new_targets: set[AttackGraphNode],
        disabled_nodes: set[AttackGraphNode],
    ):
        sorted_new_targets: list = []
        if self.settings['seed']:
            # If a seed is set, we assume the user wants determinism in the
            # simulation. Thus, we sort to an ordered list to make sure the
            # non-deterministic ordering of the action_surface set does not
            # break simulation determinism.
            sorted_new_targets = sorted(new_targets, key=lambda n: n.id)
        else:
            # sorted above returns a list already
            sorted_new_targets = list(new_targets)

        if self.settings['randomize']:
            self.rng.shuffle(sorted_new_targets)

        if self.current_target and not self.current_target.is_compromised():
            # If self.current_target is not yet compromised, e.g. due to TTCs,
            # keep using that as the target.
            sorted_new_targets.append(self.current_target)

        # Enabled defenses may remove previously possible attack steps.
        for node in disabled_nodes:
            if node in self.targets:
                self.targets.remove(node)

        getattr(self.targets, self._extend_method)(sorted_new_targets)

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
