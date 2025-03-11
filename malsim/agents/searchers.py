from __future__ import annotations
import logging
import random

from collections import deque
from typing import Optional, TYPE_CHECKING, Any

from .decision_agent import DecisionAgent
from ..mal_simulator import MalSimAgentStateView

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode

logger = logging.getLogger(__name__)


class BreadthFirstAttacker(DecisionAgent):
    """A Breadth-First agent, with possible randomization at each level."""

    # A human-friendly name for the agent.
    name = 'Breadth First Attacker'

    # Controls where newly discovered steps will be appended to the deque of
    # available actions. Differentiates between BFS and DFS agents.
    _extend_method = 'extendleft'

    _default_settings: dict[str, Any] = {
        # Whether to randomize next target selection, still respecting the
        # policy of the agent (BFS or DFS).
        'randomize': False,
        # The random seed to initialize the randomness engine with.
        # If set, the simulation will be deterministic.
        'seed': None,
    }

    def __init__(self, agent_config: dict) -> None:
        """Initialize a BFS/DFS agent.

        Args:
            agent_config: Dict with settings to override defaults
        """
        self._targets: deque[AttackGraphNode] = deque()
        self._current_target: Optional[AttackGraphNode] = None
        self._settings = self._default_settings | agent_config

        self._rng = random.Random(self._settings.get('seed'))
        self._started = False

    def get_next_action(
        self, agent_state: MalSimAgentStateView, **kwargs
    ) -> Optional[AttackGraphNode]:
        """Receive the next action according to agent policy (bfs/dfs)"""

        self._update_targets(
            new_nodes=(
                agent_state.step_action_surface_additions
                if self._started else agent_state.action_surface
            ),
            disabled_nodes=(
                agent_state.step_action_surface_removals
            )
        )

        self._select_next_target()
        self._started = True
        return self._current_target

    def _update_targets(
        self,
        new_nodes: set[AttackGraphNode],
        disabled_nodes: set[AttackGraphNode],
    ):
        new_targets: list[AttackGraphNode] = []
        if self._settings['seed']:
            # If a seed is set, we assume the user wants determinism in the
            # simulation. Thus, we sort to an ordered list to make sure the
            # non-deterministic ordering of the action_surface set does not
            # break simulation determinism.
            new_targets = sorted(new_nodes, key=lambda n: n.id)
        else:
            # sorted above returns a list already
            new_targets = list(new_nodes)

        if self._settings['randomize']:
            self._rng.shuffle(new_targets)

        if (
            self._current_target
            and not self._current_target.is_compromised()
            and self._current_target not in disabled_nodes
        ):
            # If self.current_target was not compromised, e.g. due to TTCs,
            # it remains in action surface and should be added as a target.
            self._targets.append(self._current_target)

        # Enabled defenses may remove previously possible attack steps.
        if disabled_nodes:
            self._targets = deque(
                n for n in self._targets if n not in disabled_nodes
            )

        # Use the extend method to add new targets
        getattr(self._targets, self._extend_method)(new_targets)

    def _select_next_target(self) -> None:
        """
        Implement the actual next target selection logic.
        """
        if self._targets:
            self._current_target = self._targets.pop()
        else:
            self._current_target = None


class DepthFirstAttacker(BreadthFirstAttacker):
    name = 'Depth First Attacker'
    _extend_method = 'extend'
