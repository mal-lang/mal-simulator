from __future__ import annotations
from collections.abc import Set
import logging
import random

from collections import deque
from typing import ClassVar, Optional, TYPE_CHECKING, Any

from ..decision_agent import DecisionAgent

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ...mal_simulator import MalSimAgentState

logger = logging.getLogger(__name__)


class BreadthFirstAttacker(DecisionAgent):
    """A Breadth-First agent, with possible randomization at each level."""

    # A human-friendly name for the agent.
    name = 'Breadth First Attacker'

    # Controls where newly discovered steps will be appended to the deque of
    # available actions. Differentiates between BFS and DFS agents.
    _extend_method = 'extendleft'

    _default_settings: ClassVar[dict[str, Any]] = {
        # Whether to randomize next target selection, still respecting the
        # policy of the agent (BFS or DFS).
        'randomize': False,
        # The random seed to initialize the randomness engine with.
        # If set, the simulation will be deterministic.
        'seed': None,
    }

    def __init__(self, agent_config: dict[str, Any]) -> None:
        """Initialize a BFS/DFS agent.

        Args:
            agent_config: Dict with settings to override defaults
        """
        self._targets: deque[AttackGraphNode] = deque()
        self._current_target: Optional[AttackGraphNode] = None
        self._settings = self._default_settings | agent_config

        self._rng = random.Random(self._settings.get('seed'))
        self.prev_state = None

    def get_next_action(
        self, agent_state: MalSimAgentState, **kwargs: Any
    ) -> Optional[AttackGraphNode]:
        """Receive the next action according to agent policy (bfs/dfs)"""

        new_nodes = (
            agent_state.action_surface - self.prev_state.action_surface
            if self.prev_state
            else agent_state.action_surface
        )

        disabled_nodes = (
            self.prev_state.action_surface - agent_state.action_surface
            if self.prev_state
            else set()
        )

        self._targets = self._update_targets(
            new_nodes=new_nodes,
            old_target_queue=self._targets,
            disabled_nodes=disabled_nodes,
            current_target=self._current_target,
            extend_method=self._extend_method,
        )

        self._current_target = self._select_next_target()
        self._started = True
        self.prev_state = agent_state
        return self._current_target

    def _update_targets(
        self,
        new_nodes: set[AttackGraphNode],
        old_target_queue: deque[AttackGraphNode],
        disabled_nodes: set[AttackGraphNode],
        extend_method: str,
        current_target: Optional[AttackGraphNode] = None,
    ) -> deque[AttackGraphNode]:
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

        if current_target and current_target not in disabled_nodes:
            # If self.current_target was not compromised, e.g. due to TTCs,
            # it remains in action surface and should be added as a target.
            old_target_queue.append(current_target)

        # Enabled defenses may remove previously possible attack steps.
        new_target_queue = (
            deque(n for n in old_target_queue if n not in disabled_nodes)
            if disabled_nodes
            else old_target_queue
        )

        # Use the extend method to add new targets
        getattr(new_target_queue, extend_method)(new_targets)
        return new_target_queue

    def _select_next_target(self) -> AttackGraphNode | None:
        """
        Implement the actual next target selection logic.
        """
        if self._targets:
            return self._targets.pop()
        else:
            return None


class DepthFirstAttacker(BreadthFirstAttacker):
    name = 'Depth First Attacker'
    _extend_method = 'extend'
