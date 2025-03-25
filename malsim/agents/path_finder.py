from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import logging
from collections import deque
import heapq
import numpy as np

from .decision_agent import DecisionAgent

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ..mal_simulator import MalSimAgentStateView

logger = logging.getLogger(__name__)


class PathFindingAttacker(DecisionAgent):
    """An attacker that finds the highest reward path in the attack graph."""

    def __init__(self, agent_config, **_):
        seed = (
            agent_config["seed"]
            if agent_config.get("seed")
            else np.random.SeedSequence().entropy
        )
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get("randomize")
            else np.random.default_rng()
        )
        self._current_path: deque[AttackGraphNode] = deque()

    def _calculate_good_reward_path(
            self,
            start_nodes: set[AttackGraphNode],
            visited_nodes: set[AttackGraphNode]
        ) -> deque[AttackGraphNode]:
        """
        Find a path to nodes with the highest reward,
        considering attack graph constraints.
        """
        priority_queue = []  # Max-heap for rewards
        visited = set(visited_nodes)

        best_path_reward = 0
        best_path = deque()
        id_to_node = {}

        for node in start_nodes:
            id_to_node[node.id] = node
            reward = node.extras.get("reward", 0)
            heapq.heappush(
                priority_queue, (-reward, node.id, [node.id]))

        while priority_queue:
            neg_reward, current_id, path = heapq.heappop(priority_queue)
            current = id_to_node[current_id]

            if current in visited:
                continue
            visited.add(current)

            if neg_reward < best_path_reward:
                best_path = deque(path)
                best_path_reward = neg_reward

            # Expand neighbors that can be compromised
            for neighbor in current.children:
                id_to_node[neighbor.id] = neighbor

                if neighbor not in visited:
                    new_path = path + [neighbor.id]
                    heapq.heappush(
                        priority_queue,
                        (-neighbor.extras.get("reward", 0), neighbor.id, new_path)
                    )

        return deque(id_to_node[node_id] for node_id in best_path)

    def get_next_action(
            self, agent_state: MalSimAgentStateView, **kwargs
        ) -> Optional[AttackGraphNode]:
        """Return the next action, recalculating the path if necessary."""

        if not self._current_path:
            self._current_path = self._calculate_good_reward_path(
                agent_state.action_surface, agent_state.performed_nodes
            )

        while self._current_path:
            next_node = self._current_path.popleft()
            if next_node in agent_state.action_surface:
                return next_node

        # If no path with rewards is fine, return random action surface node
        return next(iter(agent_state.action_surface))
