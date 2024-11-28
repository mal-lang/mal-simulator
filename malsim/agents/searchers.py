import logging
from collections import deque
from typing import Any, Deque, Dict, List, Union, Optional

import numpy as np

from maltoolbox.attackgraph import AttackGraphNode
from .agent_base import MalSimAgent

logger = logging.getLogger(__name__)


def get_new_targets(
    action_surface: List[AttackGraphNode], discovered_targets: List[AttackGraphNode]
) -> List[AttackGraphNode]:
    surface_nodes = [node for node in action_surface]
    new_targets = [node for node in surface_nodes if node not in discovered_targets]
    return new_targets, surface_nodes


class PassiveAgent(MalSimAgent):
    def compute_next_action(self, _) -> list:
        return []


class BreadthFirstAttacker(MalSimAgent):
    def __init__(self, agent_config: Dict[str, Any]= {}) -> None:
        self.targets: Deque[AttackGraphNode] = deque([])
        self.current_target: Union[AttackGraphNode, None] = None
        seed = agent_config.get("seed", np.random.SeedSequence().entropy)
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get("randomize", False)
            else None
        )

    def compute_next_action(
        self, action_surface: List[AttackGraphNode]
    ) -> list[AttackGraphNode]:
        new_targets, surface_nodes = get_new_targets(action_surface, list(self.targets))

        # Add new targets to the back of the queue
        if self.rng:
            self.rng.shuffle(new_targets)
        for node in new_targets:
            self.targets.appendleft(node)

        self.current_target, done = self.select_next_target(
            self.current_target, self.targets, surface_nodes
        )

        self.current_target = None if done else self.current_target
        return [self.current_target] if self.current_target else []

    @staticmethod
    def select_next_target(
        current_target: Optional[AttackGraphNode],
        targets: Union[List[AttackGraphNode], Deque[AttackGraphNode]],
        attack_surface: List[AttackGraphNode],
    ) -> tuple[Optional[AttackGraphNode], bool]:

        # If the current target was not compromised,
        # put it back at the bottom of the stack

        if current_target in attack_surface:
            targets.appendleft(current_target)
            current_target = targets.pop()

        while current_target not in attack_surface or current_target.is_compromised():
            if len(targets) == 0:
                return None, True
            current_target = targets.pop()
        if not current_target.is_viable:
            breakpoint()
        return current_target, False


class DepthFirstAttacker(MalSimAgent):
    """An agent that selects steps by using DFS"""
    def __init__(self, agent_config: Dict[str, Any] = {}) -> None:
        self.current_target: Optional[AttackGraphNode] = None
        self.targets: List[AttackGraphNode] = []
        seed = agent_config.get("seed", np.random.SeedSequence().entropy)
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get("randomize", False)
            else None
        )

    def compute_next_action(
        self, action_surface: List[AttackGraphNode]
    ) -> list[AttackGraphNode]:

        new_targets, surface_nodes = \
            get_new_targets(action_surface, self.targets)

        # Add new targets to the top of the stack
        if self.rng:
            self.rng.shuffle(new_targets)
        for node in new_targets:
            self.targets.append(node)

        self.current_target, done = self.select_next_target(
            self.current_target, self.targets, surface_nodes
        )

        self.current_target = None if done else self.current_target
        return [self.current_target] if self.current_target else []

    @staticmethod
    def select_next_target(
        current_target: Optional[AttackGraphNode],
        targets: Union[List[AttackGraphNode], Deque[AttackGraphNode]],
        attack_surface: List[AttackGraphNode],
    ) -> tuple[Optional[AttackGraphNode], bool]:

        if current_target in attack_surface:
            return current_target, False

        while current_target not in attack_surface or current_target.is_compromised():
            if len(targets) == 0:
                return None, True
            current_target = targets.pop()

        if not current_target.is_viable:
            breakpoint()

        return current_target, False
