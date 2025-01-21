from __future__ import annotations
import logging

from collections import deque
from typing import Deque, List, Set, Union, Optional, TYPE_CHECKING
import numpy as np

from .decision_agent import DecisionAgent
from ..sims import MalSimAgentStateView

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode

logger = logging.getLogger(__name__)

def get_new_targets(
    discovered_targets: set[int],
    possible_actions: set[int]
) -> list[int]:
    """Return targets that are not already discovered"""
    new_targets = [id for id in possible_actions
                   if id not in discovered_targets]
    return new_targets

class BreadthFirstAttacker(DecisionAgent):
    def __init__(self, agent_config: dict) -> None:
        self.targets: Deque[int] = deque([])
        self.current_target: int = None

        seed = (
            agent_config["seed"]
            if agent_config.get("seed", None)
            else np.random.SeedSequence().entropy
        )
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get("randomize", False)
            else None
        )

    def get_next_action(
        self, agent: MalSimAgentStateView, **kwargs
    ) -> Optional[AttackGraphNode]:

        # Create a dict of possible actions
        # mapping id to node
        possible_actions = {
            n.id: n for n in agent.action_surface
            if not n.is_compromised()
        }

        # Get targets that are not discovered yet
        new_targets = get_new_targets(
            self.targets, possible_actions.keys()
        )

        # Add new targets to the back of the queue
        # if desired, shuffle the new targets to
        # make the attacker more unpredictable
        if self.rng:
            self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.appendleft(c)

        # Select next target
        self.current_target = self.select_next_target(
            self.current_target,
            self.targets,
            possible_actions.keys()
        )

        # Convert the current target id to AttackGraphNode
        action_node = None
        if self.current_target is not None:
            action_node = possible_actions[self.current_target]

        return action_node

    @staticmethod
    def select_next_target(
        previous_target: int,
        targets: Union[List[int], Deque[int]],
        attack_surface: Set[int],
    ) -> Optional[int]:
        """Select a target from attack surface
        by going through the target queue"""

        next_target = None
        if previous_target in attack_surface:
            # If the current target was not compromised, put it
            # back, but on the bottom of the stack.
            targets.appendleft(previous_target)
            next_target = targets.pop()

        while next_target not in attack_surface:
            if len(targets) == 0:
                return None

            next_target = targets.pop()

        return next_target


class DepthFirstAttacker(DecisionAgent):
    def __init__(self, agent_config: dict) -> None:
        self.current_target = -1
        self.targets: List[int] = []
        seed = (
            agent_config["seed"]
            if agent_config.get("seed", None)
            else np.random.SeedSequence().entropy
        )
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get("randomize", False)
            else None
        )

    def get_next_action(
        self, agent: MalSimAgentStateView, **kwargs
    ) -> Optional[AttackGraphNode]:

        # Create a dict of possible actions
        # mapping id to node
        possible_actions = {
            n.id: n for n in agent.action_surface
            if not n.is_compromised()
        }

        # Get targets that are not discovered yet
        new_targets = get_new_targets(
            self.targets, possible_actions.keys()
        )

        # Add new targets to the top of the stack
        if self.rng:
            self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.append(c)

        self.current_target = self.select_next_target(
            self.current_target, self.targets, possible_actions.keys()
        )

        # Convert the current target id to AttackGraphNode
        action_node = None
        if self.current_target is not None:
            action_node = possible_actions[self.current_target]

        return action_node

    @staticmethod
    def select_next_target(
        previous_target: int,
        targets: Union[List[int], Deque[int]],
        attack_surface: Set[int],
    ) -> Optional[int]:
        if previous_target in attack_surface:
            return previous_target

        next_target = None
        while next_target not in attack_surface:
            if len(targets) == 0:
                return None
            next_target = targets.pop()

        return next_target


AGENTS = {
    BreadthFirstAttacker.__name__: BreadthFirstAttacker,
    DepthFirstAttacker.__name__: DepthFirstAttacker,
}
