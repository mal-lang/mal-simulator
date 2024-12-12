import logging
from collections import deque
from typing import Any, Deque, Dict, List, Union, Optional, Tuple

import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from .agent_base import AgentType, MalSimAgent, MalSimAttacker, MalSimDefender

logger = logging.getLogger(__name__)

# Helper functions for searchers
def initialize_rng(
        agent_config: Dict[str, Any]
    ) -> Optional[np.random.Generator]:
    """Create random number generator for searcher agents if requested"""

    if agent_config is None:
        return None

    seed = agent_config.get("seed", np.random.SeedSequence().entropy)
    return np.random.default_rng(seed)\
           if agent_config.get("randomize", False) else None


def update_targets(
    new_targets: List[AttackGraphNode],
    targets: Union[Deque[AttackGraphNode], List[AttackGraphNode]],
    rng: Optional[np.random.Generator],
    append_to_front: bool = True,
) -> None:
    """Add new targets to the target list/queue with optional shuffling."""

    if rng:
        rng.shuffle(new_targets)

    if append_to_front:
        for node in new_targets:
            targets.appendleft(node)
    else:
        targets.extend(new_targets)


def process_current_target(
    current_target: Optional[AttackGraphNode],
    targets: Union[Deque[AttackGraphNode], List[AttackGraphNode]],
    attack_surface: List[AttackGraphNode],
    is_valid: callable,
) -> Tuple[Optional[AttackGraphNode], bool]:
    """Return next valid target that is in the attack surface
    and a value that tells whether agent is done or not"""

    while current_target not in attack_surface or not is_valid(current_target):
        if not targets:
            return None, True
        current_target = targets.pop()
    return current_target, False


# Base Class for searcher logic
class BaseSearcherAgent(MalSimAgent):

    def __init__(
            self,
            name: str,
            agent_type: AgentType,
            queue_type=None,
            agent_config=None,
            **kwargs
        ):
        super().__init__(name, agent_type, **kwargs)
        self.current_target: Optional[AttackGraphNode] = None
        if queue_type == "deque":
            self.targets: Deque[AttackGraphNode] = deque([])
        elif queue_type == "list":
            self.targets: List[AttackGraphNode] = []
        else:
            raise ValueError("Invalid queue_type. Use 'deque' or 'list'.")
        self.rng = initialize_rng(agent_config)

    def get_next_action(
        self, **kwargs
    ) -> List[AttackGraphNode]:
        """Compute the next action for searcher agent"""

        new_targets = [node for node in self.action_surface
                       if node not in self.targets]

        update_targets(
            new_targets,
            self.targets,
            self.rng,
            kwargs['append_to_front']
        )

        self.current_target, done = \
            process_current_target(
                self.current_target,
                self.targets,
                self.action_surface,
                kwargs['is_valid']
        )

        self.current_target = None if done else self.current_target
        return [self.current_target] if self.current_target else []


# Breadth-First Attacker
class BreadthFirstAttacker(MalSimAttacker, BaseSearcherAgent):
    def __init__(
            self,
            name: str,
            attacker_id: int,
            agent_config=None
        ):
        super().__init__(
            name,
            attacker_id=attacker_id,
            queue_type="deque",
            agent_config=agent_config
        )

    def get_next_action(
            self
        ) -> List[AttackGraphNode]:
        return super().get_next_action(
            is_valid=lambda x: not x.is_compromised(),
            append_to_front=True
        )


# Depth-First Attacker
class DepthFirstAttacker(MalSimAttacker, BaseSearcherAgent):

    def __init__(
            self,
            name: str,
            attacker_id: int,
            agent_config=None,
            **kwargs
        ):
        super().__init__(
            name,
            attacker_id,
            queue_type="list",
            agent_config=agent_config,
            **kwargs
        )

    def get_next_action(
            self
        ) -> List[AttackGraphNode]:
        return super().get_next_action(
            is_valid=lambda x: not x.is_compromised(),
            append_to_front=False
        )


# Breadth-First Defender
class BreadthFirstDefender(MalSimDefender, BaseSearcherAgent):
    def __init__(
            self,
            name: str,
            agent_config=None,
            **kwargs
        ):
        super().__init__(
            name,
            queue_type="deque",
            agent_config=agent_config,
            **kwargs
        )

    def get_next_action(
            self
        ) -> List[AttackGraphNode]:
        return super().get_next_action(
            is_valid=lambda x: x.is_available_defense(),
            append_to_front=True
        )


# Depth-First Defender
class DepthFirstDefender(MalSimDefender, BaseSearcherAgent):
    def __init__(
            self,
            name: str,
            agent_config=None,
            **kwargs
        ):
        super().__init__(
            name,
            queue_type="list",
            agent_config=agent_config,
            **kwargs
        )

    def get_next_action(
            self
        ) -> List[AttackGraphNode]:
        return super().get_next_action(
            is_valid=lambda x: x.is_available_defense(),
            append_to_front=False
        )
