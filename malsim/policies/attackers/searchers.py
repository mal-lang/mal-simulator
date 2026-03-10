from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
import logging
import random

from collections import deque
from typing import ClassVar, Optional, TYPE_CHECKING, Any

from ..decision_agent import DecisionAgent

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ...mal_simulator import MalSimAgentState

logger = logging.getLogger(__name__)


class ActionOrdering(Enum):
    RANDOM = 'random'  # randomize the order of new targets at each step
    SORTED = 'sorted'  # sort by node id
    NOTHING = 'nothing'  # rely on the ordering of the action surface set, which is theoeritically non-deterministic and may lead to non-deterministic simulations


@dataclass
class AgentConfig:
    # Whether to sort next target selection, still respecting the
    # policy of the agent (BFS or DFS).
    action_ordering: ActionOrdering = ActionOrdering.NOTHING
    # The random seed to initialize the randomness engine with.
    # If None, no seed is set and thus results may be non-deterministic.
    seed: Optional[int] = None


def parse_agent_config(config_dict: dict[str, Any]) -> AgentConfig:
    """Parse a dict into an AgentConfig, validating the fields."""
    try:
        action_ordering = ActionOrdering(config_dict.get('action_ordering', 'nothing'))
    except ValueError as e:
        raise ValueError(
            f'Invalid action_ordering: {config_dict.get("action_ordering")}. '
            f'Valid options are: {[ao.value for ao in ActionOrdering]}'
        ) from e
    seed = config_dict.get('seed')
    if seed is not None and not isinstance(seed, int):
        raise ValueError(f'Seed must be an integer or None, got {type(seed)}')
    return AgentConfig(action_ordering=action_ordering, seed=seed)


def shuffle(rng: random.Random, nodes: list[AttackGraphNode]) -> list[AttackGraphNode]:
    """Shuffle a list of nodes using the provided random generator."""
    nodes_copy = nodes.copy()
    rng.shuffle(nodes_copy)
    return nodes_copy


class BreadthFirstAttacker(DecisionAgent):
    """A Breadth-First agent, with possible randomization at each level."""

    # A human-friendly name for the agent.
    name = 'Breadth First Attacker'

    _default_settings: ClassVar[AgentConfig] = AgentConfig(
        action_ordering=ActionOrdering.NOTHING,
        seed=None,
    )
    _extend_method = 'extendleft'

    def __init__(self, agent_config: AgentConfig | dict[str, Any]) -> None:
        """Initialize a BFS/DFS agent.

        Args:
            agent_config: Dict with settings to override defaults
        """
        self._targets: deque[AttackGraphNode] = deque()
        self._current_target: Optional[AttackGraphNode] = None

        config = (
            parse_agent_config(agent_config)
            if isinstance(agent_config, dict)
            else agent_config
        )

        settings = replace(self._default_settings, **config.__dict__)

        self._rng = random.Random(config.seed)
        self._prev_state = None

        # _rng = random.Random(config.seed)
        # order_funcs: dict[
        #     ActionOrdering,
        #     Callable[[frozenset[AttackGraphNode]], list[AttackGraphNode]],
        # ] = {
        #     ActionOrdering.RANDOM: lambda nodes: list(shuffle(_rng, list(nodes))),
        #     ActionOrdering.SORTED: lambda nodes: sorted(nodes, key=lambda n: n.id),
        #     ActionOrdering.NOTHING: lambda nodes: list(nodes),
        # }
        # self.order_func = order_funcs[config.action_ordering]
        self._settings = settings

    # def extend_method(
    #     self, targets: deque[AttackGraphNode], new_nodes: list[AttackGraphNode]
    # ) -> deque[AttackGraphNode]:
    #     """Extend the deque of targets with new nodes according to the policy of the agent (BFS or DFS)"""
    #     targets.extendleft(new_nodes)
    #     return targets

    def get_next_action(
        self, agent_state: MalSimAgentState, **kwargs: Any
    ) -> Optional[AttackGraphNode]:
        """Receive the next action according to agent policy (bfs/dfs)"""

        new_nodes = (
            agent_state.action_surface - self._prev_state.action_surface
            if self._prev_state
            else agent_state.action_surface
        )

        disabled_nodes = (
            self._prev_state.action_surface - agent_state.action_surface
            if self._prev_state
            else frozenset()
        )

        self._targets = self._update_targets(
            new_nodes=new_nodes,
            old_target_queue=self._targets,
            extend_method=self._extend_method,
            disabled_nodes=disabled_nodes,
            current_target=self._current_target,
        )

        self._current_target = _select_next_target(self._targets)
        self._prev_state = agent_state
        return self._current_target

    def _update_targets(
        self,
        new_nodes: list[AttackGraphNode],
        old_target_queue: deque[AttackGraphNode],
        extend_method: str,
        disabled_nodes: frozenset[AttackGraphNode],
        current_target: Optional[AttackGraphNode] = None,
    ):
        new_targets: list[AttackGraphNode] = []
        if self._settings.action_ordering == ActionOrdering.SORTED:
            # If a seed is set, we assume the user wants determinism in the
            # simulation. Thus, we sort to an ordered list to make sure the
            # non-deterministic ordering of the action_surface set does not
            # break simulation determinism.
            new_targets = sorted(new_nodes, key=lambda n: n.id)
        else:
            # sorted above returns a list already
            new_targets = list(new_nodes)

        if self._settings.action_ordering == ActionOrdering.RANDOM:
            new_targets = sorted(new_nodes, key=lambda n: n.id)
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

def _select_next_target(
    targets: deque[AttackGraphNode] | None,
) -> AttackGraphNode | None:
    """
    Implement the actual next target selection logic.
    """
    if targets:
        return targets.pop()
    else:
        return None


class DepthFirstAttacker(BreadthFirstAttacker):
    name = 'Depth First Attacker'
    _extend_method = 'extend'  # DFS extends to the right, BFS extends to the left

    # def extend_method(
    #     self, targets: deque[AttackGraphNode], new_nodes: list[AttackGraphNode]
    # ) -> deque[AttackGraphNode]:
    #     targets.extend(new_nodes)
    #     return targets
