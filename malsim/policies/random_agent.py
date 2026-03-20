from __future__ import annotations
from typing import Any, TYPE_CHECKING
import random

from .decision_agent import DecisionAgent

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ..mal_simulator import AgentState


class RandomAgent(DecisionAgent):
    """An agent that selects random actions"""

    def __init__(self, agent_config: dict[str, Any], **_: Any):
        seed = agent_config.get('seed')
        self.rng = random.Random(seed)

        self.wait_prob = agent_config.get('wait_prob', 0.0)

    def get_next_action(
        self, agent_state: AgentState, **kwargs: Any
    ) -> AttackGraphNode | None:
        """Return a random node from the action surface"""
        possible_choices = list(agent_state.action_surface)
        possible_choices.sort(key=lambda n: n.id)
        if not possible_choices or self.rng.random() < self.wait_prob:
            return None
        return self.rng.choice(possible_choices)
