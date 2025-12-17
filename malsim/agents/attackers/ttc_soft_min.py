"""
Use softargmax to prioritize low TTC attack steps when running attacker
"""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
import random
import numpy as np

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from mal_simulator import MalSimAttackerState


class TTCSoftMinAttacker:
    """An agent that selects random actions"""

    def __init__(self, agent_config: dict[str, Any]):
        seed = agent_config.get('seed')
        self.rng = random.Random(seed)
        self.beta = agent_config.get('beta', 1.0)

    def get_next_action(
        self, agent_state: MalSimAttackerState, **kwargs: Any
    ) -> Optional[AttackGraphNode]:
        """Sample node from the action surface based on ttc softargmax"""

        possible_choices: list[AttackGraphNode] = list(agent_state.action_surface)
        if not possible_choices:
            return None

        ttcs_left = [
            agent_state.sim.node_ttc_value(n) - agent_state.num_attempts[n]
            for n in possible_choices
        ]

        beta = self.beta  # adjust sharpness of softargmax
        weights = np.exp(-beta * np.array(ttcs_left, dtype=np.float64))
        weights /= weights.sum()
        idx = self.rng.choices(range(len(possible_choices)), weights, k=1)[0]

        return possible_choices[idx]
