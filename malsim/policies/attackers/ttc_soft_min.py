"""
Use softargmax to prioritize low TTC attack steps when running attacker
"""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

from malsim.mal_simulator.state_query import node_ttc_value

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from mal_simulator import MalSimAttackerState


class TTCSoftMinAttacker:
    """An agent that selects random actions"""

    def __init__(self, agent_config: dict[str, Any]):
        seed = agent_config.get('seed')
        self.rng = np.random.default_rng(seed)
        self.beta = agent_config.get('beta', 1.0)

    def get_next_action(
        self, agent_state: MalSimAttackerState, **kwargs: Any
    ) -> Optional[AttackGraphNode]:
        """Sample node from the action surface based on ttc softargmax"""

        possible_choices = sorted(
            list(agent_state.action_surface), key=lambda n: n.id
        )

        if not possible_choices:
            return None

        ttcs_left = [
            node_ttc_value(agent_state, n) - agent_state.num_attempts[n]
            for n in possible_choices
        ]

        beta = self.beta  # adjust sharpness of softargmax
        weights = np.exp(-beta * np.array(ttcs_left, dtype=np.float64))
        weights /= weights.sum()
        idx = self.rng.choice(range(len(possible_choices)), p=weights)
        return possible_choices[idx]
