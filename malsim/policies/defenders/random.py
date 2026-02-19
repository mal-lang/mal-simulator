from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
import logging

import numpy as np
from numpy.typing import ArrayLike


if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ...mal_simulator import MalSimDefenderState

logger = logging.getLogger(__name__)


class RandomDefender:
    """A defender that enables a random defense step, at random time intervals."""

    def __init__(self, agent_config: dict[str, Any], **_: Any):
        # Seed and rng not currently used
        seed = (
            agent_config['seed']
            if agent_config.get('seed')
            else np.random.SeedSequence().entropy
        )
        self.rng = np.random.default_rng(seed)
        self.action_prob = agent_config.get('action_prob', 0.05)

    def get_next_action(
        self, agent_state: MalSimDefenderState, **kwargs: Any
    ) -> Optional[AttackGraphNode]:
        """Return an action that disables a compromised node"""
        actions = list(agent_state.action_surface)
        # To ensure seed replication by setting the seed, we sort the actions by id
        actions = sorted(actions, key=lambda n: n.id)
        action: Optional[AttackGraphNode] = (
            None
            if (self.rng.random() > self.action_prob) or (len(actions) == 0)
            else self.rng.choice(np.array(actions))
        )

        return action
