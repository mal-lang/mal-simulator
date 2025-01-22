from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import logging

import numpy as np

from .decision_agent import DecisionAgent

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ..sims.mal_sim_agent_state import MalSimAgentStateView

logger = logging.getLogger(__name__)

class ShutdownCompromisedMachinesDefender(DecisionAgent):
    """A defender that defends compromised assets using notPresent"""

    def __init__(self, agent_config, **_):
        # Seed and rng not currently used
        seed = (
            agent_config["seed"]
            if agent_config.get("seed")
            else np.random.SeedSequence().entropy
        )
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get("randomize")
            else None
        )

    def get_next_action(
        self, agent: MalSimAgentStateView, **kwargs
    ) -> Optional[AttackGraphNode]:

        """Return an action that disables a compromised asset"""

        selected_node = None
        for node in agent.action_surface:

            # Child of a defense node is compromised -> enable the defense
            # TODO: optionally randomize order so not always same.
            for child_node in node.children:
                if child_node.is_compromised():
                    return node

        return selected_node
