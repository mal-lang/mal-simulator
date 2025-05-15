from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np

from .decision_agent import DecisionAgent

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ..mal_simulator import MalSimAgentStateView

class RandomAgent(DecisionAgent):
    """An agent that selects random actions"""

    def __init__(self, agent_config, **_):
        seed = (
            agent_config["seed"]
            if agent_config.get("seed")
            else np.random.SeedSequence().entropy
        )
        self.rng = np.random.default_rng(seed)

    def get_next_action(
        self, agent_state: MalSimAgentStateView, **kwargs
    ) -> Optional[AttackGraphNode]:
        """Return a random node from the action surface"""
        return self.rng.choice(
            np.array(list(agent_state.action_surface)).tolist()
        )
