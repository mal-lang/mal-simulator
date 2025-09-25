
from __future__ import annotations
from typing import Any, TYPE_CHECKING, Optional
import random
import logging

from ..utils.path_finding import get_shortest_paths_for_attacker

if TYPE_CHECKING:
    from malsim.mal_simulator import MalSimAttackerState
    from maltoolbox.attackgraph import AttackGraphNode

logger = logging.getLogger(__name__)

class ShortestPathAttacker:
    """
    An agent that finds shortest path in respect to TTC.
    Requires that the agent has goal set in agent state,
    and that TTCMode is PRE_SAMPLE or EXPECTED_VALUE.

    Note: Experimental, not proven correct
    """

    def __init__(self, agent_config: dict[str, Any]):
        print("Note: ShortestPathAttacker is in experimental mode")
        logger.warning("This agent is in experimental mode")
        seed = agent_config.get("seed")
        self.rng = random.Random(seed)

    def get_next_action(
        self, agent_state: MalSimAttackerState, **kwargs: Any
    ) -> Optional[AttackGraphNode]:
        """Sample node from the action surface based on ttc softargmax"""

        shortest_paths_per_goal = get_shortest_paths_for_attacker(agent_state)
        shortest_path_random_goal = random.choice(
            list(shortest_paths_per_goal.values())
        )
        return next(iter(shortest_path_random_goal))
