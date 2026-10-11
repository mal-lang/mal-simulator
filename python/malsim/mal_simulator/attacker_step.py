from __future__ import annotations
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from malsim.mal_simulator.attacker_state import AttackerState

logger = logging.getLogger(__name__)


def attacker_is_terminated(attacker_state: AttackerState) -> bool:
    """Check if attacker is terminated
    Can be overridden by subclass for custom termination condition.

    Args:
    - attacker_state: the attacker state to check for termination
    """

    if len(attacker_state.action_surface) == 0:
        # Attacker is terminated if it has no more actions to take
        logger.info(
            'Attacker "%s" action surface is empty, terminate', attacker_state.name
        )
        return True
    goals = attacker_state.settings.goals
    if goals:
        # Attacker is terminated if it has goals and all goals are met
        return goals & attacker_state.performed_nodes == goals
    # Otherwise not terminated
    return False
