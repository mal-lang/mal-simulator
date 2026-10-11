from __future__ import annotations
from typing import TYPE_CHECKING

from malsim.mal_simulator.agent_states import attacker_states
from malsim.mal_simulator.attacker_step import attacker_is_terminated

if TYPE_CHECKING:
    from malsim.mal_simulator.agent_states import AgentStates


def defender_is_terminated(agent_states: AgentStates) -> bool:
    """Check if defender is terminated
    Can be overridden by subclass for custom termination condition.
    """
    # Defender is terminated if all attackers are terminated
    return all(
        attacker_is_terminated(a) for a in attacker_states(agent_states).values()
    )
