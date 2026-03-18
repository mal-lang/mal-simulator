from __future__ import annotations
from typing import TYPE_CHECKING

from maltoolbox.attackgraph import AttackGraphNode

if TYPE_CHECKING:
    from malsim.config.agent_settings import AttackerSettings, DefenderSettings
    from malsim.mal_simulator.attacker_state import AttackerState
    from malsim.mal_simulator.defender_state import DefenderState

AgentSettings = dict[str, 'AttackerSettings | DefenderSettings']
AgentStates = dict[str, 'AttackerState | DefenderState']
AgentRewards = dict[str, float]
Recording = dict[int, dict[str, list[AttackGraphNode]]]
