from __future__ import annotations
from collections.abc import Set
from typing import TYPE_CHECKING

from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.graph_utils import node_is_actionable, node_is_blocked

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from malsim.mal_simulator.simulator_state import MalSimulatorState


def get_defense_surface(
    sim_state: MalSimulatorState,
    agent_actionability_rule: NodePropertyRule[bool] | None,
) -> Set[AttackGraphNode]:
    """Get the defense surface.
    All non-suppressed defense steps that are not already enabled.

    Arguments:
    graph       - the attack graph
    """

    return {
        node
        for node in sim_state.attack_graph.defense_steps
        if node_is_actionable(agent_actionability_rule, node)
        and not node_is_blocked(sim_state, node)
        and node not in sim_state.enabled_defenses
        and 'suppress' not in node.tags
    }
