from __future__ import annotations

from malsim.mal_simulator.sim_data import SimData
from malsim.mal_simulator.graph_state import GraphState
from malsim.mal_simulator.node import (
    node_is_actionable,
    node_is_necessary,
    node_is_traversable,
    node_is_viable,
)
from malsim.mal_simulator.settings import MalSimulatorSettings


from typing import Optional
from collections.abc import Set


from maltoolbox.attackgraph import AttackGraphNode

from malsim.scenario import AttackerSettings


def get_attack_surface(
    graph_state: GraphState,
    sim_data: SimData,
    sim_settings: MalSimulatorSettings,
    agent_settings: AttackerSettings,
    performed_nodes: Set[AttackGraphNode],
    from_nodes: Optional[Set[AttackGraphNode]] = None,
) -> frozenset[AttackGraphNode]:
    """
    Calculate the attack surface of the attacker.
    If from_nodes are provided only calculate the attack surface
    stemming from those nodes, otherwise use all performed_nodes.
    The attack surface includes all of the traversable children nodes.

    Arguments:
    agent_name      - the agent to get attack surface for
    performed_nodes - the nodes the agent has performed
    from_nodes      - the nodes to calculate the attack surface from

    """

    from_nodes = from_nodes if from_nodes is not None else performed_nodes
    attack_surface: set[AttackGraphNode] = set()

    skip_compromised = sim_settings.attack_surface_skip_compromised
    skip_unviable = sim_settings.attack_surface_skip_unviable
    skip_unnecessary = sim_settings.attack_surface_skip_unnecessary

    for parent in from_nodes:
        for child in parent.children:
            if skip_compromised and child in performed_nodes:
                continue

            if skip_unviable and not node_is_viable(graph_state, child):
                continue

            if skip_unnecessary and not node_is_necessary(graph_state, child):
                continue

            if not node_is_actionable(sim_data, child, agent_settings):
                continue

            if node_is_traversable(graph_state, performed_nodes, child):
                attack_surface.add(child)

    return frozenset(attack_surface)
