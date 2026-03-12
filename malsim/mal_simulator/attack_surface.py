from __future__ import annotations
from typing import TYPE_CHECKING
from collections import deque
from collections.abc import MutableSet, Set

from malsim.config.node_property_rule import NodePropertyRule
from malsim.config.sim_settings import AttackSurfaceSettings
from malsim.mal_simulator.graph_utils import (
    node_is_actionable,
    node_is_necessary,
    node_is_traversable,
    node_is_viable,
)

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from malsim.mal_simulator.simulator_state import MalSimulatorState


def get_effects_of_attack_step(
    sim_state: MalSimulatorState,
    attack_step: AttackGraphNode,
    performed_nodes: Set[AttackGraphNode],
) -> Set[AttackGraphNode]:
    """Get nodes performed as a consequence of `attack_step` being compromised"""
    performed = set(performed_nodes) | {attack_step}
    effects: MutableSet[AttackGraphNode] = set()
    potential_effects = deque(
        n for n in attack_step.children if n.causal_mode == 'effect'
    )
    while potential_effects:
        effect = potential_effects.popleft()
        has_visited = performed | set(effects)
        if effect not in has_visited and node_is_traversable(
            sim_state, has_visited, effect
        ):
            effects.add(effect)
            potential_effects += (
                n for n in effect.children if n.causal_mode == 'effect'
            )
    return frozenset(effects)


def get_attack_surface(
    settings: AttackSurfaceSettings,
    sim_state: MalSimulatorState,
    actionability: Optional[NodePropertyRule[bool]],
    performed_nodes: Set[AttackGraphNode],
    from_nodes: Set[AttackGraphNode] | None = None,
) -> Set[AttackGraphNode]:
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
    attack_surface: MutableSet[AttackGraphNode] = set()

    skip_compromised = settings.skip_compromised
    skip_unviable = settings.skip_unviable
    skip_unnecessary = settings.skip_unnecessary

    for parent in from_nodes:
        for child in parent.children:
            if child.causal_mode == 'effect':
                # Nodes marked as effects are not actions/attacks
                continue

            if skip_compromised and child in performed_nodes:
                continue

            if skip_unviable and not node_is_viable(sim_state, child):
                continue

            if skip_unnecessary and not node_is_necessary(sim_state, child):
                continue

            if not node_is_actionable(actionability, child):
                continue

            if node_is_traversable(sim_state, performed_nodes, child):
                attack_surface.add(child)

    return frozenset(attack_surface)
