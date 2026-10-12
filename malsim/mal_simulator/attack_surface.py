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
    actionability: NodePropertyRule[bool] | None,
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

    skip_compromised = settings.skip_compromised
    skip_unnecessary = settings.skip_unnecessary

    def cheaply_eligible(node: AttackGraphNode) -> bool:
        # Nodes marked as effects are not actions/attacks.
        # These checks are cheap (attribute/dict/set lookups), so they run
        # in Python to shrink the candidate set before the native
        # traversability check, which is the expensive part.
        is_action = node.causal_mode != 'effect'
        return (
            is_action
            and (node not in performed_nodes if skip_compromised else True)
            and (node_is_necessary(sim_state, node) if skip_unnecessary else True)
            and node_is_actionable(actionability, node)
        )

    from_node_children = {node for parent in from_nodes for node in parent.children}
    candidates = [node for node in from_node_children if cheaply_eligible(node)]
    if not candidates:
        return frozenset()

    traversable_ids = set(
        sim_state.graph_state.attack_graph_index.filter_traversable(
            [node.id for node in candidates],
            [node.id for node in performed_nodes],
            [node.id for node in sim_state.enabled_defenses],
        )
    )
    return frozenset(node for node in candidates if node.id in traversable_ids)
