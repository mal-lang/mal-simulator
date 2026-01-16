from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from collections import deque
from collections.abc import Set

from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.graph_utils import (
    node_is_actionable,
    node_is_necessary,
    node_is_traversable,
    node_is_viable,
)

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from malsim.config.sim_settings import MalSimulatorSettings
    from malsim.mal_simulator.simulator_state import MalSimulatorState


def get_effects_of_attack_step(
    sim_state: MalSimulatorState,
    attack_step: AttackGraphNode,
    performed_nodes: Set[AttackGraphNode],
) -> set[AttackGraphNode]:
    """Get nodes performed as a consequence of `attack_step` being compromised"""
    performed = set(performed_nodes) | {attack_step}
    effects: set[AttackGraphNode] = set()
    potential_effects = deque(
        n for n in attack_step.children if n.causal_mode == 'effect'
    )
    while potential_effects:
        effect = potential_effects.popleft()
        has_visited = performed | set(effects)
        if node_is_traversable(sim_state, has_visited, effect):
            effects.add(effect)
            potential_effects += (
                n for n in effect.children if n.causal_mode == 'effect'
            )
    return effects


def get_attack_surface(
    sim_settings: MalSimulatorSettings,
    sim_state: MalSimulatorState,
    agent_actionability_rule: Optional[NodePropertyRule],
    global_actionability: dict[AttackGraphNode, bool],
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
            if child.causal_mode == 'effect':
                # Nodes marked as effects are not actions/attacks
                continue

            if skip_compromised and child in performed_nodes:
                continue

            if skip_unviable and not node_is_viable(sim_state, child):
                continue

            if skip_unnecessary and not node_is_necessary(sim_state, child):
                continue

            if not node_is_actionable(
                agent_actionability_rule, global_actionability, child
            ):
                continue

            if node_is_traversable(sim_state, performed_nodes, child):
                attack_surface.add(child)

    return frozenset(attack_surface)
