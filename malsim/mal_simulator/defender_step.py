from __future__ import annotations
from collections.abc import Set
from typing import TYPE_CHECKING
from maltoolbox.attackgraph import AttackGraphNode
import logging


from malsim.mal_simulator.agent_states import attacker_states
from malsim.mal_simulator.attacker_step import attacker_is_terminated
from malsim.mal_simulator.simulator_state import MalSimulatorState

if TYPE_CHECKING:
    from malsim.mal_simulator.agent_states import AgentStates
    from malsim.mal_simulator.defender_state import DefenderState

logger = logging.getLogger(__name__)


def defender_is_terminated(agent_states: AgentStates) -> bool:
    """Check if defender is terminated
    Can be overridden by subclass for custom termination condition.
    """
    # Defender is terminated if all attackers are terminated
    return all(
        attacker_is_terminated(a) for a in attacker_states(agent_states).values()
    )


def defender_step(
    sim_state: MalSimulatorState,
    agent: DefenderState,
    nodes: list[AttackGraphNode],
) -> list[AttackGraphNode]:
    """Enable defense step nodes with defender.

    Args:
    agent - the agent to activate defense nodes with
    nodes - the defense step nodes to enable

    Returns a tuple of a list and a set, `enabled_defenses`
    """

    logger.debug('Stepping with %s', agent.name)
    enabled_defenses: list[AttackGraphNode] = []

    for node in nodes:
        assert node == sim_state.attack_graph.nodes[node.id], (
            f'{agent.name} tried to enable a node that is not part '
            'of this simulators attack_graph. Make sure the node '
            'comes from the agents action surface.'
        )

        if node not in agent.action_surface:
            logger.warning(
                'Defender agent "%s" tried to step through "%s"(%d), '
                'which is not part of its defense surface. Defender '
                'step will skip!',
                agent.name,
                node.full_name,
                node.id,
            )
        else:
            enabled_defenses.append(node)
            logger.info(
                'Defender agent "%s" enabled "%s"',
                agent.name,
                node.full_name,
            )

    return enabled_defenses
