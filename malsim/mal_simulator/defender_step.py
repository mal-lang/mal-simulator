from __future__ import annotations
from typing import TYPE_CHECKING
from maltoolbox.attackgraph import AttackGraphNode
import logging

from malsim.mal_simulator.attacker_state import get_attacker_agents
from malsim.mal_simulator.attacker_step import attacker_is_terminated
from malsim.mal_simulator.graph_processing import make_node_unviable
from malsim.mal_simulator.graph_utils import node_reward
from malsim.mal_simulator.simulator_state import MalSimulatorState

if TYPE_CHECKING:
    from malsim.mal_simulator.types import AgentStates
    from malsim.mal_simulator.defender_state import MalSimDefenderState

logger = logging.getLogger(__name__)


def defender_is_terminated(agent_states: AgentStates, alive_agents: set[str]) -> bool:
    """Check if defender is terminated
    Can be overridden by subclass for custom termination condition.
    """
    # Defender is terminated if all attackers are terminated
    return all(
        attacker_is_terminated(a)
        for a in get_attacker_agents(agent_states, alive_agents)
    )


def defender_step(
    sim_state: MalSimulatorState,
    agent: MalSimDefenderState,
    nodes: list[AttackGraphNode],
) -> tuple[list[AttackGraphNode], set[AttackGraphNode]]:
    """Enable defense step nodes with defender.

    Args:
    agent - the agent to activate defense nodes with
    nodes - the defense step nodes to enable

    Returns a tuple of a list and a set, `enabled_defenses`
    and `attack_steps_made_unviable`.
    """

    logger.info('Stepping with %s', agent.name)
    enabled_defenses: list[AttackGraphNode] = list()
    attack_steps_made_unviable: set[AttackGraphNode] = set()

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
            sim_state.graph_state.viability_per_node, made_unviable = (
                make_node_unviable(
                    node,
                    sim_state.graph_state.viability_per_node,
                    sim_state.graph_state.impossible_attack_steps,
                )
            )
            attack_steps_made_unviable |= made_unviable
            logger.info(
                'Defender agent "%s" enabled "%s" (reward: %d).',
                agent.name,
                node.full_name,
                node_reward(agent, node),
            )

    return enabled_defenses, attack_steps_made_unviable
