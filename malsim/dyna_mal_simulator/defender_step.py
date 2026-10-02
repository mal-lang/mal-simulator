from __future__ import annotations
from typing import TYPE_CHECKING
from maltoolbox.attackgraph import AttackGraphNode
import logging

import numpy as np


from malsim.dyna_mal_simulator.model_effects import execute_model_effects
from malsim.dyna_mal_simulator.simulator_state import DynaMalSimulatorState

if TYPE_CHECKING:
    from malsim.mal_simulator.defender_state import DefenderState

logger = logging.getLogger(__name__)


def dyna_defender_step(
    sim_state: DynaMalSimulatorState,
    agent: DefenderState,
    nodes: list[AttackGraphNode],
    rng: np.random.Generator,
) -> tuple[list[AttackGraphNode], DynaMalSimulatorState]:
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
            sim_state = execute_model_effects(sim_state, node, rng)
            logger.info(
                'Defender agent "%s" enabled "%s"',
                agent.name,
                node.full_name,
            )

    return enabled_defenses, sim_state
