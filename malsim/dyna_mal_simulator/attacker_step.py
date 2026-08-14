from __future__ import annotations
from typing import TYPE_CHECKING
import logging

import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from malsim.config.sim_settings import TTCMode
from malsim.dyna_mal_simulator.model_effects import execute_model_effects
from malsim.dyna_mal_simulator.simulator_state import DynaMalSimulatorState
from malsim.mal_simulator.graph_utils import (
    node_is_traversable,
)
from malsim.mal_simulator.attacker_step import (
    attacker_step_effects,
)
from malsim.mal_simulator.state_query import node_ttc_value
from malsim.mal_simulator.ttc_utils import TTCDist

if TYPE_CHECKING:
    from malsim.mal_simulator.attacker_state import AttackerState

logger = logging.getLogger(__name__)


def dyna_attacker_step(
    sim_state: DynaMalSimulatorState,
    agent: AttackerState,
    nodes: list[AttackGraphNode],
    rng: np.random.Generator,
) -> tuple[list[AttackGraphNode], list[AttackGraphNode], DynaMalSimulatorState]:
    """Compromise attack step nodes with attacker

    Args:
    agent - the agent to compromise nodes with
    nodes - the nodes to compromise

    Returns: two lists with compromised, attempted nodes
    """

    logger.debug('Stepping with agent %s', agent.name)
    successful_compromises: list[AttackGraphNode] = []
    attempted_compromises: list[AttackGraphNode] = []

    for node in nodes:
        assert node == sim_state.attack_graph.nodes[node.id], (
            f'{agent.name} tried to enable a node that is not part '
            'of this simulators attack_graph. Make sure the node '
            'comes from the agents action surface.'
        )

        if node in agent.settings.entry_points:
            # Entrypoints can always be compromised
            # TODO: should this actually be the case?
            can_compromise = True
        else:
            # Otherwise attacker is limited by attack surface and traversability
            can_compromise = node in agent.action_surface and node_is_traversable(
                sim_state, agent.performed_nodes, node
            )

        if can_compromise:
            if dyna_attempt_attacker_step(
                sim_state, rng, sim_state.settings.ttc_mode, agent, node
            ):
                successful_compromises.append(node)
                logger.info(
                    'Attacker agent "%s" compromised "%s"', agent.name, node.full_name
                )
                # Run effects as a compromise of performing `node`
                sim_state = execute_model_effects(sim_state, node, rng)
                successful_compromises += attacker_step_effects(sim_state, agent, node)
            else:
                logger.info(
                    'Attacker agent "%s" attempted "%s" (attempt %d).',
                    agent.name,
                    node.full_name,
                    agent.num_attempts[node],
                )
                attempted_compromises.append(node)

        else:
            logger.warning('Attacker could not compromise %s', node.full_name)

    return successful_compromises, attempted_compromises, sim_state


def dyna_attempt_attacker_step(
    sim_state: DynaMalSimulatorState,
    rng: np.random.Generator,
    ttc_mode: TTCMode,
    agent: AttackerState,
    node: AttackGraphNode,
) -> bool:
    """Attempt a step with a TTC distribution.

    Return True if the attempt was successful.
    """

    # # New nodes may be created during the simulation,
    # # so we need to check if the node is in the num_attempts dict
    # if node not in agent.num_attempts:
    #     num_attempts = 0
    # else:
    #     num_attempts = agent.num_attempts[node] + 1
    num_attempts = agent.num_attempts.get(node, 0) + 1

    if agent.settings.ttc_dists and node in agent.settings.ttc_dists:
        # If this agent has custom ttc distribution set for this node, use it
        ttc_dist = agent.settings.ttc_dists[node]
    else:
        ttc_dist = TTCDist.from_node(node)

    if ttc_mode == TTCMode.DISABLED:
        # Always suceed if disabled TTCs
        return True

    elif ttc_mode == TTCMode.EFFORT_BASED_PER_STEP_SAMPLE:
        # Run trial to decide success if config says so (SANDOR mode)
        return ttc_dist.attempt_ttc_with_effort(num_attempts, rng)

    elif ttc_mode == TTCMode.PER_STEP_SAMPLE:
        # Sample ttc value every time if config says so (ANDREI mode)
        _node_ttc_value = ttc_dist.sample_value(rng)
        return _node_ttc_value <= 1

    # Compare attempts to ttc expected value in EXPECTED_VALUE mode
    # or presampled ttcs in PRE_SAMPLE mode
    elif ttc_mode in (TTCMode.EXPECTED_VALUE, TTCMode.PRE_SAMPLE):
        _node_ttc_value = node_ttc_value(agent, node)
        return num_attempts + 1 >= _node_ttc_value

    else:
        raise ValueError(f'Invalid TTC mode: {ttc_mode}')
