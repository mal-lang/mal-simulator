from __future__ import annotations
from typing import TYPE_CHECKING
import logging

import numpy as np

from maltoolbox.attackgraph import AttackGraphNode

from malsim.mal_simulator.graph_utils import (
    node_is_traversable,
    node_is_viable,
    node_reward,
)
from malsim.mal_simulator.state_query import node_ttc_value
from malsim.mal_simulator.settings import TTCMode
from malsim.mal_simulator.ttc_utils import (
    TTCDist,
)
from malsim.mal_simulator.simulator_state import MalSimulatorState

if TYPE_CHECKING:
    from malsim.mal_simulator.attacker_state import MalSimAttackerState

logger = logging.getLogger(__name__)


def attacker_is_terminated(attacker_state: MalSimAttackerState) -> bool:
    """Check if attacker is terminated
    Can be overridden by subclass for custom termination condition.

    Args:
    - attacker_state: the attacker state to check for termination
    """

    if len(attacker_state.action_surface) == 0:
        # Attacker is terminated if it has no more actions to take
        logger.info(
            'Attacker "%s" action surface is empty, terminate', attacker_state.name
        )
        return True
    if attacker_state.goals:
        # Attacker is terminated if it has goals and all goals are met
        return (
            attacker_state.goals & attacker_state.performed_nodes
            == attacker_state.goals
        )
    # Otherwise not terminated
    return False


def attempt_attacker_step(
    sim_state: MalSimulatorState,
    rng: np.random.Generator,
    ttc_mode: TTCMode,
    agent: MalSimAttackerState,
    node: AttackGraphNode,
) -> bool:
    """Attempt a step with a TTC distribution.

    Return True if the attempt was successful.
    """

    num_attempts = agent.num_attempts[node] + 1

    if node in agent.ttc_overrides:
        # If this agent has custom ttc distribution set for this node, use it
        ttc_dist = agent.ttc_overrides[node]
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


def attacker_step(
    sim_state: MalSimulatorState,
    agent: MalSimAttackerState,
    nodes: list[AttackGraphNode],
    rng: np.random.Generator,
) -> tuple[list[AttackGraphNode], list[AttackGraphNode]]:
    """Compromise attack step nodes with attacker

    Args:
    agent - the agent to compromise nodes with
    nodes - the nodes to compromise

    Returns: two lists with compromised, attempted nodes
    """

    logger.info('Stepping with agent %s', agent.name)
    successful_compromises: list[AttackGraphNode] = list()
    attempted_compromises: list[AttackGraphNode] = list()

    for node in nodes:
        assert node == sim_state.attack_graph.nodes[node.id], (
            f'{agent.name} tried to enable a node that is not part '
            'of this simulators attack_graph. Make sure the node '
            'comes from the agents action surface.'
        )

        if node in agent.entry_points:
            # Entrypoints can be compromised as long as they are viable
            can_compromise = node_is_viable(sim_state, node)
        else:
            # Otherwise it is limited by traversability
            can_compromise = node_is_traversable(sim_state, agent.performed_nodes, node)
        if can_compromise:
            if attempt_attacker_step(
                sim_state, rng, sim_state.settings.ttc_mode, agent, node
            ):
                successful_compromises.append(node)
                logger.info(
                    'Attacker agent "%s" compromised "%s" (reward: %d).',
                    agent.name,
                    node.full_name,
                    node_reward(agent, node),
                )
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

    return successful_compromises, attempted_compromises
