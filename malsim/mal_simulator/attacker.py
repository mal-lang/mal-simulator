from typing import Optional
from malsim.mal_simulator.agent_state import AgentStates
from malsim.mal_simulator.agent_state import (
    AgentSettings,
    MalSimAttackerState,
)
from malsim.mal_simulator.graph_state import GraphState
from malsim.mal_simulator.node_utils import (
    full_name_dict_to_node_dict,
    full_name_or_node_to_node,
    node_is_actionable,
    node_is_necessary,
    node_is_traversable,
    node_is_viable,
    node_reward,
    node_ttc_value,
)
from malsim.mal_simulator.settings import MalSimulatorSettings, RewardMode, TTCMode
import logging
from malsim.mal_simulator.ttc_utils import (
    TTCDist,
    attack_step_ttc_values,
    get_impossible_attack_steps,
)


import numpy as np
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode


from collections.abc import Callable, Set

from malsim.scenario import AttackerSettings

logger = logging.getLogger(__name__)

def attacker_step_reward(
    performed_attacks_func: Callable[[MalSimAttackerState], frozenset[AttackGraphNode]],
    attacker_state: MalSimAttackerState,
    rng: np.random.Generator,
    agent_settings: AgentSettings,
    reward_mode: RewardMode,
    ttc_mode: TTCMode,
    rewards: dict[AttackGraphNode, float],
) -> float:
    """
    Calculate current attacker reward either cumulative or one-off.
    If cumulative, sum previous and one-off reward, otherwise
    just return the one-off reward.

    Args:
    - attacker_state: the current attacker state
    - reward_mode: which way to calculate reward
    """

    performed_steps = performed_attacks_func(attacker_state)
    action = attacker_state.step_attempted_nodes

    # Attacker is rewarded for compromised nodes
    step_reward = sum(
        node_reward(agent_settings, rewards, n, attacker_state.name)
        for n in performed_steps
    )

    if ttc_mode != TTCMode.DISABLED:
        # If TTC Mode is not disabled, attacker is penalized for each attempt
        step_reward -= len(action)
    elif ttc_mode == TTCMode.DISABLED:
        # If TTC Mode is disabled but reward mode uses TTCs, penalize with TTCs
        for node in performed_steps:
            if reward_mode == RewardMode.EXPECTED_TTC:
                step_reward -= TTCDist.from_node(node).expected_value if node.ttc else 0  # type: ignore
            elif reward_mode == RewardMode.SAMPLE_TTC:
                step_reward -= (
                    TTCDist.from_node(node).sample_value(rng) if node.ttc else 0  # type: ignore
                )

    return step_reward


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
    agent_states: AgentStates,
    graph_state: GraphState,
    attack_graph: AttackGraph,
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
        _node_ttc_value = node_ttc_value(
            agent_states, graph_state, attack_graph, ttc_mode, node, agent.name
        )
        return num_attempts + 1 >= _node_ttc_value

    else:
        raise ValueError(f'Invalid TTC mode: {ttc_mode}')


def attacker_step(
    agent_states: AgentStates,
    rng: np.random.Generator,
    ttc_mode: TTCMode,
    agent_settings: AgentSettings,
    rewards: dict[AttackGraphNode, float],
    graph_state: GraphState,
    attack_graph: AttackGraph,
    agent: MalSimAttackerState,
    nodes: list[AttackGraphNode],
) -> tuple[list[AttackGraphNode], list[AttackGraphNode]]:
    """Compromise attack step nodes with attacker

    Args:
    agent - the agent to compromise nodes with
    nodes - the nodes to compromise

    Returns: two lists with compromised, attempted nodes
    """

    successful_compromises: list[AttackGraphNode] = list()
    attempted_compromises: list[AttackGraphNode] = list()

    for node in nodes:
        assert node == attack_graph.nodes[node.id], (
            f'{agent.name} tried to enable a node that is not part '
            'of this simulators attack_graph. Make sure the node '
            'comes from the agents action surface.'
        )

        if node in agent.entry_points:
            # Entrypoints can be compromised as long as they are viable
            can_compromise = node_is_viable(graph_state, attack_graph, node)
        else:
            # Otherwise it is limited by traversability
            can_compromise = node_is_traversable(
                graph_state, attack_graph, agent.performed_nodes, node
            )
        if can_compromise:
            if attempt_attacker_step(
                agent_states, graph_state, attack_graph, rng, ttc_mode, agent, node
            ):
                successful_compromises.append(node)
                logger.info(
                    'Attacker agent "%s" compromised "%s" (reward: %d).',
                    agent.name,
                    node.full_name,
                    node_reward(agent_settings, rewards, node, agent.name),
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


def attacker_overriding_ttc_settings(
    attack_graph: AttackGraph,
    attacker_settings: AttackerSettings,
    ttc_mode: TTCMode,
    rng: np.random.Generator,
) -> tuple[
    dict[AttackGraphNode, TTCDist],
    dict[AttackGraphNode, float],
    set[AttackGraphNode],
]:
    """
    Get overriding TTC distributions, TTC values, and impossible attack steps
    from attacker settings if they exist.

    Returns three separate collections:
        - a dict of TTC distributions
        - a dict of TTC values
        - a set of impossible steps
    """

    if not attacker_settings.ttc_overrides:
        return {}, {}, set()

    ttc_overrides_names = attacker_settings.ttc_overrides.per_node(attack_graph)

    # Convert names to TTCDist objects and map from AttackGraphNode
    # objects instead of from full names
    ttc_overrides = {
        full_name_or_node_to_node(attack_graph, node): TTCDist.from_name(name)
        for node, name in ttc_overrides_names.items()
    }
    ttc_value_overrides = attack_step_ttc_values(
        ttc_overrides.keys(),
        ttc_mode,
        rng,
        ttc_dists=full_name_dict_to_node_dict(attack_graph, ttc_overrides),
    )
    impossible_step_overrides = get_impossible_attack_steps(
        ttc_overrides.keys(),
        rng,
        ttc_dists=full_name_dict_to_node_dict(attack_graph, ttc_overrides),
    )
    return ttc_overrides, ttc_value_overrides, impossible_step_overrides


def get_attack_surface(
    sim_settings: MalSimulatorSettings,
    graph_state: GraphState,
    attack_graph: AttackGraph,
    agent_settings: AgentSettings,
    agent_states: AgentStates,
    node_actionabilities: dict[AttackGraphNode, bool],
    agent_name: str,
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

            if skip_unviable and not node_is_viable(graph_state, attack_graph, child):
                continue

            if skip_unnecessary and not node_is_necessary(
                graph_state, attack_graph, child
            ):
                continue

            if not node_is_actionable(
                agent_settings, node_actionabilities, child, agent_name
            ):
                continue

            if node_is_traversable(graph_state, attack_graph, performed_nodes, child):
                attack_surface.add(child)

    return frozenset(attack_surface)
