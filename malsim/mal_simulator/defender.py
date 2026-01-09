from collections.abc import Callable, Set
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
import numpy as np
from malsim.mal_simulator.agent_state import AgentStates
from malsim.mal_simulator.agent_state import (
    AgentSettings,
    MalSimDefenderState,
    get_attacker_agents,
)
from malsim.mal_simulator.attacker import attacker_is_terminated
from malsim.mal_simulator.false_alerts import (
    generate_false_negatives,
    generate_false_positives,
)
from malsim.mal_simulator.graph_processing import make_node_unviable
from malsim.mal_simulator.graph_state import GraphState
from malsim.mal_simulator.node_utils import (
    node_is_actionable,
    node_is_enabled_defense,
    node_is_observable,
    node_is_viable,
    node_reward,
)
import logging

logger = logging.getLogger(__name__)


def defender_step_reward(
    agent_settings: AgentSettings,
    enabled_defenses_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    enabled_attacks_func: Callable[[MalSimDefenderState], frozenset[AttackGraphNode]],
    defender_state: MalSimDefenderState,
    rewards: dict[AttackGraphNode, float],
) -> float:
    """
    Calculate current defender reward either cumulative or one-off.
    If cumulative, sum previous and one-off reward, otherwise
    just return the one-off reward.

    Args:
    - defender_state: the defender state before defenses were enabled
    - reward_mode: which way to calculate reward
    """
    enabled_defenses = enabled_defenses_func(defender_state)
    compromised_nodes = enabled_attacks_func(defender_state)

    # Defender is penalized for compromised steps and enabled defenses
    step_reward = -sum(
        node_reward(agent_settings, rewards, n, defender_state.name)
        for n in enabled_defenses | compromised_nodes
    )

    return step_reward


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
    graph_state: GraphState,
    agent: MalSimDefenderState,
    rewards: dict[AttackGraphNode, float],
    agent_settings: AgentSettings,
    nodes: list[AttackGraphNode],
    attack_graph: AttackGraph,
) -> tuple[list[AttackGraphNode], set[AttackGraphNode]]:
    """Enable defense step nodes with defender.

    Args:
    agent - the agent to activate defense nodes with
    nodes - the defense step nodes to enable

    Returns a tuple of a list and a set, `enabled_defenses`
    and `attack_steps_made_unviable`.
    """

    enabled_defenses: list[AttackGraphNode] = list()
    attack_steps_made_unviable: set[AttackGraphNode] = set()

    for node in nodes:
        assert node == attack_graph.nodes[node.id], (
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
            graph_state.viability_per_node, made_unviable = make_node_unviable(
                node,
                graph_state.viability_per_node,
                graph_state.impossible_attack_steps,
            )
            attack_steps_made_unviable |= made_unviable
            logger.info(
                'Defender agent "%s" enabled "%s" (reward: %d).',
                agent.name,
                node.full_name,
                node_reward(agent_settings, rewards, node, agent.name),
            )

    return enabled_defenses, attack_steps_made_unviable


def defender_observed_nodes(
    agent_settings: AgentSettings,
    false_negative_rates: dict[AttackGraphNode, float],
    false_positive_rates: dict[AttackGraphNode, float],
    rng: np.random.Generator,
    attack_graph: AttackGraph,
    node_observabilities: dict[AttackGraphNode, bool],
    defender_name: str,
    compromised_nodes: Set[AttackGraphNode],
) -> set[AttackGraphNode]:
    """Generate set of observed compromised nodes
    From set of compromised nodes, generate observed nodes for a defender
    in regards to observability, false negatives and false positives.
    """
    observable_steps = set(
        n
        for n in compromised_nodes
        if node_is_observable(agent_settings, node_observabilities, n, defender_name)
    )
    false_negatives = generate_false_negatives(
        agent_settings, false_negative_rates, rng, defender_name, compromised_nodes
    )
    false_positives = generate_false_positives(
        false_positive_rates, agent_settings, attack_graph, defender_name, rng
    )

    observed_nodes = (observable_steps - false_negatives) | false_positives
    return observed_nodes


def get_defense_surface(
    agent_settings: AgentSettings,
    attack_graph: AttackGraph,
    agent_states: AgentStates,
    graph_state: GraphState,
    live_agents: set[str],
    node_actionabilities: dict[AttackGraphNode, bool],
    agent_name: str,
) -> set[AttackGraphNode]:
    """Get the defense surface.
    All non-suppressed defense steps that are not already enabled.

    Arguments:
    graph       - the attack graph
    """
    return {
        node
        for node in attack_graph.defense_steps
        if node_is_actionable(agent_settings, node_actionabilities, node, agent_name)
        and node_is_viable(graph_state, attack_graph, node)
        and 'suppress' not in node.tags  # type: ignore
        and not node_is_enabled_defense(attack_graph, agent_states, live_agents, node)
    }
