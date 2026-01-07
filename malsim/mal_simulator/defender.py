from typing import Optional
from malsim.mal_simulator.agent_state import get_defender_agents
from malsim.mal_simulator.agent_state import (
    MalSimDefenderState,
    create_defender_state,
    AgentData,
    get_attacker_agents,
)
from malsim.mal_simulator.graph_state import GraphState
from malsim.mal_simulator.sim_data import SimData
from malsim.mal_simulator.attacker import attacker_is_terminated
from malsim.mal_simulator.graph_processing import make_node_unviable
from malsim.mal_simulator.node import node_reward
from malsim.mal_simulator.settings import RewardMode
from maltoolbox.attackgraph import AttackGraphNode, AttackGraph
import logging

from malsim.scenario import DefenderSettings

logger = logging.getLogger(__name__)


def register_defender(
    sim_data: SimData, graph_state: GraphState, agent_data: AgentData, name: str
) -> None:
    """Register a mal sim defender agent without setting object"""
    defender_settings = DefenderSettings(name)
    register_defender_settings(sim_data, graph_state, agent_data, defender_settings)


def register_defender_settings(
    sim_data: SimData,
    attack_graph: AttackGraph,
    graph_state: GraphState,
    agent_data: AgentData,
    defender_settings: DefenderSettings,
    compromised_nodes: set[AttackGraphNode],
    defender_reward_mode: RewardMode,
) -> None:
    """Register a mal sim defender agent"""

    if get_defender_agents(agent_data):
        print(
            'WARNING: You have registered more than one defender agent. '
            'It does not make sense to have more than one, '
            'since all defender agents have the same state.'
        )
    assert defender_settings.name not in agent_data.agent_settings, (
        f'Duplicate agent named {defender_settings.name} not allowed'
    )

    agent_data.agent_settings[defender_settings.name] = defender_settings

    agent_state = initial_defender_state(
        sim_data=sim_data,
        graph_state=graph_state,
        attack_graph=sim_data,
        agent_data=agent_data,
        defender_settings=defender_settings,
        pre_compromised_nodes=compromised_nodes,
        pre_enabled_defenses=graph_state.pre_enabled_defenses,
    )
    agent_data.agent_states[defender_settings.name] = agent_state
    agent_data.alive_agents.add(defender_settings.name)
    agent_data.agent_rewards[defender_settings.name] = defender_step_reward(
        sim_data, agent_state, defender_reward_mode
    )


def defender_step(
    sim_data: SimData,
    graph_state: GraphState,
    agent: MalSimDefenderState,
    nodes: list[AttackGraphNode],
    attack_graph: AttackGraph,
    agent_settings: DefenderSettings,
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
                node_reward(sim_data, node, agent_settings),
            )

    return enabled_defenses, attack_steps_made_unviable


def defender_step_reward(
    sim_data: SimData,
    defender_state: MalSimDefenderState,
    reward_mode: RewardMode,
    defender_settings: DefenderSettings,
) -> float:
    """
    Calculate current defender reward either cumulative or one-off.
    If cumulative, sum previous and one-off reward, otherwise
    just return the one-off reward.

    Args:
    - defender_state: the defender state before defenses were enabled
    - reward_mode: which way to calculate reward
    """
    step_enabled_defenses = defender_state.step_performed_nodes
    step_compromised_nodes = defender_state.step_compromised_nodes

    # Defender is penalized for compromised steps and enabled defenses
    step_reward = -sum(
        node_reward(sim_data, n, defender_settings)
        for n in step_enabled_defenses | step_compromised_nodes
    )

    return step_reward


def initial_defender_state(
    sim_data: SimData,
    attack_graph: AttackGraph,
    agent_data: AgentData,
    graph_state: GraphState,
    defender_settings: DefenderSettings,
    pre_compromised_nodes: set[AttackGraphNode],
    pre_enabled_defenses: set[AttackGraphNode],
) -> MalSimDefenderState:
    """Create a defender state from defender settings"""
    return create_defender_state(
        sim_data=sim_data,
        graph_state=graph_state,
        agent_data=agent_data,
        attack_graph=attack_graph,
        agent_settings=defender_settings,
        step_compromised_nodes=pre_compromised_nodes,
        step_enabled_defenses=pre_enabled_defenses,
    )


def defender_is_terminated(agent_data: AgentData) -> bool:
    """Check if defender is terminated
    Can be overridden by subclass for custom termination condition.
    """
    # Defender is terminated if all attackers are terminated
    return all(attacker_is_terminated(a) for a in get_attacker_agents(agent_data))
