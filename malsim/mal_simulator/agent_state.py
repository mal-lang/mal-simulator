from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional
from collections.abc import Set, Mapping
from types import MappingProxyType

from maltoolbox.attackgraph import AttackGraphNode


from malsim.mal_simulator.false_alerts import defender_observed_nodes
from malsim.mal_simulator.graph_state import GraphState
from malsim.mal_simulator.node import node_is_actionable, node_is_viable
from malsim.mal_simulator.settings import MalSimulatorSettings
from malsim.mal_simulator.sim_data import SimData
from malsim.mal_simulator.surface import get_attack_surface
from malsim.mal_simulator.ttc_utils import TTCDist
from malsim.scenario import AttackerSettings, DefenderSettings
from maltoolbox.attackgraph import AttackGraph


from malsim.scenario import AttackerSettings, DefenderSettings


from typing import NamedTuple


class AgentData(NamedTuple):
    agent_settings: dict[str, AttackerSettings | DefenderSettings] = {}
    agent_states: dict[str, MalSimAttackerState | MalSimDefenderState] = {}
    alive_agents: set[str] = set()
    agent_rewards: dict[str, float] = {}


def get_defender_agents(
    agent_data: AgentData, only_alive: bool = False
) -> list[MalSimDefenderState]:
    """Return list of mutable defender agent states of defenders.
    If `only_alive` is set to True, only return the agents that are alive.
    """
    return [
        a
        for a in agent_data.agent_states.values()
        if (a.name in agent_data.alive_agents or not only_alive)
        and isinstance(a, MalSimDefenderState)
    ]


def get_attacker_agents(
    agent_data: AgentData, only_alive: bool = False
) -> list[MalSimAttackerState]:
    """Return list of mutable attacker agent states of attackers.
    If `only_alive` is set to True, only return the agents that are alive.
    """
    return [
        a
        for a in agent_data.agent_states.values()
        if (a.name in agent_data.alive_agents or not only_alive)
        and isinstance(a, MalSimAttackerState)
    ]


def get_defense_surface(
    sim_data: SimData,
    attack_graph: AttackGraph,
    graph_state: GraphState,
    agent_settings: DefenderSettings,
    agent_data: AgentData,
) -> set[AttackGraphNode]:
    """Get the defense surface.
    All non-suppressed defense steps that are not already enabled.

    Arguments:
    graph       - the attack graph
    """
    return {
        node
        for node in attack_graph.defense_steps
        if node_is_actionable(sim_data, node, agent_settings)
        and node_is_viable(graph_state, node)
        and 'suppress' not in node.tags
        and not node_is_enabled_defense(agent_data, node)
    }

@dataclass(frozen=True)
class MalSimAgentState:
    """Stores the state of an agent in the simulator"""

    # Identifier of the agent, used in MalSimulator for lookup
    name: str
    # Contains possible actions for the agent in the next step
    action_surface: frozenset[AttackGraphNode]
    # Contains all nodes that this agent has performed successfully
    performed_nodes: frozenset[AttackGraphNode]
    # Contains the nodes performed successfully in the last step
    step_performed_nodes: frozenset[AttackGraphNode]
    # Contains possible nodes that became available in the last step
    step_action_surface_additions: frozenset[AttackGraphNode]
    # Contains nodes that became unavailable in the last step
    step_action_surface_removals: frozenset[AttackGraphNode]
    # Contains nodes that became unviable in the last step by defender actions
    step_unviable_nodes: frozenset[AttackGraphNode]
    iteration: int


@dataclass(frozen=True)
class MalSimAttackerState(MalSimAgentState):
    """Stores the state of an attacker in the simulator"""

    # The starting points of an attacker agent
    entry_points: frozenset[AttackGraphNode]
    # Number of attempts to compromise a step (used for ttc caculations)
    num_attempts: MappingProxyType[AttackGraphNode, int]
    # Steps attempted but not succeeded (because of TTC value)
    step_attempted_nodes: frozenset[AttackGraphNode]
    # Goals affect simulation termination but is optional
    goals: frozenset[AttackGraphNode] = field(default_factory=frozenset)

    # TTC distributions that override TTCs set in language
    ttc_overrides: MappingProxyType[AttackGraphNode, TTCDist] = field(
        default_factory=lambda: MappingProxyType({})
    )
    # Only used if `ttc_overrides` is set
    ttc_value_overrides: MappingProxyType[AttackGraphNode, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    # Only used if `ttc_overrides` is set
    impossible_step_overrides: frozenset[AttackGraphNode] = field(
        default_factory=frozenset
    )  # Steps that are impossible to perform

    # Picklable
    def __getstate__(self) -> dict[str, Any]:
        # convert MappingProxyType to dict for pickling
        state = self.__dict__.copy()
        state['num_attempts'] = dict(state['num_attempts'])
        state['ttc_overrides'] = dict(state['ttc_overrides'])
        state['ttc_value_overrides'] = dict(state['ttc_value_overrides'])
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # restore MappingProxyType, ttc_overrides, ttc_value_overrides
        object.__setattr__(
            self, 'num_attempts', MappingProxyType(state['num_attempts'])
        )
        object.__setattr__(
            self, 'ttc_overrides', MappingProxyType(state['ttc_overrides'])
        )
        object.__setattr__(
            self, 'ttc_value_overrides', MappingProxyType(state['ttc_value_overrides'])
        )
        # set other frozen attributes
        for key, value in state.items():
            if key not in ('num_attempts', 'ttc_overrides', 'ttc_value_overrides'):
                object.__setattr__(self, key, value)


def create_attacker_state(
    graph_state: GraphState,
    sim_data: SimData,
    attack_graph: AttackGraph,
    sim_settings: MalSimulatorSettings,
    agent_settings: AttackerSettings,
    entry_points: Set[AttackGraphNode],
    goals: Set[AttackGraphNode] = frozenset(),
    step_compromised_nodes: Set[AttackGraphNode] = frozenset(),
    step_attempted_nodes: Set[AttackGraphNode] = frozenset(),
    step_nodes_made_unviable: Set[AttackGraphNode] = frozenset(),
    ttc_overrides: Mapping[AttackGraphNode, TTCDist] = MappingProxyType({}),
    ttc_value_overrides: Mapping[AttackGraphNode, float] = MappingProxyType({}),
    impossible_step_overrides: Set[AttackGraphNode] = frozenset(),
    previous_state: Optional[MalSimAttackerState] = None,
) -> MalSimAttackerState:
    """
    Update a previous attacker state based on what the agent compromised
    and what nodes became unviable.
    """

    if previous_state is None:
        # Initial compromised nodes
        if sim_settings.compromise_entrypoints_at_start:
            step_compromised_nodes |= entry_points
        compromised_nodes = step_compromised_nodes

        # Create an initial attack surface
        new_action_surface = get_attack_surface(
            graph_state, sim_data, sim_settings, agent_settings, compromised_nodes
        )
        action_surface_removals: set[AttackGraphNode] = set()
        action_surface_additions = new_action_surface

        if not sim_settings.compromise_entrypoints_at_start:
            # If entrypoints not compromised at start,
            # we need to put them in action surface
            new_action_surface |= entry_points
            action_surface_additions |= entry_points

        previous_num_attempts: Mapping[AttackGraphNode, int] = {
            n: 0 for n in attack_graph.attack_steps
        }

    else:
        ttc_value_overrides = previous_state.ttc_value_overrides
        impossible_step_overrides = previous_state.impossible_step_overrides
        compromised_nodes = previous_state.performed_nodes | step_compromised_nodes

        # Build on previous attack surface (for performance)
        action_surface_additions = (
            get_attack_surface(
                graph_state,
                sim_data,
                sim_settings,
                agent_settings,
                compromised_nodes | step_compromised_nodes,
                from_nodes=step_compromised_nodes,
            )
            - previous_state.action_surface
        )
        action_surface_removals = set(
            (step_nodes_made_unviable & previous_state.action_surface)
            | step_compromised_nodes
        )
        new_action_surface = frozenset(
            (previous_state.action_surface - action_surface_removals)
            | action_surface_additions
        )
        previous_num_attempts = previous_state.num_attempts

    new_num_attempts = dict(previous_num_attempts)
    for node in step_attempted_nodes:
        new_num_attempts[node] += 1

    return MalSimAttackerState(
        name,
        entry_points=frozenset(entry_points),
        goals=frozenset(goals),
        sim=sim,
        performed_nodes=frozenset(compromised_nodes | step_compromised_nodes),
        action_surface=new_action_surface,
        step_action_surface_additions=action_surface_additions,
        step_action_surface_removals=frozenset(action_surface_removals),
        step_performed_nodes=frozenset(step_compromised_nodes),
        step_unviable_nodes=frozenset(step_nodes_made_unviable),
        step_attempted_nodes=frozenset(step_attempted_nodes),
        num_attempts=MappingProxyType(new_num_attempts),
        ttc_overrides=MappingProxyType(ttc_overrides),
        ttc_value_overrides=MappingProxyType(ttc_value_overrides),
        impossible_step_overrides=frozenset(impossible_step_overrides),
        iteration=(previous_state.iteration + 1) if previous_state else 1,
    )


@dataclass(frozen=True)
class MalSimDefenderState(MalSimAgentState):
    """Stores the state of a defender in the simulator"""

    # Contains all steps performed by any attacker
    compromised_nodes: frozenset[AttackGraphNode]
    # Contains steps performed by any attacker in last step
    step_compromised_nodes: frozenset[AttackGraphNode]
    # Contains all observed steps by any attacker
    # in regards to false positives/negatives and observability
    observed_nodes: frozenset[AttackGraphNode]
    # Contains observed steps made by any attacker in last step
    step_observed_nodes: frozenset[AttackGraphNode]

    @property
    def step_all_compromised_nodes(self) -> None:
        raise DeprecationWarning(
            "Use 'step_compromised_nodes' instead of 'step_all_compromised_nodes'"
        )


def create_defender_state(
    sim_data: SimData,
    attack_graph: AttackGraph,
    agent_data: AgentData,
    graph_state: GraphState,
    agent_settings: DefenderSettings,
    step_compromised_nodes: Set[AttackGraphNode] = frozenset(),
    step_enabled_defenses: Set[AttackGraphNode] = frozenset(),
    step_nodes_made_unviable: Set[AttackGraphNode] = frozenset(),
    previous_state: Optional[MalSimDefenderState] = None,
) -> MalSimDefenderState:
    """
    Update a previous defender state based on what steps
    were enabled/compromised during last step
    """

    action_surface = frozenset(
        get_defense_surface(
            sim_data, attack_graph, graph_state, agent_settings, agent_data
        )
    )

    if previous_state is None:
        # Initialize
        previous_enabled_defenses: Set[AttackGraphNode] = frozenset()
        previous_compromised_nodes: Set[AttackGraphNode] = frozenset()
        previous_observed_nodes: Set[AttackGraphNode] = frozenset()
        action_surface_additions: Set[AttackGraphNode] = action_surface
        action_surface_removals: Set[AttackGraphNode] = frozenset()
    else:
        previous_enabled_defenses = previous_state.performed_nodes
        previous_compromised_nodes = previous_state.compromised_nodes
        previous_observed_nodes = previous_state.observed_nodes
        action_surface_additions = frozenset()
        action_surface_removals = step_enabled_defenses

    step_observed_nodes = defender_observed_nodes(name, step_compromised_nodes)
    return MalSimDefenderState(
        name,
        sim=sim,
        performed_nodes=frozenset(previous_enabled_defenses | step_enabled_defenses),
        compromised_nodes=frozenset(
            previous_compromised_nodes | step_compromised_nodes
        ),
        step_compromised_nodes=frozenset(step_compromised_nodes),
        observed_nodes=frozenset(previous_observed_nodes | step_observed_nodes),
        step_observed_nodes=frozenset(step_observed_nodes),
        step_action_surface_additions=frozenset(action_surface_additions),
        step_action_surface_removals=frozenset(action_surface_removals),
        action_surface=frozenset(action_surface),
        step_performed_nodes=frozenset(step_enabled_defenses),
        step_unviable_nodes=frozenset(step_nodes_made_unviable),
        iteration=(previous_state.iteration + 1) if previous_state else 1,
    )


def node_is_enabled_defense(agent_data: AgentData, node: AttackGraphNode) -> bool:
    """Get a nodes defense status"""
    return any(
        node in agent.performed_nodes for agent in get_defender_agents(agent_data)
    )


def node_is_compromised(agent_data: AgentData, node: AttackGraphNode) -> bool:
    """Return True if node is compromised by any attacker agent"""
    return any(
        node in attacker_agent.performed_nodes
        for attacker_agent in get_attacker_agents(agent_data)
    )


def node_ttc_value(
    graph_state: GraphState,
    node: AttackGraphNode,
    agent_state: Optional[MalSimAttackerState] = None,
) -> float:
    """Return ttc value of node if it has been sampled"""

    # assert malsim.sim_settings.ttc_mode in (
    #     TTCMode.PRE_SAMPLE,
    #     TTCMode.EXPECTED_VALUE,
    # ), 'TTC value only when TTCMode is PRE_SAMPLE or EXPECTED_VALUE'

    if agent_state:
        # If agent name is given and it overrides the global TTC values
        # Return that value instead of the global ttc value
        if node in agent_state.ttc_value_overrides:
            return agent_state.ttc_value_overrides[node]

    assert node in graph_state.ttc_values, (
        f'Node {node.full_name} does not have a ttc value'
    )
    return graph_state.ttc_values[node]
