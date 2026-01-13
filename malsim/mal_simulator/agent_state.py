from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from types import MappingProxyType

from maltoolbox.attackgraph import AttackGraphNode

from malsim.mal_simulator.ttc_utils import TTCDist
from malsim.mal_simulator.simulator_state import MalSimulatorState

@dataclass(frozen=True)
class MalSimAgentState:
    """Stores the state of an agent in the simulator"""

    # Identifier of the agent, used in MalSimulator for lookup
    name: str
    # Reference to the simulator
    sim_state: MalSimulatorState
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


AgentStates = dict[str, MalSimAttackerState | MalSimDefenderState]
AgentRewards = dict[str, float]


def get_attacker_agents(
    agent_states: AgentStates, alive_agents: set[str], only_alive: bool = False
) -> list[MalSimAttackerState]:
    """Return list of mutable attacker agent states of attackers.
    If `only_alive` is set to True, only return the agents that are alive.
    """
    return [
        a
        for a in agent_states.values()
        if (a.name in alive_agents or not only_alive)
        and isinstance(a, MalSimAttackerState)
    ]


def get_defender_agents(
    agent_states: AgentStates, alive_agents: set[str], only_alive: bool = False
) -> list[MalSimDefenderState]:
    """Return list of mutable defender agent states of defenders.
    If `only_alive` is set to True, only return the agents that are alive.
    """
    return [
        a
        for a in agent_states.values()
        if (a.name in alive_agents or not only_alive)
        and isinstance(a, MalSimDefenderState)
    ]
