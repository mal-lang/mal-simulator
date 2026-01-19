from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional
from maltoolbox.attackgraph import AttackGraphNode
from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.agent_state import MalSimAgentState
from malsim.types import AgentStates


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

    # Agent specific rules for node properties
    reward_rule: Optional[NodePropertyRule] = None
    actionability_rule: Optional[NodePropertyRule] = None
    false_positive_rates_rule: Optional[NodePropertyRule] = None
    false_negative_rates_rule: Optional[NodePropertyRule] = None
    observability_rule: Optional[NodePropertyRule] = None

    # Pickling
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['performed_nodes_order'] = dict(state['performed_nodes_order'])
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(
            self,
            'performed_nodes_order',
            MappingProxyType(state['performed_nodes_order']),
        )
        for key, value in state.items():
            if key not in ('performed_nodes_order'):
                object.__setattr__(self, key, value)


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
