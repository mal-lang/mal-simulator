from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional
from maltoolbox.attackgraph import AttackGraphNode
from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator.agent_state import MalSimAgentState
from malsim.mal_simulator.ttc_utils import TTCDist
from malsim.types import AgentStates


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

    # Agent specific rules for node properties
    reward_rule: Optional[NodePropertyRule] = None
    actionability_rule: Optional[NodePropertyRule] = None

    # Picklable
    def __getstate__(self) -> dict[str, Any]:
        # convert MappingProxyType to dict for pickling
        state = self.__dict__.copy()
        state['num_attempts'] = dict(state['num_attempts'])
        state['ttc_overrides'] = dict(state['ttc_overrides'])
        state['ttc_value_overrides'] = dict(state['ttc_value_overrides'])
        state['performed_nodes_order'] = dict(state['performed_nodes_order'])
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
        object.__setattr__(
            self,
            'performed_nodes_order',
            MappingProxyType(state['performed_nodes_order']),
        )
        # set other frozen attributes
        for key, value in state.items():
            if key not in ('num_attempts', 'ttc_overrides', 'ttc_value_overrides'):
                object.__setattr__(self, key, value)


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
