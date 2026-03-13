from collections.abc import Set
from dataclasses import dataclass
from typing import Any
from maltoolbox.attackgraph import AttackGraphNode
from malsim.config.agent_settings import DefenderSettings
from malsim.mal_simulator.agent_state import MalSimAgentState


@dataclass(frozen=True)
class MalSimDefenderState(MalSimAgentState):
    """Stores the state of a defender in the simulator"""

    # Contains all steps performed by any attacker
    compromised_nodes: Set[AttackGraphNode]
    # Contains steps performed by any attacker in last step
    step_compromised_nodes: Set[AttackGraphNode]
    # Contains all observed steps by any attacker
    # in regards to false positives/negatives and observability
    observed_nodes: Set[AttackGraphNode]
    # Contains observed steps made by any attacker in last step
    step_observed_nodes: Set[AttackGraphNode]
    settings: DefenderSettings

    # Pickling
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['performed_nodes_order'] = dict(state['performed_nodes_order'])
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(
            self,
            'performed_nodes_order',
            state['performed_nodes_order'],
        )
        for key, value in state.items():
            if key not in ('performed_nodes_order'):
                object.__setattr__(self, key, value)
