from __future__ import annotations
from collections.abc import Set
from dataclasses import dataclass
from typing import Any
from maltoolbox.attackgraph import AttackGraphNode
from malsim.config.agent_settings import DefenderSettings
from malsim.mal_simulator.agent_state import AgentState


@dataclass(frozen=True)
class DefenderState(AgentState):
    """Stores the state of a defender in the simulator"""

    # Contains all steps performed by any attacker
    compromised_nodes: Set[AttackGraphNode]

    # Contains all observed steps by any attacker
    # in regards to false positives/negatives and observability
    observed_nodes: Set[AttackGraphNode]

    settings: DefenderSettings
    previous_state: DefenderState | None = None

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

    @property
    def step_observed_nodes(self):
        previous_observed_nodes = self.previous_state.observed_nodes if self.previous_state else set()
        return self.observed_nodes - previous_observed_nodes

    @property
    def step_compromised_nodes(self):
        previous_compromised_nodes = self.previous_state.compromised_nodes if self.previous_state else set()
        return self.compromised_nodes - previous_compromised_nodes

    @property
    def step_performed_nodes(self):
        previous_performed_nodes = self.previous_state.performed_nodes if self.previous_state else set()
        return self.performed_nodes - previous_performed_nodes

    @property
    def step_action_surface_additions(self):
        previous_action_surface = self.previous_state.action_surface if self.previous_state else set()
        return self.action_surface - previous_action_surface

    @property
    def step_action_surface_removals(self):
        previous_action_surface = self.previous_state.action_surface if self.previous_state else set()
        return previous_action_surface - self.action_surface
