from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping, Set
from typing import Any
from maltoolbox.attackgraph import AttackGraphNode
from malsim.config.agent_settings import AttackerSettings
from malsim.mal_simulator.agent_state import AgentState


@dataclass(frozen=True)
class AttackerState(AgentState):
    """Stores the state of an attacker in the simulator"""

    settings: AttackerSettings[AttackGraphNode]
    # Number of attempts to compromise a step (used for ttc caculations)
    num_attempts: Mapping[AttackGraphNode, int]
    # Steps attempted but not succeeded (because of TTC value)
    attempted_nodes: Set[AttackGraphNode]

    ttc_values: Mapping[AttackGraphNode, float]

    # Goals affect simulation termination but is optional
    goals: Set[AttackGraphNode] = field(default_factory=frozenset)

    # Only used if `ttc_overrides` is set
    impossible_steps: Set[AttackGraphNode] = field(
        default_factory=frozenset
    )  # Steps that are impossible to perform

    previous_state: AttackerState | None = None

    # Picklable
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['num_attempts'] = dict(state['num_attempts'])
        state['performed_nodes_order'] = dict(state['performed_nodes_order'])
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(self, 'num_attempts', state['num_attempts'])

        object.__setattr__(
            self,
            'performed_nodes_order',
            state['performed_nodes_order'],
        )
        # set other frozen attributes
        for key, value in state.items():
            if key not in ('num_attempts', 'ttc_overrides', 'ttc_value_overrides'):
                object.__setattr__(self, key, value)

    @property
    def step_attempted_nodes(self):
        previous_attempted_nodes = self.previous_state.attempted_nodes if self.previous_state else set()
        return self.attempted_nodes - previous_attempted_nodes

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
