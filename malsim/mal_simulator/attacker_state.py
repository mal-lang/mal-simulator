from dataclasses import dataclass, field
from collections.abc import Mapping, Set
from typing import Any
from maltoolbox.attackgraph import AttackGraphNode
from malsim.config.agent_settings import AttackerSettings
from malsim.mal_simulator.agent_state import MalSimAgentState
from malsim.types import AgentStates


@dataclass(frozen=True)
class MalSimAttackerState(MalSimAgentState):
    """Stores the state of an attacker in the simulator"""

    settings: AttackerSettings[AttackGraphNode]
    # Number of attempts to compromise a step (used for ttc caculations)
    num_attempts: Mapping[AttackGraphNode, int]
    # Steps attempted but not succeeded (because of TTC value)
    step_attempted_nodes: Set[AttackGraphNode]
    ttc_values: Mapping[AttackGraphNode, float]
    # Only used if `ttc_overrides` is set
    impossible_steps: Set[AttackGraphNode] = field(
        default_factory=frozenset
    )  # Steps that are impossible to perform

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


def get_attacker_agents(
    agent_states: AgentStates, alive_agents: Set[str], only_alive: bool = False
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
