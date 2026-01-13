from __future__ import annotations
from collections.abc import Set, Mapping
from types import MappingProxyType
from typing import Optional, TYPE_CHECKING

from malsim.mal_simulator.attack_surface import get_attack_surface
from malsim.mal_simulator.defense_surface import get_defense_surface

from malsim.mal_simulator.agent_state import (
    MalSimAttackerState,
    MalSimDefenderState,
)

from malsim.mal_simulator.observability import defender_observed_nodes

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from malsim.mal_simulator.agent_state import (
        AgentStates,
        AgentSettings
    )
    from malsim.mal_simulator.ttc_utils import TTCDist
    from malsim.mal_simulator.simulator_state import MalSimulatorState
    import numpy as np

def create_attacker_state(
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    name: str,
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
        if sim_state.settings.compromise_entrypoints_at_start:
            step_compromised_nodes |= entry_points
        compromised_nodes = step_compromised_nodes

        # Create an initial attack surface
        new_action_surface = get_attack_surface(
            sim_state.settings,
            sim_state,
            agent_settings,
            sim_state.global_actionability,
            name,
            compromised_nodes
        )
        action_surface_removals: set[AttackGraphNode] = set()
        action_surface_additions = new_action_surface

        if not sim_state.settings.compromise_entrypoints_at_start:
            # If entrypoints not compromised at start,
            # we need to put them in action surface
            new_action_surface |= entry_points
            action_surface_additions |= entry_points

        previous_num_attempts: Mapping[AttackGraphNode, int] = {
            n: 0 for n in sim_state.attack_graph.attack_steps
        }

    else:
        ttc_value_overrides = previous_state.ttc_value_overrides
        impossible_step_overrides = previous_state.impossible_step_overrides
        compromised_nodes = previous_state.performed_nodes | step_compromised_nodes

        # Build on previous attack surface (for performance)
        action_surface_additions = (
            get_attack_surface(
                sim_state.settings,
                sim_state,
                agent_settings,
                sim_state.global_actionability,
                name,
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
        sim_state=sim_state,
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

def create_defender_state(
    sim_state: MalSimulatorState,
    agent_settings: AgentSettings,
    name: str,
    rng: np.random.Generator,
    step_compromised_nodes: Set[AttackGraphNode] = frozenset(),
    step_enabled_defenses: Set[AttackGraphNode] = frozenset(),
    step_nodes_made_unviable: Set[AttackGraphNode] = frozenset(),
    previous_state: Optional[MalSimDefenderState] = None,
) -> MalSimDefenderState:
    """
    Update a previous defender state based on what steps
    were enabled/compromised during last step
    """

    action_surface = get_defense_surface(
        agent_settings,
        sim_state,
        sim_state.global_actionability,
        name
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
        action_surface -= previous_state.performed_nodes

    step_observed_nodes = defender_observed_nodes(
        agent_settings, sim_state, rng, name, step_compromised_nodes
    )
    return MalSimDefenderState(
        name,
        sim_state=sim_state,
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
