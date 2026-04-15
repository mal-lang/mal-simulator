from __future__ import annotations
from collections.abc import Mapping, Set
from dataclasses import dataclass

from maltoolbox.attackgraph import AttackGraphNode
from malsim.mal_simulator.simulator_state import MalSimulatorState


@dataclass(frozen=True)
class AgentState:
    """Stores the state of an agent in the simulator"""

    # Identifier of the agent, used in MalSimulator for lookup
    name: str
    # Reference to the simulator
    sim_state: MalSimulatorState
    # Contains possible acWions for the agent in the next step
    action_surface: Set[AttackGraphNode]
    # Contains all nodes that this agent has performed successfully
    performed_nodes: Set[AttackGraphNode]
    # Contains the order of performed nodes
    performed_nodes_order: Mapping[int, Set[AttackGraphNode]]

    unviable_nodes: Set[AttackGraphNode]

    # The iteration this state was created in
    iteration: int
