from malsim.mal_simulator.attacker_state import AttackerState
from malsim.mal_simulator.defender_state import DefenderState

AgentStates = dict[str, AttackerState | DefenderState]


def defender_states(agent_states: AgentStates) -> dict[str, DefenderState]:
    return {
        name: state
        for name, state in agent_states.items()
        if isinstance(state, DefenderState)
    }


def attacker_states(agent_states: AgentStates) -> dict[str, AttackerState]:
    return {
        name: state
        for name, state in agent_states.items()
        if isinstance(state, AttackerState)
    }
