from malsim.mal_simulator.attacker_state import MalSimAttackerState
from malsim.mal_simulator.defender_state import MalSimDefenderState

AgentStates = dict[str, MalSimAttackerState | MalSimDefenderState]


def defender_states(agent_states: AgentStates) -> dict[str, MalSimDefenderState]:
    return {
        name: state
        for name, state in agent_states.items()
        if isinstance(state, MalSimDefenderState)
    }


def attacker_states(agent_states: AgentStates) -> dict[str, MalSimAttackerState]:
    return {
        name: state
        for name, state in agent_states.items()
        if isinstance(state, MalSimAttackerState)
    }
