from malsim.mal_simulator.attacker_state import MalSimAttackerState
from malsim.mal_simulator.defender_state import MalSimDefenderState


AgentStates = dict[str, MalSimAttackerState | MalSimDefenderState]


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