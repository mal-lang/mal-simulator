from venv import logger
from maltoolbox.attackgraph import AttackGraphNode

from malsim.policies.decision_agent import DecisionAgent
from malsim.mal_simulator.simulator import MalSimulator


def run_simulation(sim: MalSimulator) -> dict[str, list[AttackGraphNode]]:
    """Run a simulation with agents

    Return selected actions by each agent in each step
    """
    agent_actions: dict[str, list[AttackGraphNode]] = {}
    total_rewards = dict.fromkeys(sim.agent_settings, 0.0)

    logger.info('Starting CLI env simulator.')
    states = sim.reset()
    iteration = 0
    while not sim.done():
        print(f'Iteration {iteration}')
        actions: dict[str, list[AttackGraphNode]] = {}

        # Select actions for each agent
        for agent_name, agent_config in sim.agent_settings.items():
            decision_agent: DecisionAgent | None = agent_config.agent
            if decision_agent is None:
                print(
                    f'Agent "{agent_name}" has no decision agent class '
                    'specified in scenario. Waiting.'
                )
                continue

            agent_state = states[agent_name]
            agent_action = decision_agent.get_next_action(agent_state)

            if agent_action:
                actions[agent_name] = [agent_action]

                # Store agent action
                agent_actions.setdefault(agent_name, []).append(agent_action)

        # Perform next step of simulation
        states = sim.step(actions)
        for agent_name in agents:
            total_rewards[agent_name] += sim.agent_reward(states[agent_name])
        iteration += 1
        print('---')

    print(f'Simulation over after {iteration} steps.')

    # Print total rewards
    for agent_name in sim.agent_settings:
        print(f'Total reward "{agent_name}"', total_rewards[agent_name])

    return agent_actions
