"""CLI to run simulations in MAL Simulator using scenario files"""

from __future__ import annotations
import argparse
import logging

from .mal_simulator import MalSimulator
from .agents import DecisionAgent
from .scenario import create_simulator_from_scenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

def run_simulation(sim: MalSimulator, agents: list[dict]):
    """Run a simulation with agents"""

    sim.reset()
    total_rewards = {agent_dict['name']: 0 for agent_dict in agents}
    all_agents_term_or_trunc = False

    logger.info("Starting CLI env simulator.")

    i = 1
    while not all_agents_term_or_trunc:
        logger.info("Iteration %s", i)
        all_agents_term_or_trunc = True
        actions = {}

        # Select actions for each agent
        for agent_dict in agents:
            decision_agent: DecisionAgent = agent_dict.get('agent')
            agent_name = agent_dict['name']
            if decision_agent is None:
                logger.warning(
                    'Agent "%s" has no decision agent class '
                    'specified in scenario. Waiting.', agent_name,
                )
                continue

            sim_agent_state = sim.agent_states[agent_name]
            agent_action = decision_agent.get_next_action(sim_agent_state)
            if agent_action:
                actions[agent_name] = [agent_action]
                logger.info(
                    'Agent "%s" chose action: %s',
                    agent_name, agent_action.full_name
                )

        # Perform next step of simulation
        sim.step(actions)

        for agent_dict in agents:
            agent_name = agent_dict['name']
            agent_state = sim.agent_states[agent_name]
            total_rewards[agent_name] += agent_state.reward
            if not agent_state.terminated and not agent_state.truncated:
                all_agents_term_or_trunc = False
        print("---\n")
        i += 1

    logger.info("Game Over.")

    # Print total rewards
    for agent_dict in agents:
        agent_name = agent_dict['name']
        print(f'Total reward "{agent_name}"', total_rewards[agent_name])

def main():
    """Entrypoint function of the MAL Toolbox CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'scenario_file',
        type=str,
        help="Can be found in https://github.com/mal-lang/malsim-scenarios/"
    )
    parser.add_argument(
        '-o', '--output-attack-graph', type=str,
        help="If set to a path, attack graph will be dumped there",
    )
    args = parser.parse_args()

    sim, agents = create_simulator_from_scenario(args.scenario_file)

    if args.output_attack_graph:
        sim.attack_graph.save_to_file(args.output_attack_graph)

    run_simulation(sim, agents)


if __name__ == '__main__':
    main()
