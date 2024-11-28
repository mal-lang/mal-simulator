"""CLI to run simulations in MAL Simulator using scenario files"""

from __future__ import annotations
import argparse
import logging

from .sims.base_mal_simulator import BaseMalSimulator, AgentType
from .sims.mal_simulator import MalSimulator
from .scenario import create_simulator_from_scenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def run_simulation(sim: BaseMalSimulator, agents: dict):
    """Run a simulation on an attack graph with given config"""

    # Init values
    agents_info = sim.reset()
    total_rewards = {agent_id: 0 for agent_id in agents}
    all_agents_done = False

    logger.info("Starting simulation.")

    while not all_agents_done:
        actions = {}

        # Select actions for each agent
        all_agents_done = True
        for agent_id, agent_info in agents.items():
            agent = agent_info['agent']
            agent_action = agent.compute_next_action(
                agents_info[agent_id].action_surface
            )
            actions[agent_id] = agent_action
            logger.info(
                'Agent "%s" chose actions: %s', agent_id,
                [n.full_name for n in agent_action]
            )
            if agent_action:
                all_agents_done = False

        # Perform next step of simulation
        agents_info = sim.step(actions)
        for agent in agents_info.values():
            total_rewards[agent.name] += agent.reward

        print("---\n")
    logger.info("Game Over.")

    # Print total rewards
    for agent in agents_info.values():
        print(f'Total reward "{agent.name}"', total_rewards[agent.name])


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

    # Create simulator from scenario
    sim, agents = create_simulator_from_scenario(args.scenario_file)
    if args.output_attack_graph:
        sim.attack_graph.save_to_file(args.output_attack_graph)

    run_simulation(sim, agents)


if __name__ == '__main__':
    main()
