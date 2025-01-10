"""CLI to run simulations in MAL Simulator using scenario files"""

from __future__ import annotations
import argparse
import logging

from malsim.sims import MalSimVectorizedObsEnv
from malsim.scenario import create_simulator_from_scenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

def run_simulation(sim: MalSimVectorizedObsEnv, agents: list[dict]):
    """Run a simulation with agents"""

    obs, infos = sim.reset()
    total_rewards = {agent_dict['name']: 0 for agent_dict in agents}
    all_agents_term_or_trunc = False

    logger.info("Starting CLI env simulator.")

    while not all_agents_term_or_trunc:
        all_agents_term_or_trunc = True
        actions = {}

        # Select actions for each agent
        for agent_dict in agents:
            agent = agent_dict.get('agent')
            agent_name = agent_dict['name']
            if agent is None:
                logger.warning(
                    'Agent "%s" has no decision agent class '
                    'specified in scenario. Waiting.', agent_name,
                )
                continue

            agent_action = \
                agent.get_next_action(sim.get_agent(agent_name))
            actions[agent_name] = agent_action

            if agent_action[0]:
                logger.info(
                    'Agent "%s" chose action: %s', agent_name,
                    [sim.index_to_node(agent_action[1]).full_name]
                )

        # Perform next step of simulation
        obs, rew, term, trunc, infos = sim.step(actions)

        for agent_dict in agents:
            agent_name = agent_dict['name']
            total_rewards[agent_name] += rew[agent_name]
            if not term[agent_name] and not trunc[agent_name]:
                all_agents_term_or_trunc = False

        print("---\n")

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

    sim, agents = \
        create_simulator_from_scenario(
            args.scenario_file,
            sim_class=MalSimVectorizedObsEnv
        )

    if args.output_attack_graph:
        sim.attack_graph.save_to_file(args.output_attack_graph)

    run_simulation(sim, agents)


if __name__ == '__main__':
    main()
