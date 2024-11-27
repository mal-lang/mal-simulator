"""CLI to run simulations in MAL Simulator using scenario files"""

from __future__ import annotations
import argparse
import logging

from .sims.base_mal_simulator import BaseMalSimulator
from .sims.mal_simulator import MalSimulator
from .scenario import load_scenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def run_simulation(sim: BaseMalSimulator, conf: dict):
    """Run a simulation on an attack graph with given config"""

    # Constants
    attacker_agent_id = None
    defender_agent_id = None

    for agent_id, agent_info in conf['agents'].items():
        if agent_info['type'] == "attacker":
            sim.register_attacker(agent_id, 0)
            attacker_agent_id = agent_id
        elif agent_info['type'] == "defender":
            sim.register_defender(agent_id)
            defender_agent_id = agent_id

    # Initialize defender and attacker according to classes
    defender_class = conf['agents'][defender_agent_id]['agent_class']
    defender_agent = defender_class()

    attacker_class = conf['agents'][attacker_agent_id]['agent_class']
    attacker_agent = attacker_class()

    agents = sim.reset()
    total_rewards = {agent.name: 0 for agent in agents.values()}
    done = False

    logger.info("Starting simulation.")

    while not done:

        actions = {}

        defender_action = defender_agent.compute_next_action(
            agents[defender_agent_id].action_surface
        )

        attacker_action = attacker_agent.compute_next_action(
            agents[attacker_agent_id].action_surface
        )

        logger.info(
            "Defender Actions: %s", [n.full_name for n in defender_action])

        logger.info(
            "Attacker Actions: %s", [n.full_name for n in attacker_action])
        if not attacker_action:
            # Stop the simulation if attacker run out of things to do
            done = True

        action_dict = {
            attacker_agent_id: attacker_action,
            defender_agent_id: defender_action
        }

        # Perform next step of simulation
        agents = sim.step(action_dict)

        for agent in agents.values():
            total_rewards[agent.name] += agent.reward

        print("---\n")

    logger.info("Game Over.")

    for agent in agents.values():
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
    attack_graph, conf = load_scenario(args.scenario_file)
    if args.output_attack_graph:
        attack_graph.save_to_file(args.output_attack_graph)

    sim = BaseMalSimulator(attack_graph)
    run_simulation(sim, conf)


if __name__ == '__main__':
    main()
