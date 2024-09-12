"""CLI to run simulations in MAL Simulator using scenario files"""

from __future__ import annotations
import argparse
import logging

from .sims.mal_simulator import MalSimulator
from .agents.keyboard_input import KeyboardAgent
from .scenario import create_simulator_from_scenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def run_simulation(sim: MalSimulator, sim_config: dict):
    """Run a simulation on an attack graph with given config"""

    # Constants
    NULL_ACTION = (0, None)

    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    defender_agent_id = next(iter(sim.get_defender_agents()), None)

    reverse_vocab = sim._index_to_full_name

    # Initialize defender and attacker according to classes
    defender_class = sim_config['agents'][defender_agent_id]['agent_class']\
                     if defender_agent_id else None
    defender_agent = (defender_class(reverse_vocab)
                          if defender_class == KeyboardAgent
                          else defender_class({})
                      if defender_class
                      else None)

    attacker_class = sim_config['agents'][attacker_agent_id]['agent_class']
    attacker_agent = (attacker_class(reverse_vocab)
                      if attacker_class == KeyboardAgent
                      else attacker_class({}))

    obs, infos = sim.reset()
    done = False

    logger.info("Starting game.")

    total_reward_defender = 0
    total_reward_attacker = 0

    while not done:

        defender_action = NULL_ACTION
        if defender_agent:
            defender_action = defender_agent.compute_action_from_dict(
                obs[defender_agent_id],
                infos[defender_agent_id]["action_mask"]
            )

        attacker_action = attacker_agent.compute_action_from_dict(
            obs[attacker_agent_id],
            infos[attacker_agent_id]["action_mask"]
        )

        if attacker_action[1] is not None:
            logger.info(
                "Attacker Action: %s", reverse_vocab[attacker_action[1]])
        else:
            logger.info("Attacker Action: None")
            # Stop the attacker if it has run out of things to do since
            # the experiment cannot progress any further.
            done = True

        action_dict = {
            attacker_agent_id: attacker_action,
            defender_agent_id: defender_action
        }

        # Perform next step of simulation
        obs, rewards, terminated, truncated, infos = sim.step(action_dict)

        logger.debug(
            "Attacker has compromised the following attack steps so far:"
        )
        attacker_obj = sim.attack_graph.attackers[
            sim.agents_dict[attacker_agent_id]["attacker"]
        ]
        for step in attacker_obj.reached_attack_steps:
            logger.debug(step.id)

        logger.info("Attacker Reward: %s", rewards.get(attacker_agent_id))

        if defender_agent:
            logger.info("Defender Reward: %s", rewards.get(defender_agent_id))

        total_reward_defender += rewards.get(defender_agent_id, 0) if defender_agent else 0
        total_reward_attacker += rewards.get(attacker_agent_id, 0)

        done |= terminated.get(attacker_agent_id, True) or truncated.get(attacker_agent_id, True)

        print("---\n")

    logger.info("Game Over.")

    if defender_agent:
        logger.info("Total Defender Reward: %s", total_reward_defender)
    logger.info("Total Attacker Reward: %s", total_reward_attacker)

    print("Press Enter to exit.")
    input()
    sim.close()


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
    simulator, sim_config = create_simulator_from_scenario(args.scenario_file)
    if args.output_attack_graph:
        simulator.attack_graph.save_to_file(args.output_attack_graph)
    run_simulation(simulator, sim_config)


if __name__ == '__main__':
    main()
