"""CLI to run simulations in MAL Simulator using scenario files"""

from __future__ import annotations
import argparse
import logging
from typing import TYPE_CHECKING

from .sims.mal_simulator import MalSimulator
from .agents.keyboard_input import KeyboardAgent
from .scenario import load_scenario

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def run_simulation(attack_graph: AttackGraph, sim_config: dict):
    """Run a simulation on an attack graph with given config"""

    attack_graph.save_to_file("tmp/attack_graph.json")

    # Constants
    NULL_ACTION = (0, None)
    AGENT_ATTACKER = 'attacker1'
    AGENT_DEFENDER = 'defender1'

    env = MalSimulator(
        attack_graph.lang_graph,
        attack_graph.model,
        attack_graph,
        max_iter=500
    )

    # Register attacker and defender
    env.register_attacker(AGENT_ATTACKER, 0)
    env.register_defender(AGENT_DEFENDER)

    # Initialize defender and attacker according to classes
    reverse_vocab = env._index_to_full_name
    defender_class = sim_config['defender_agent_class']
    defender_agent = defender_class(reverse_vocab) if defender_class else None
    attacker_class = sim_config['attacker_agent_class']
    attacker_agent = (attacker_class(reverse_vocab)
                if attacker_class == KeyboardAgent
                else attacker_class({}))

    obs, infos = env.reset()
    done = False

    logger.info("Starting game.")

    total_reward_defender = 0
    total_reward_attacker = 0

    while not done:

        defender_action = NULL_ACTION
        if defender_agent:
            defender_action = defender_agent.compute_action_from_dict(
                obs[AGENT_DEFENDER],
                infos[AGENT_DEFENDER]["action_mask"]
            )

        attacker_action = attacker_agent.compute_action_from_dict(
            obs[AGENT_ATTACKER],
            infos[AGENT_ATTACKER]["action_mask"]
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
            AGENT_ATTACKER: attacker_action,
            AGENT_DEFENDER: defender_action
        }

        # Perform next step of simulation
        obs, rewards, terminated, truncated, infos = env.step(action_dict)

        logger.debug(
            "Attacker has compromised the following attack steps so far:"
        )
        attacker_obj = env.attack_graph.attackers[
            env.agents_dict[AGENT_ATTACKER]["attacker"]
        ]
        for step in attacker_obj.reached_attack_steps:
            logger.debug(step.id)

        logger.info("Attacker Reward: %s", rewards[AGENT_ATTACKER])

        if defender_agent:
            logger.info("Defender Reward: %s", rewards.get(AGENT_DEFENDER))

        total_reward_defender += rewards.get(AGENT_DEFENDER, 0) if defender_agent else 0
        total_reward_attacker += rewards.get(AGENT_ATTACKER, 0)

        done |= terminated[AGENT_ATTACKER] or truncated[AGENT_ATTACKER]

        print("---\n")

    logger.info("Game Over.")

    if defender_agent:
        logger.info("Total Defender Reward: %s", total_reward_defender)
    logger.info("Total Attacker Reward: %s", total_reward_attacker)

    print("Press Enter to exit.")
    input()
    env.close()


def main():
    """Entrypoint function of the MAL Toolbox CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'scenario_file',
        type=str,
        help="Can be found in https://github.com/mal-lang/malsim-scenarios/"
    )
    args = parser.parse_args()

    # Load AttackGraph and config from scenario file and run simulation
    attack_graph, sim_config = load_scenario(args.scenario_file)
    run_simulation(attack_graph, sim_config)


if __name__ == '__main__':
    main()
