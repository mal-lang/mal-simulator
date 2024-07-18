import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from json import JSONEncoder
import numpy as np
import logging

from maltoolbox.language import LanguageClassesFactory, LanguageGraph
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.model import Model

from malsim.scenario import load_scenario
from malsim.agents.keyboard_input import KeyboardAgent
from malsim.agents.searchers import BreadthFirstAttacker
from malsim.sims.mal_simulator import MalSimulator


logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("maltoolbox").setLevel(logging.DEBUG)


# Raise the logging level for the py2neo module to clean the logs a bit
py2neo_logger = logging.getLogger("py2neo")
py2neo_logger.setLevel(logging.INFO)

null_action = (0, None)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.int64):
            return int(o)
        return JSONEncoder.default(self, o)


attacker_only = False

AGENT_ATTACKER = "attacker"
AGENT_DEFENDER = "defender"

def run_simulation(scenario_file):
    """Run a simulation for scenario"""
    # Load the attack graph from scenario
    attack_graph = load_scenario(scenario_file)
    attack_graph.save_to_file("tmp/attack_graph.json")

    # TODO: why send langgraph and model in separate args?
    env = MalSimulator( 
        attack_graph.lang_graph,
        attack_graph.model,
        attack_graph,
        max_iter=500
    )

    env.register_attacker(AGENT_ATTACKER, 0)
    env.register_defender(AGENT_DEFENDER)

    control_attacker = False
    reverse_vocab = env._index_to_full_name

    defender = KeyboardAgent(reverse_vocab)
    attacker = (
        KeyboardAgent(reverse_vocab) if control_attacker
        else BreadthFirstAttacker({})
    )

    obs, infos = env.reset()
    done = False

    logger.info("Starting game.")

    total_reward_defender = 0
    total_reward_attacker = 0

    while not done:
        defender_action = (
            defender.compute_action_from_dict(
                obs[AGENT_DEFENDER],
                infos[AGENT_DEFENDER]["action_mask"]
            )
            if not attacker_only
            else null_action
        )
        attacker_action = attacker.compute_action_from_dict(
            obs[AGENT_ATTACKER],
            infos[AGENT_ATTACKER]["action_mask"]
        )

        if attacker_action[1] is not None:
            print("Attacker Action: ", reverse_vocab[attacker_action[1]])
            logger.debug(
                "Attacker Action: %s", reverse_vocab[attacker_action[1]])
        else:
            print("Attacker Action: None")
            logger.debug("Attacker Action: None")
            # Stop the attacker if it has run out of things to do since the
            # experiment cannot progress any further.
            # TODO Perhaps we want to only do this if none of the agents have
            # anything to do or we may simply wish to have them running to accrue
            # penalties/rewards. This was added just to make it easier to debug.
            done = True
        action_dict = {
            AGENT_ATTACKER: attacker_action,
            AGENT_DEFENDER: defender_action
        }
        obs, rewards, terminated, truncated, infos = env.step(action_dict)

        logger.debug("Attacker has compromised the following attack steps so far:")
        attacker_obj = env.attack_graph.attackers[
            env.agents_dict[AGENT_ATTACKER]["attacker"]
        ]
        for step in attacker_obj.reached_attack_steps:
            logger.debug(step.id)

        print("Attacker Reward: ", rewards[AGENT_ATTACKER])
        logger.debug("Attacker Reward: %s", rewards[AGENT_ATTACKER])
        if not attacker_only:
            print("Defender Reward: ", rewards[AGENT_DEFENDER])
            logger.debug("Defender Reward: %s", rewards[AGENT_DEFENDER])
        total_reward_defender += rewards[AGENT_DEFENDER] if not attacker_only else 0
        total_reward_attacker += rewards[AGENT_ATTACKER]

        done |= terminated[AGENT_ATTACKER] or truncated[AGENT_ATTACKER]

        print("---\n")

    # env.render()
    print("Game Over.")
    logger.debug("Game Over.")
    if not attacker_only:
        print("Total Defender Reward: ", total_reward_defender)
        logger.debug("Total Defender Reward: %s", total_reward_defender)
    print("Total Attacker Reward: ", total_reward_attacker)
    logger.debug("Total Attacker Reward: %s", total_reward_attacker)
    print("Press Enter to exit.")
    input()
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'scenario_file',
        type=str,
        help="Can be found in https://github.com/mal-lang/malsim-scenarios/"
    )
    args = parser.parse_args()
    run_simulation(args.scenario_file)
