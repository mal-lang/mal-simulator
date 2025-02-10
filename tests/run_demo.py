import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from json import JSONEncoder
import numpy as np
import logging

from maltoolbox.language import LanguageClassesFactory, LanguageGraph
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.model import Model

from malsim.agents.keyboard_input import KeyboardAgent
from malsim.agents.searchers import BreadthFirstAttacker
from malsim.sims.mal_simulator import MalSimulator


logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("maltoolbox").setLevel(logging.DEBUG)


# Raise the logging level for the py2neo module to clean the logs a bit
# cleaner.
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


#MAL toolbox to load the graph attack
lang_file = "tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar"
lang_graph = LanguageGraph.from_mar_archive(lang_file)
lang_classes_factory = LanguageClassesFactory(lang_graph)

model = Model.load_from_file("tests/testdata/models/run_demo_model.json", lang_classes_factory)

attack_graph = AttackGraph(lang_graph, model)
attack_graph.attach_attackers()
attack_graph.save_to_file("logs/attack_graph.json")

env = MalSimulator(lang_graph, model, attack_graph, max_iter=500)

env.register_attacker(AGENT_ATTACKER, 0)
env.register_defender(AGENT_DEFENDER)

control_attacker = False

reverse_vocab = env._index_to_full_name

defender = KeyboardAgent(reverse_vocab)
attacker = (
    KeyboardAgent(reverse_vocab) if control_attacker else BreadthFirstAttacker({})
)

obs, infos = env.reset()
done = False

# Set rewards
# TODO Have a nice and configurable way of doing this when we have the
# scenario configuration format decided upon.
MAX_REWARD = int(1e9)

env.attack_graph.get_node_by_full_name("OS App:notPresent").extras['reward'] = 50
env.attack_graph.get_node_by_full_name("OS App:supplyChainAuditing").extras['reward'] = MAX_REWARD
env.attack_graph.get_node_by_full_name("Program 1:notPresent").extras['reward'] = 30
env.attack_graph.get_node_by_full_name("Program 1:supplyChainAuditing").extras['reward'] = MAX_REWARD
env.attack_graph.get_node_by_full_name("SoftwareVulnerability:2:notPresent").extras['reward'] = 40
env.attack_graph.get_node_by_full_name("Data:3:notPresent").extras['reward'] = 20
env.attack_graph.get_node_by_full_name("Credentials:4:notPhishable").extras['reward'] = MAX_REWARD
env.attack_graph.get_node_by_full_name("Identity:5:notPresent").extras['reward'] = 35
env.attack_graph.get_node_by_full_name("ConnectionRule:6:restricted").extras['reward'] = 40
env.attack_graph.get_node_by_full_name("ConnectionRule:6:payloadInspection").extras['reward'] = 30
env.attack_graph.get_node_by_full_name("Other OS App:notPresent").extras['reward'] = 50
env.attack_graph.get_node_by_full_name("Other OS App:supplyChainAuditing").extras['reward'] = MAX_REWARD

env.attack_graph.get_node_by_full_name("OS App:fullAccess").extras['reward'] = 100
env.attack_graph.get_node_by_full_name("Program 1:fullAccess").extras['reward'] = 50
env.attack_graph.get_node_by_full_name("Identity:5:assume").extras['reward'] = 50
env.attack_graph.get_node_by_full_name("Other OS App:fullAccess").extras['reward'] = 200


logger.info("Starting game.")

total_reward_defender = 0
total_reward_attacker = 0

while not done:
    # env.render()
    defender_action = (
        defender.compute_action_from_dict(
            obs[AGENT_DEFENDER], infos[AGENT_DEFENDER]["action_mask"]
        )
        if not attacker_only
        else null_action
    )
    attacker_action = attacker.compute_action_from_dict(
        obs[AGENT_ATTACKER], infos[AGENT_ATTACKER]["action_mask"]
    )

    if attacker_action[1] is not None:
        print("Attacker Action: ", reverse_vocab[attacker_action[1]])
        logger.debug(f"Attacker Action: {reverse_vocab[attacker_action[1]]}")
    else:
        print("Attacker Action: None")
        logger.debug("Attacker Action: None")
        # Stop the attacker if it has run out of things to do since the
        # experiment cannot progress any further.
        # TODO Perhaps we want to only do this if none of the agents have
        # anything to do or we may simply wish to have them running to accrue
        # penalties/rewards. This was added just to make it easier to debug.
        done = True
    action_dict = {AGENT_ATTACKER: attacker_action, AGENT_DEFENDER: defender_action}
    obs, rewards, terminated, truncated, infos = env.step(action_dict)

    logger.debug("Attacker has compromised the following attack steps so " "far:")
    attacker_obj = env.attack_graph.attackers[
        env.agents_dict[AGENT_ATTACKER]["attacker"]
    ]
    for step in attacker_obj.reached_attack_steps:
        logger.debug(step.id)

    print("Attacker Reward: ", rewards[AGENT_ATTACKER])
    logger.debug(f"Attacker Reward: {rewards[AGENT_ATTACKER]}")
    if not attacker_only:
        print("Defender Reward: ", rewards[AGENT_DEFENDER])
        logger.debug(f"Defender Reward: {rewards[AGENT_DEFENDER]}")
    total_reward_defender += rewards[AGENT_DEFENDER] if not attacker_only else 0
    total_reward_attacker += rewards[AGENT_ATTACKER]

    done |= terminated[AGENT_ATTACKER] or truncated[AGENT_ATTACKER]

    print("---\n")

# env.render()
print("Game Over.")
logger.debug("Game Over.")
if not attacker_only:
    print("Total Defender Reward: ", total_reward_defender)
    logger.debug(f"Total Defender Reward: {total_reward_defender}")
print("Total Attacker Reward: ", total_reward_attacker)
logger.debug(f"Total Attacker Reward: {total_reward_attacker}")
print("Press Enter to exit.")
input()
env.close()
