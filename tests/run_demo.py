import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from json import JSONEncoder
import numpy as np
import logging

from maltoolbox.language import LanguageClassesFactory, LanguageGraph, specification
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.model import Model

from malpzsim.agents.keyboard_input import KeyboardAgent
from malpzsim.agents.searchers import BreadthFirstAttacker
from malpzsim.sims.mal_petting_zoo_simulator import MalPettingZooSimulator


logger = logging.getLogger(__name__)

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
lang_file = "org.mal-lang.coreLang-1.0.0.mar"
lang_spec = specification.load_language_specification_from_mar(lang_file)
specification.save_language_specification_to_json(lang_spec, "lang_spec.json")
lang_classes_factory = LanguageClassesFactory(lang_spec)
lang_classes_factory.create_classes()

lang_graph = LanguageGraph(lang_spec)

model = Model("Test Model", lang_spec, lang_classes_factory)
model.load_from_file("example_model.json")

attack_graph = AttackGraph(lang_spec, model)
attack_graph.attach_attackers(model)
attack_graph.save_to_file("tmp/attack_graph.json")

env = MalPettingZooSimulator(lang_graph, model, attack_graph, max_iter=500)


env.register_attacker(AGENT_ATTACKER, 0)
env.register_defender(AGENT_DEFENDER)

control_attacker = False

reverse_vocab = env._index_to_id

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
env.attack_graph.get_node_by_id("OS App:0:notPresent").reward = 50
env.attack_graph.get_node_by_id("OS App:0:successfulReverseReach").reward = MAX_REWARD
env.attack_graph.get_node_by_id("OS App:0:networkConnectInspected").reward = 30
env.attack_graph.get_node_by_id("OS App:0:successfulUnsafeUserActivity").reward = MAX_REWARD
env.attack_graph.get_node_by_id("SoftwareVulnerability:2:2:lowPrivilegesRequired").reward = 40
env.attack_graph.get_node_by_id("Data:3:3:applicationRespondConnect").reward = 20
env.attack_graph.get_node_by_id("Credentials:4:4:attemptDenyFromReplica").reward = MAX_REWARD
env.attack_graph.get_node_by_id("Identity:5:5:attemptAssume").reward = 35
env.attack_graph.get_node_by_id("Credentials:4:4:notDisclosed").reward = 40
env.attack_graph.get_node_by_id("ConnectionRule:6:6:successfulAccessNetworksUninspected").reward = 30
env.attack_graph.get_node_by_id("OS App:0:successfulRead").reward = 50
env.attack_graph.get_node_by_id("OS App:0:specificAccessNetworkConnect").reward = MAX_REWARD

env.attack_graph.get_node_by_id("OS App:0:specificAccessFromNetworkConnection").reward = 100
env.attack_graph.get_node_by_id("OS App:0:fullAccess").reward = 50
env.attack_graph.get_node_by_id("Identity:5:5:assume").reward = 50
env.attack_graph.get_node_by_id("OS App:0:attemptReadFromSoftProdVulnerability").reward = 200


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

    print(infos[AGENT_ATTACKER])
    if attacker_action[1] is not None:
        print("Attacker Action: ", reverse_vocab[attacker_action[1]])
        logger.debug(f"Attacker Action: {reverse_vocab[attacker_action[1]]}")
    else:
        print("Attacker Action: None")
        logger.debug("Attacker Action: None")
    action_dict = {AGENT_ATTACKER: attacker_action, AGENT_DEFENDER: defender_action}



    obs, rewards, terminated, truncated, infos = env.step(action_dict)

    logger.debug("Attacker has compromised the following attack steps so far:")
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
