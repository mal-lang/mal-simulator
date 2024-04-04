import logging
import numpy as np
import gymnasium as gym
from gymnasium.utils import env_checker
from pettingzoo.test import parallel_api_test

from maltoolbox.language import classes_factory
from maltoolbox.language import specification
from maltoolbox.language import languagegraph as mallanguagegraph
from maltoolbox.attackgraph import attackgraph as malattackgraph
from maltoolbox import model as malmodel

from malsim.sims.mal_simulator import MalSimulator
from malsim.wrappers.gym_wrapper import AttackerEnv, DefenderEnv

logger = logging.getLogger(__name__)

def test_pz():

    lang_file = "tests/org.mal-lang.coreLang-1.0.0.mar"
    lang_spec = specification.load_language_specification_from_mar(lang_file)
    specification.save_language_specification_to_json(lang_spec, "tests/lang_spec.json")
    lang_classes_factory = classes_factory.LanguageClassesFactory(lang_spec)
    lang_classes_factory.create_classes()

    lang_graph = mallanguagegraph.LanguageGraph(lang_spec)

    model = malmodel.Model("Test Model", lang_spec, lang_classes_factory)
    model.load_from_file("tests/example_model.json")

    attack_graph = malattackgraph.AttackGraph()
    attack_graph.generate_graph(lang_spec, model)
    attack_graph.attach_attackers()
    attack_graph.save_to_file("tmp/attack_graph.json")

    env = MalSimulator(lang_graph, model, attack_graph, max_iter=5)

    env.register_attacker("attacker", 0)
    env.register_defender("defender")

    logger.debug("Run Parrallel API test.")
    parallel_api_test(env, num_cycles=50)

    env.close()


def test_gym():
    logger.debug("Run Gym Test.")
    gym.register("MALDefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "MALDefenderEnv-v0",
        model_file="tests/demo1_model.json",
        attack_graph_file="tests/demo1_attack_graph.json",
        lang_file="tests/org.mal-lang.coreLang-1.0.0.mar",
        unholy=True,
    )

    env_checker.check_env(env.unwrapped)

    gym.register("MALAttackerEnv-v0", entry_point=AttackerEnv)
    env = gym.make(
        "MALAttackerEnv-v0",
        model_file="tests/demo1_model.json",
        attack_graph_file="tests/demo1_attack_graph.json",
        lang_file="tests/org.mal-lang.coreLang-1.0.0.mar",
    )

    env_checker.check_env(env.unwrapped)

def test_random_defender_actions():
    gym.register("MALDefenderEnv-v0", entry_point=DefenderEnv)
    
    env = gym.make(
        "MALDefenderEnv-v0",
        model_file="2024_04_05_16_16_generated_model.json",
        attack_graph_file="2024_04_05_16_16_generated_attack_graph.json",
        lang_file="org.mal-lang.coreLang-1.0.0.mar",
    )
    
    def available_steps(x):
        np.flatnonzero(x["hacked_action_mask"][1])

    def available_actions(x):
        np.flatnonzero(x["hacked_action_mask"][0])

    done = False
    _, info = env.reset()
    while not done:
        available_s = available_steps(info)
        defense = np.random.choice(1, available_s)
        available_a = available_actions(info)
        action = np.random.choice(1, available_a)

        _, _, term, trunc, info = env.step((action, defense))
        done = term or trunc


def test_episode():
    logger.debug("Run Episode Test.")
    gym.register("MALDefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "MALDefenderEnv-v0",
        model_file="tests/demo1_model.json",
        attack_graph_file="tests/demo1_attack_graph.json",
        lang_file="tests/org.mal-lang.coreLang-1.0.0.mar",
        unholy=False,
    )

    done = False
    obs, info = env.reset()
    step = 0
    _return = 0.0
    while not done:
        obs, reward, term, trunc, info = env.step((0, None))
        done = term or trunc
        logger.debug(f"Step {step}:{obs}, {reward}, {done}, {info}")
        step += 1
        _return += reward

    assert done
    assert _return < 0.0 # If the defender does nothing then it will get a penalty for being attacked

def test_defender_penalty():
    gym.register("MALDefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "MALDefenderEnv-v0",
        model_file="tests/example_model.json",
        lang_file="tests/org.mal-lang.coreLang-1.0.0.mar",
        unholy=True,
    )


    _, info = env.reset()

    possible_defense_steps = np.flatnonzero(info['action_mask'][1])
    step = np.random.choice(possible_defense_steps)
    _, reward, _, _, info = env.step((1, step))
    assert reward < 0 # All defense steps cost something


test_random_defender_actions()
