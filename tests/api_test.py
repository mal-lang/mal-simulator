import logging

from maltoolbox.language import classes_factory
from maltoolbox.language import specification
from maltoolbox.language import languagegraph as mallanguagegraph
from maltoolbox.attackgraph import attackgraph as malattackgraph
from maltoolbox.model import model as malmodel

from pettingzoo.test import parallel_api_test

from malpzsim.sims.mal_petting_zoo_simulator import MalPettingZooSimulator
import gymnasium as gym
from gymnasium.utils import env_checker

from malpzsim.wrappers.gym_wrapper import AttackerEnv, DefenderEnv


def test_pz():
    logger = logging.getLogger(__name__)

    lang_file = "tests/org.mal-lang.coreLang-1.0.0.mar"
    lang_spec = specification.load_language_specification_from_mar(lang_file)
    specification.save_language_specification_to_json(lang_spec, "tests/lang_spec.json")
    lang_classes_factory = classes_factory.LanguageClassesFactory(lang_spec)
    lang_classes_factory.create_classes()

    lang_graph = mallanguagegraph.LanguageGraph()
    lang_graph.generate_graph(lang_spec)

    model = malmodel.Model("Test Model", lang_spec, lang_classes_factory)
    model.load_from_file("tests/example_model.json")

    attack_graph = malattackgraph.AttackGraph()
    attack_graph.generate_graph(lang_spec, model)
    attack_graph.attach_attackers(model)
    attack_graph.save_to_file("tmp/attack_graph.json")

    env = MalPettingZooSimulator(lang_graph, model, attack_graph, max_iter=5)

    env.register_attacker("attacker", 0)
    env.register_defender("defender")

    logger.debug("Run Parrallel API test.")
    parallel_api_test(env, num_cycles=50)

    env.close()


def test_gym():
    gym.register("MALDefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "MALDefenderEnv-v0",
        model_file="tests/example_model.json",
        lang_file="tests/org.mal-lang.coreLang-1.0.0.mar",
        unholy=True,
    )

    env_checker.check_env(env.unwrapped)

    gym.register("MALAttackerEnv-v0", entry_point=AttackerEnv)
    env = gym.make(
        "MALAttackerEnv-v0",
        model_file="tests/example_model.json",
        lang_file="tests/org.mal-lang.coreLang-1.0.0.mar",
    )

    env_checker.check_env(env.unwrapped)


def test_step():
    gym.register("MALDefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make(
        "MALDefenderEnv-v0",
        model_file="tests/example_model.json",
        lang_file="tests/org.mal-lang.coreLang-1.0.0.mar",
        unholy=True,
    )

    done = False
    obs, info = env.reset()
    step = 0
    _return = 0.0
    while not done:
        obs, reward, term, trunc, info = env.step((0, None))
        done = term or trunc
        print(obs, reward, done, info)
        step += 1
        _return += reward

    assert done
    assert _return < 0.0
