from os import path
import pytest

from maltoolbox.language import LanguageGraph, LanguageClassesFactory
from maltoolbox.model import Model
from maltoolbox.wrappers import create_attack_graph
from malsim.sims import MalSimVectorizedObsEnv, MalSimulator

model_file_name='tests/testdata/models/simple_test_model.yml'
attack_graph_file_name=path.join('tmp','attack_graph.json')
lang_file_name='tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar'

## Helpers

def path_testdata(filename):
    """Returns the absolute path of a test data file (in ./testdata)

    Arguments:
    filename    - filename to append to path of ./testdata
    """
    current_dir = path.dirname(path.realpath(__file__))
    return path.join(current_dir, f"testdata/{filename}")


def empty_model(name, lang_classes_factory):
    """Fixture that generates a model for tests

    Uses coreLang specification (fixture) to create and return Model
    """

    # Create instance model from model json file
    return Model(name, lang_classes_factory)

## Fixtures

@pytest.fixture(scope="session", name="env")
def fixture_env()-> MalSimVectorizedObsEnv:

    attack_graph = create_attack_graph(lang_file_name, model_file_name)
    attack_graph.save_to_file(attack_graph_file_name)
    env = MalSimVectorizedObsEnv(MalSimulator(attack_graph, max_iter=1000))
    env.register_defender('defender')

    attacker_id = env.sim.attack_graph.attackers[0].id
    env.register_attacker('attacker', attacker_id)

    return env


@pytest.fixture
def corelang_lang_graph():
    """Fixture that returns the coreLang language specification as dict"""
    mar_file_path = path_testdata("org.mal-lang.coreLang-1.0.0.mar")
    return LanguageGraph.from_mar_archive(mar_file_path)


@pytest.fixture
def traininglang_lang_graph():
    """Fixture that returns the trainingLang language specification as dict"""
    mar_file_path = path_testdata("langs/org.mal-lang.trainingLang-1.0.0.mar")
    return LanguageGraph.from_mar_archive(mar_file_path)


@pytest.fixture
def traininglang_model(traininglang_lang_graph):
    """Fixture that generates a model for tests

    Uses trainingLang specification (fixture) to create and return a
    Model object with no assets or associations
    """
    # Init LanguageClassesFactory
    traininglang_model_file = 'tests/testdata/models/traininglang_model.yml'
    lang_classes_factory = LanguageClassesFactory(traininglang_lang_graph)
    return Model.load_from_file(traininglang_model_file, lang_classes_factory)


@pytest.fixture
def model(corelang_lang_graph):
    """Fixture that generates a model for tests

    Uses coreLang specification (fixture) to create and return a
    Model object with no assets or associations
    """
    # Init LanguageClassesFactory
    lang_classes_factory = LanguageClassesFactory(corelang_lang_graph)
    return Model.load_from_file(model_file_name, lang_classes_factory)
