from os import path
import pytest

from maltoolbox.model import Model
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode, create_attack_graph
from maltoolbox.language import (
    LanguageGraph,
    LanguageGraphAttackStep,
    LanguageGraphAsset,
)
from malsim.mal_simulator import MalSimulator
from malsim.envs import MalSimVectorizedObsEnv

model_file_name = 'tests/testdata/models/simple_test_model.yml'
attack_graph_file_name = path.join('/tmp', 'attack_graph.json')
lang_file_name = 'tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar'

## Helpers


def get_node(graph: AttackGraph, full_name: str) -> AttackGraphNode:
    node = graph.get_node_by_full_name(full_name)
    assert node, f'Node {full_name} does not exist in graph'
    return node


def path_testdata(filename: str) -> str:
    """Returns the absolute path of a test data file (in ./testdata)

    Arguments:
    filename    - filename to append to path of ./testdata
    """
    current_dir = path.dirname(path.realpath(__file__))
    return path.join(current_dir, f'testdata/{filename}')


## Fixtures


@pytest.fixture(scope='session', name='env')
def fixture_env() -> MalSimVectorizedObsEnv:
    attack_graph = create_attack_graph(lang_file_name, model_file_name)
    attack_graph.save_to_file(attack_graph_file_name)
    env = MalSimVectorizedObsEnv(MalSimulator(attack_graph))
    env.register_defender('defender')

    os_app_fa = get_node(attack_graph, 'OS App:fullAccess')
    env.register_attacker('attacker', {os_app_fa})

    return env


@pytest.fixture
def corelang_lang_graph() -> LanguageGraph:
    """Fixture that returns the coreLang language specification as dict"""
    mar_file_path = path_testdata('org.mal-lang.coreLang-1.0.0.mar')
    return LanguageGraph.from_mar_archive(mar_file_path)


@pytest.fixture
def traininglang_lang_graph() -> LanguageGraph:
    """Fixture that returns the trainingLang language specification as dict"""
    mar_file_path = path_testdata('langs/org.mal-lang.trainingLang-1.0.0.mar')
    return LanguageGraph.from_mar_archive(mar_file_path)


@pytest.fixture
def traininglang_model(traininglang_lang_graph: LanguageGraph) -> Model:
    """Fixture that generates a model for tests

    Uses trainingLang specification (fixture) to create and return a
    Model object with no assets or associations
    """
    # Init LanguageClassesFactory
    traininglang_model_file = 'tests/testdata/models/traininglang_model.yml'
    return Model.load_from_file(traininglang_model_file, traininglang_lang_graph)


@pytest.fixture
def model(corelang_lang_graph: LanguageGraph) -> Model:
    """Fixture that generates a model for tests

    Uses coreLang specification (fixture) to create and return a
    Model object with no assets or associations
    """
    # Init LanguageClassesFactory
    return Model.load_from_file(model_file_name, corelang_lang_graph)


@pytest.fixture
def dummy_lang_graph(corelang_lang_graph: LanguageGraph) -> LanguageGraph:
    """Fixture that generates a dummy LanguageGraph with a dummy
    LanguageGraphAsset and LanguageGraphAttackStep
    """
    lang_graph = LanguageGraph()
    dummy_asset = LanguageGraphAsset(name='DummyAsset')
    lang_graph.assets['DummyAsset'] = dummy_asset
    dummy_or_attack_step_node = LanguageGraphAttackStep(
        name='DummyOrAttackStep',
        type='or',
        asset=dummy_asset,
        ttc={'arguments': [1.0], 'name': 'Bernoulli', 'type': 'function'},
    )
    dummy_asset.attack_steps['DummyOrAttackStep'] = dummy_or_attack_step_node

    dummy_and_attack_step_node = LanguageGraphAttackStep(
        name='DummyAndAttackStep',
        type='and',
        asset=dummy_asset,
        ttc={'arguments': [1.0], 'name': 'Bernoulli', 'type': 'function'},
    )
    dummy_asset.attack_steps['DummyAndAttackStep'] = dummy_and_attack_step_node

    dummy_defense_attack_step_node = LanguageGraphAttackStep(
        name='DummyDefenseAttackStep',
        type='defense',
        asset=dummy_asset,
        ttc={'arguments': [0.0], 'name': 'Bernoulli', 'type': 'function'},
    )
    dummy_asset.attack_steps['DummyDefenseAttackStep'] = dummy_defense_attack_step_node

    dummy_exist_attack_step_node = LanguageGraphAttackStep(
        name='DummyExistAttackStep', type='exist', asset=dummy_asset
    )
    dummy_asset.attack_steps['DummyExistAttackStep'] = dummy_exist_attack_step_node

    dummy_exist_attack_step_node = LanguageGraphAttackStep(
        name='DummyNotExistAttackStep', type='notExist', asset=dummy_asset
    )
    dummy_asset.attack_steps['DummyNotExistAttackStep'] = dummy_exist_attack_step_node

    return lang_graph
