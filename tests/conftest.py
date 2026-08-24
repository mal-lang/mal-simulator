from os import path
import pytest

from maltoolbox.model import Model
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from maltoolbox.language import (
    LanguageGraph,
    LanguageGraphAttackStep,
    LanguageGraphAsset,
)

from malsim.scenario.scenario import Scenario

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
    model_file_name = 'tests/testdata/models/simple_test_model.yml'
    return Model.load_from_file(model_file_name, corelang_lang_graph)


@pytest.fixture
def dummy_lang_graph(corelang_lang_graph: LanguageGraph) -> LanguageGraph:
    """Fixture that generates a dummy LanguageGraph with a dummy
    LanguageGraphAsset and LanguageGraphAttackStep
    """
    lang_graph = LanguageGraph()
    lang_graph.metadata = {}
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


@pytest.fixture
def wiperLang_lang_graph() -> LanguageGraph:
    """Fixture that returns the wiperLang language specification as dict"""
    mal_spec_file_path = path_testdata('langs/wiperLang.mal')
    return LanguageGraph.from_mal_spec(mal_spec_file_path)


@pytest.fixture
def wiperLang_model(wiperLang_lang_graph: LanguageGraph) -> Model:
    """Fixture that generates an example model for wiperLang"""

    model_file_path = path_testdata('models/wiper_model.yml')
    return Model.load_from_file(model_file_path, wiperLang_lang_graph)


@pytest.fixture
def wiperLang_attack_graph(
    wiperLang_model: Model, wiperLang_lang_graph: LanguageGraph
) -> AttackGraph:
    """Fixture that generates an attack graph for wiperLang"""

    return AttackGraph(wiperLang_lang_graph, wiperLang_model)


@pytest.fixture
def wiperLang_scenario() -> Scenario:
    """Fixture that generates a scenario for wiperLang"""
    scenario_file_path = path_testdata('scenarios/wiper_scenario.yml')
    return Scenario.load_from_file(scenario_file_path)


@pytest.fixture
def dynamic_add_remove_scenario() -> Scenario:
    """Fixture that generates a scenario for dynamic_add_remove"""
    scenario_file_path = path_testdata('scenarios/dynamic_add_remove.yml')
    return Scenario.load_from_file(scenario_file_path)


@pytest.fixture
def int_dynamic_test_lang6_scenario() -> Scenario:
    """Fixture for the intDynamicTestLang6 example (intermediate)"""
    scenario_file_path = path_testdata(
        'scenarios/dynamal_example_scenarios/intermediate/'
        'intDynamicTestLang6_scenario.yml'
    )
    return Scenario.load_from_file(scenario_file_path)


@pytest.fixture
def int_dynamic_test_lang3_scenario() -> Scenario:
    """Fixture for the intDynamicTestLang3 example (intermediate)"""
    scenario_file_path = path_testdata(
        'scenarios/dynamal_example_scenarios/intermediate/'
        'intDynamicTestLang3_scenario.yml'
    )
    return Scenario.load_from_file(scenario_file_path)


@pytest.fixture
def int_dynamic_test_lang12_scenario() -> Scenario:
    """Fixture for the intDynamicTestLang12 example (intermediate)"""
    scenario_file_path = path_testdata(
        'scenarios/dynamal_example_scenarios/intermediate/'
        'intDynamicTestLang12_scenario.yml'
    )
    return Scenario.load_from_file(scenario_file_path)


@pytest.fixture
def easy_ransomware_lang_scenario() -> Scenario:
    """Fixture for the easyRansomwareLang example (basic)"""
    scenario_file_path = path_testdata(
        'scenarios/dynamal_example_scenarios/basic/easyRansomwareLang_scenario.yml'
    )
    return Scenario.load_from_file(scenario_file_path)


@pytest.fixture
def dynamic_remove_many_assoc_scenario() -> Scenario:
    """Fixture for the dynamic_remove_many_assoc example"""
    scenario_file_path = path_testdata(
        'scenarios/dynamic_remove_many_assoc_scenario.yml'
    )
    return Scenario.load_from_file(scenario_file_path)


@pytest.fixture
def rand_multiplicity_scenario() -> Scenario:
    """Fixture for the multiplicity example scenario"""
    scenario_file_path = path_testdata('scenarios/rand_multiplicity_scenario.yml')
    return Scenario.load_from_file(scenario_file_path)
