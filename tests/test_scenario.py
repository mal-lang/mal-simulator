"""Test functions that load scenarios"""

import os
import pytest

from malsim.scenario import load_scenario
from malsim.agents.keyboard_input import KeyboardAgent
from malsim.agents.searchers import BreadthFirstAttacker


def path_relative_to_tests(filename):
    """Returns the absolute path of a file in ./tests

    Arguments:
    filename    - filename to append to tests path
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, f"{filename}")


def test_load_scenario():
    """Make sure we can load a scenario"""

    # Load the scenario
    attack_graph, config = load_scenario(
        path_relative_to_tests('./testdata/scenarios/simple_scenario.yml')
    )

    # Verify rewards were added as defined in './testdata/simple_scenario.yml'
    assert attack_graph.get_node_by_full_name('OS App:notPresent')\
        .extras['reward'] == 2
    assert attack_graph.get_node_by_full_name('OS App:supplyChainAuditing')\
        .extras['reward'] == 7
    assert attack_graph.get_node_by_full_name('Program 1:notPresent')\
        .extras['reward'] == 3
    assert attack_graph.get_node_by_full_name('Program 1:supplyChainAuditing')\
        .extras['reward'] == 7
    assert attack_graph.get_node_by_full_name('SoftwareVulnerability:4:notPresent')\
        .extras['reward'] == 4
    assert attack_graph.get_node_by_full_name('Data:5:notPresent')\
        .extras['reward'] == 1
    assert attack_graph.get_node_by_full_name('Credentials:6:notPhishable')\
        .extras['reward'] == 7
    assert attack_graph.get_node_by_full_name('Identity:11:notPresent')\
        .extras['reward'] == 3.5

    # One attacker from scenario (overrides attacker from model)
    assert len(attack_graph.attackers) == 1

    # Verify attacker entrypoint was added
    attack_step = attack_graph.get_node_by_full_name(
        'Credentials:6:attemptCredentialsReuse'
    )
    attacker_name = "Attacker1"
    attacker = next(
        (attacker for attacker in attack_graph.attackers
         if attacker.name == attacker_name)
    )
    assert attack_step in attacker.entry_points

    assert config['agents']['attacker']['agent_class'] == BreadthFirstAttacker
    assert config['agents']['defender']['agent_class'] == KeyboardAgent


def test_load_scenario_no_attacker_in_model():
    """Make sure we can load a scenario"""

    # Load the scenario
    attack_graph, _ = load_scenario(
        path_relative_to_tests('./testdata/scenarios/no_existing_attacker_in_model_scenario.yml')
    )

    # Verify one attacker entrypoint was added (model is missing attacker)
    assert len(attack_graph.attackers) == 1
    attack_step = attack_graph.get_node_by_full_name(
        'Credentials:6:attemptCredentialsReuse'
    )
    attacker_name = "Attacker1"
    attacker = next(
        (attacker for attacker in attack_graph.attackers
         if attacker.name == attacker_name)
    )
    assert attack_step in attacker.entry_points


def test_load_scenario_attacker_in_model():
    """
    Make sure model attacker is removed if scenario has attacker
    Make sure model attacker is not removed if scenario has no attacker
    """

    # Load the scenario that has entry point defined
    attack_graph, _ = load_scenario(
        path_relative_to_tests(
            'testdata/scenarios/simple_scenario.yml')
    )
    assert len(attack_graph.attackers) == 1
    assert attack_graph.attackers[0].name == 'Attacker1' # From scenario

    # Load the scenario that has no entry point defined
    attack_graph, _ = load_scenario(
        path_relative_to_tests(
            'testdata/scenarios/no_entry_points_simple_scenario.yml')
    )
    assert len(attack_graph.attackers) == 1
    assert attack_graph.attackers[0].name == 'Attacker:15' # From model


def test_load_scenario_no_defender_agent():
    """Make sure we can load a scenario"""

    # Load the scenario
    _, config = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/no_defender_agent_scenario.yml'
        )
    )
    assert 'defender' not in config['agents']
    assert config['agents']['attacker']['agent_class'] == BreadthFirstAttacker


def test_load_scenario_agent_class_error():
    """Make sure we get error when loading with wrong class"""

    # Load the scenario
    with pytest.raises(LookupError):
        load_scenario(
            path_relative_to_tests(
                './testdata/scenarios/wrong_agent_classes_scenario.yml'
            )
        )
