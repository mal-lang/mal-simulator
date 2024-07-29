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
        path_relative_to_tests('./testdata/simple_scenario.yml')
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

    # Verify attacker entrypoint was added
    # 0: ['Credentials:6:attemptCredentialsReuse']
    attack_step = attack_graph.get_node_by_full_name(
        'Credentials:6:attemptCredentialsReuse'
    )
    attacker_id = 0
    assert attack_step in attack_graph\
        .get_attacker_by_id(attacker_id).entry_points

    assert config.get('attacker_agent_class') == BreadthFirstAttacker
    assert config.get('defender_agent_class') == KeyboardAgent


def test_load_scenario_agent_class_error():
    """Make sure we get error when loading with wrong class"""

    # Load the scenario
    with pytest.raises(LookupError):
        load_scenario(
            path_relative_to_tests(
                './testdata/scenario_wrong_class.yml'
            )
        )
