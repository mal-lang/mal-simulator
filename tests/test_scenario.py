"""Test functions that load scenarios"""

import os
from malsim.scenario import load_scenario


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
    attack_graph = load_scenario(
        path_relative_to_tests('./testdata/simple_scenario.yml')
    )

    # Verify rewards were added as defined in './testdata/simple_scenario.yml'
    assert attack_graph.get_node_by_id('OS App:notPresent')\
        .reward == 2
    assert attack_graph.get_node_by_id('OS App:supplyChainAuditing')\
        .reward == 7
    assert attack_graph.get_node_by_id('Program 1:notPresent')\
        .reward == 3
    assert attack_graph.get_node_by_id('Program 1:supplyChainAuditing')\
        .reward == 7
    assert attack_graph.get_node_by_id('SoftwareVulnerability:4:notPresent')\
        .reward == 4
    assert attack_graph.get_node_by_id('Data:5:notPresent')\
        .reward == 1
    assert attack_graph.get_node_by_id('Credentials:6:notPhishable')\
        .reward == 7
    assert attack_graph.get_node_by_id('Identity:11:notPresent')\
        .reward == 3.5
