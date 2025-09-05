"""Test functions that load scenarios"""

import os
import pytest
from typing import Any

from maltoolbox.attackgraph import create_attack_graph
from malsim.scenario import (
    apply_scenario_node_property,
    load_scenario,
    _validate_scenario_node_property_config
)
from malsim.agents import PassiveAgent, BreadthFirstAttacker

from .conftest import get_node

def path_relative_to_tests(filename: str) -> str:
    """Returns the absolute path of a file in ./tests

    Arguments:
    filename    - filename to append to tests path
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, f"{filename}")


def test_load_scenario() -> None:
    """Make sure we can load a scenario"""


    # Load the scenario
    scenario = load_scenario(
        path_relative_to_tests('./testdata/scenarios/simple_scenario.yml')
    )

    # Verify rewards were added as defined in './testdata/simple_scenario.yml'
    assert get_node(scenario.attack_graph, 'OS App:notPresent')\
        .extras['reward'] == 2
    assert get_node(scenario.attack_graph, 'OS App:supplyChainAuditing')\
        .extras['reward'] == 7
    assert get_node(scenario.attack_graph, 'Program 1:notPresent')\
        .extras['reward'] == 3
    assert get_node(scenario.attack_graph, 'Program 1:supplyChainAuditing')\
        .extras['reward'] == 7
    assert get_node(scenario.attack_graph, 'SoftwareVulnerability:4:notPresent')\
        .extras['reward'] == 4
    assert get_node(scenario.attack_graph, 'Data:5:notPresent')\
        .extras['reward'] == 1
    assert get_node(scenario.attack_graph, 'Credentials:6:notPhishable')\
        .extras['reward'] == 7
    assert get_node(scenario.attack_graph, 'Identity:11:notPresent')\
        .extras['reward'] == 3.5

    # Verify attacker entrypoint was added
    attack_step = get_node(scenario.attack_graph, 'OS App:fullAccess')
    assert attack_step in scenario.agents[0]['entry_points']

    assert isinstance(scenario.agents[0]['agent'], BreadthFirstAttacker)
    assert isinstance(scenario.agents[1]['agent'], PassiveAgent)


def test_extend_scenario() -> None:
    """Make sure we can extend a scenario"""

    # Load the scenario
    scenario = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/traininglang_scenario_extended.yml'
        )
    )
    num_nodes_with_reward = 0
    for node in scenario.attack_graph.nodes.values():
        reward = node.extras.get('reward')
        if reward:
            # All nodes with reward set should have reward 1
            # Since this is defined in the overriding scenario
            num_nodes_with_reward += 1
            assert reward == 1
    assert num_nodes_with_reward == 7

    # 2 agents are defined in the original scenario
    assert len(scenario.agents) == 2


def test_extend_scenario_deeper() -> None:
    """
    Make sure we can extend a scenario several levels
    and in different sub directory
    """

    # Load the scenario from a sub folder extending another scenario
    scenario = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/sub/traininglang_scenario_extended_again.yml'
        )
    )
    num_nodes_with_reward = 0
    for node in scenario.attack_graph.nodes.values():
        reward = node.extras.get('reward')
        if reward:
            # All nodes with reward set should have reward 1
            # Since this is defined in the extended scenario
            num_nodes_with_reward += 1
            assert reward == 1
    assert num_nodes_with_reward == 7

    # 1 agents are defined in the extended_again scenario
    assert len(scenario.agents) == 1


def test_extend_scenario_override_lang_model() -> None:
    """
    Make sure we can extend a scenario several levels
    and in different sub directory
    """

    # Load the scenario from a sub folder extending another scenario
    # that overrides lang and model
    scenario = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/sub/traininglang_scenario_override_lang_model.yml'
        )
    )

    # No reward overrides
    assert get_node(scenario.attack_graph, 'Host:0:notPresent').extras['reward'] == 2
    assert get_node(scenario.attack_graph, 'Host:0:access').extras['reward'] == 4
    assert get_node(scenario.attack_graph, 'Host:1:notPresent').extras['reward'] == 7
    assert get_node(scenario.attack_graph, 'Host:1:access').extras['reward'] == 5
    assert get_node(scenario.attack_graph, 'Data:2:notPresent').extras['reward'] == 8
    assert get_node(scenario.attack_graph, 'Data:2:read').extras['reward'] == 5
    assert get_node(scenario.attack_graph, 'Data:2:modify').extras['reward'] == 10

    # No agent overrides
    assert len(scenario.agents) == 2


def test_load_scenario_no_defender_agent() -> None:
    """Make sure we can load a scenario"""

    # Load the scenario
    scenario = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/no_defender_agent_scenario.yml'
        )
    )
    assert 'defender' not in [a['name'] for a  in scenario.agents]
    assert isinstance(scenario.agents[0]['agent'], BreadthFirstAttacker)


def test_load_scenario_agent_class_error() -> None:
    """Make sure we get error when loading with wrong class"""

    # Load the scenario
    with pytest.raises(LookupError):
        load_scenario(
            path_relative_to_tests(
                './testdata/scenarios/wrong_agent_classes_scenario.yml'
            )
        )


def test_load_scenario_observability_given() -> None:
    """Load a scenario with observability settings given and
    make sure observability is applied correctly"""

    # Load scenario with observability specified
    scenario = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/simple_filtered_observability_scenario.yml')
    )

    # Make sure only attack steps of name fullAccess
    # part of asset type Application are observable.
    for node in scenario.attack_graph.nodes.values():
        if node.lg_attack_step.asset.name == "Application" and node.name == "fullAccess":
            assert node.extras['observable']
        elif node.lg_attack_step.asset.name == "Application" and node.name == "supplyChainAuditing":
            assert node.extras['observable']
        elif node.model_asset and node.model_asset.name == "Identity:8" and node.name == "assume":
            assert node.extras['observable']
        else:
            assert not node.extras['observable']


def test_load_scenario_observability_not_given() -> None:
    """Load a scenario where no observability settings are given"""
    # Load scenario with no observability specifed
    scenario = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/simple_scenario.yml'
        )
    )

    assert not scenario.is_observable


def test_apply_scenario_observability() -> None:
    """Try different cases for observability settings"""

    # Load scenario with no observability specified
    scenario = load_scenario(
        path_relative_to_tests(
            'testdata/scenarios/simple_scenario.yml')
    )

    # Make Data: read, write, delete observable
    # Make Application: fullAccess, notPresent observable
    observability_rules = {
        'by_asset_type': {
            'Data': ['read', 'write', 'delete'],
            'Application': ['fullAccess', 'notPresent']
        },
        'by_asset_name': {
            'OS App': ['read']
        }
    }

    # Apply observability rules
    observable = apply_scenario_node_property(
        scenario.attack_graph,
        'observable',
        observability_rules,
        default_value = 0
    )

    # Make sure all attack steps are observable
    # if no observability settings are given
    for node in scenario.attack_graph.nodes.values():
        if node.lg_attack_step.asset.name == 'Data' and node.name in ('read', 'write', 'delete'):
            assert observable[node]
        elif node.lg_attack_step.asset.name == 'Application' and node.name in ('fullAccess', 'notPresent'):
            assert observable[node]
        elif node.model_asset and node.model_asset.name == 'OS App' and node.name in ('read'):
            assert observable[node]
        else:
            assert not observable[node]

def test_apply_scenario_observability_faulty() -> None:
    """Try different failing cases for observability settings"""

    # Load scenario with no observability specified
    scenario = load_scenario(
        path_relative_to_tests(
            'testdata/scenarios/simple_scenario.yml')
    )

    # Wrong key in rule dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property(
            scenario.attack_graph,
            'observable',
            {'NotAllowedKey': {'Data': ['read', 'write', 'delete']}},
            default_value = 0
        )

    # Correct asset type and attack step
    apply_scenario_node_property(
        scenario.attack_graph,
        'observable',
        {'by_asset_type': { 'Application': ['read']}},
        default_value = 0
    )

    # Wrong asset type in rule asset type to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property(
            scenario.attack_graph,
            'observable',
            {'by_asset_type': {'NonExistingType': ['read']}},
            default_value = 0
        )

    # Wrong attack step name in rule asset type to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property(
            scenario.attack_graph,
            'observable',
            {'by_asset_type': {'Data': ['nonExistingAttackStep']}},
            default_value = 0
        )

    # Correct asset name and attack step
    apply_scenario_node_property(
        scenario.attack_graph,
        'observable',
        {'by_asset_name': { 'OS App': ['read']}},
        default_value = 0
    )

    # Wrong asset name in rule asset name to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property(
            scenario.attack_graph,
            'observable',
            {'by_asset_name': { 'NonExistingName': ['read']}},
            default_value = 0
        )

    # Wrong attack step name in rule asset name to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property(
            scenario.attack_graph,
            'observable',
            {'by_asset_name': {'OS App': ['nonExistingAttackStep']}},
            default_value = 0
        )


def test_load_scenario_false_positive_negative_rate() -> None:
    """Load a scenario with observability settings given and
    make sure observability is applied correctly"""

    # Load scenario with observability specifed
    scenario = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/traininglang_fp_fn_scenario.yml')
    )

    # Defined in scenario file
    host_0_access_fp_rate = 0.2
    host_1_access_fp_rate = 0.3
    host_0_access_fn_rate = 0.4
    host_1_access_fn_rate = 0.5
    user_3_compromise_fn_rate = 1.0

    for node in scenario.attack_graph.nodes.values():

        if node.full_name == "Host:0:access":
            # According to scenario file
            assert node.extras['false_positive_rate'] == host_0_access_fp_rate
            assert node.extras['false_negative_rate'] == host_0_access_fn_rate

        elif node.full_name == "Host:1:access":
            # According to scenario file
            assert node.extras['false_positive_rate'] == host_1_access_fp_rate
            assert node.extras['false_negative_rate'] == host_1_access_fn_rate

        elif node.full_name == "User:3:compromise":
            # According to scenario file
            assert 'false_positive_rate' not in node.extras
            assert node.extras['false_negative_rate'] \
                == user_3_compromise_fn_rate

        else:
            # If no rules - don't set fpr/fnr
            assert 'false_positive_rate' not in node.extras
            assert 'false_negative_rate' not in node.extras

def test_apply_scenario_fpr_fnr() -> None:
    """Try different cases for false positives/negatives rates"""

    # Load scenario with no specified
    scenario = load_scenario(
        path_relative_to_tests(
            'testdata/scenarios/simple_scenario.yml')
    )

    property_values = {
        'by_asset_type': {
            'Data': {
                'read': 0.5,
                'write': 0.6,
                'delete': 0.7
            },
            'Application': {
                'fullAccess': 0.8,
                'read': 1.0
            }
        },
        'by_asset_name': {
            'OS App': {
                'read': 0.9 # Has precedence
            }
        }
    }

    # Apply false negative rate rules
    false_negatives_rates = apply_scenario_node_property(
        scenario.attack_graph, 'false_negative_rate', property_values
    )

    # Make sure all attack steps are observable
    # if no observability settings are given
    for node in scenario.attack_graph.nodes.values():
        if node.lg_attack_step.asset.name == 'Data' and node.name == 'read':
            assert node.extras['false_negative_rate'] == 0.5
            assert false_negatives_rates[node] == 0.5
        elif node.lg_attack_step.asset.name == 'Data' and node.name == 'write':
            assert node.extras['false_negative_rate'] == 0.6
            assert false_negatives_rates[node] == 0.6
        elif node.lg_attack_step.asset.name == 'Data' and node.name == 'delete':
            assert node.extras['false_negative_rate'] == 0.7
            assert false_negatives_rates[node] == 0.7
        elif node.model_asset and node.model_asset.name == 'OS App' and node.name == 'read':
            assert node.extras['false_negative_rate'] == 0.9
            assert false_negatives_rates[node] == 0.9
        elif node.lg_attack_step.asset.name == 'Application' and node.name == 'read':
            assert node.extras['false_negative_rate'] == 1.0
            assert false_negatives_rates[node] == 1.0
        elif node.lg_attack_step.asset.name == 'Application' and node.name == 'fullAccess':
            assert node.extras['false_negative_rate'] == 0.8
            assert false_negatives_rates[node] == 0.8
        else:
            assert 'false_negative_rate' not in node.extras


def test_apply_scenario_rewards_old_format() -> None:
    """Try different cases for rewards"""

    # Load scenario with no specified
    lang_file = 'tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar'
    model_file = 'tests/testdata/models/simple_test_model.yml'
    scenario: dict[str, Any] = {
        'lang_file': lang_file,
        'model_file': model_file,

        # Rewards for each attack step (DEPRECATED format)
        'rewards': {
            'OS App:notPresent': 2,
            'OS App:supplyChainAuditing': 7,
            'Program 1:notPresent': 3,
            'Program 1:supplyChainAuditing': 7,
            'Data:5:notPresent': 1,
            'Credentials:6:notPhishable': 7,
            'Identity:11:notPresent': 3.5,
            'Identity:8:assume': 50
        },

        # Add entry points to AttackGraph with attacker names
        # and attack step full_names
        'agents': {}
    }

    attack_graph = create_attack_graph(lang_file, model_file)

    with pytest.raises(AssertionError):
        # Make sure we get error when loading with wrong rewards format
        _validate_scenario_node_property_config(
            attack_graph, scenario['rewards']
        )
