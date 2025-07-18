"""Test functions that load scenarios"""

import os
import pytest

from maltoolbox.attackgraph import create_attack_graph
from malsim.scenario import (
    apply_scenario_node_property,
    load_scenario,
    apply_scenario_to_attack_graph
)
from malsim.agents import PassiveAgent, BreadthFirstAttacker

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
    attack_graph, agents = load_scenario(
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
        (attacker for attacker in attack_graph.attackers.values()
         if attacker.name == attacker_name)
    )
    assert attack_step in attacker.entry_points

    # Entry points list and reached attack steps list are different lists
    assert id(attacker.entry_points) != id(attacker.reached_attack_steps)

    assert isinstance(agents[0]['agent'], BreadthFirstAttacker)
    assert isinstance(agents[1]['agent'], PassiveAgent)


def test_load_scenario_no_attacker_in_model() -> None:
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
        (attacker for attacker in attack_graph.attackers.values()
         if attacker.name == attacker_name)
    )
    assert attack_step in attacker.entry_points


def test_load_scenario_attacker_in_model() -> None:
    """
    Make sure model attacker is removed if scenario has attacker
    Make sure model attacker is not removed if scenario has no attacker
    """

    # Load the scenario that has entry point defined
    attack_graph, _ = load_scenario(
        path_relative_to_tests(
            'testdata/scenarios/simple_scenario.yml')
    )

    all_attackers = list(attack_graph.attackers.values())
    assert len(all_attackers) == 1
    assert all_attackers[0].name == 'Attacker1' # From scenario

    # Load the scenario that has no entry point defined
    attack_graph, _ = load_scenario(
        path_relative_to_tests(
            'testdata/scenarios/no_entry_points_simple_scenario.yml')
    )
    all_attackers = list(attack_graph.attackers.values())
    assert len(all_attackers) == 1
    assert all_attackers[0].name == 'Attacker:15' # From scenario


def test_load_scenario_no_defender_agent() -> None:
    """Make sure we can load a scenario"""

    # Load the scenario
    _, agents = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/no_defender_agent_scenario.yml'
        )
    )
    assert 'defender' not in [a['name'] for a  in agents]
    assert isinstance(agents[0]['agent'], BreadthFirstAttacker)


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
    attack_graph, _ = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/simple_filtered_observability_scenario.yml')
    )

    # Make sure only attack steps of name fullAccess
    # part of asset type Application are observable.
    for node in attack_graph.nodes.values():
        if node.lg_attack_step.asset.name == "Application" and node.name == "fullAccess":
            assert node.extras['observable']
        elif node.lg_attack_step.asset.name == "Application" and node.name == "supplyChainAuditing":
            assert node.extras['observable']
        elif node.model_asset.name == "Identity:8" and node.name == "assume":
            assert node.extras['observable']
        else:
            assert not node.extras['observable']


def test_load_scenario_observability_not_given() -> None:
    """Load a scenario where no observability settings are given"""
    # Load scenario with no observability specifed
    attack_graph, _ = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/simple_scenario.yml'
        )
    )
    # Make sure all attack steps are observable
    # if no observability settings are given
    for node in attack_graph.nodes.values():
        assert node.extras['observable']


def test_apply_scenario_observability() -> None:
    """Try different cases for observability settings"""

    # Load scenario with no observability specified
    attack_graph, _ = load_scenario(
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
    apply_scenario_node_property(
        attack_graph,
        'observable',
        observability_rules,
        assumed_value = 1,
        default_value = 0
    )

    # Make sure all attack steps are observable
    # if no observability settings are given
    for node in attack_graph.nodes.values():
        if node.lg_attack_step.asset.name == 'Data' and node.name in ('read', 'write', 'delete'):
            assert node.extras['observable']
        elif node.lg_attack_step.asset.name == 'Application' and node.name in ('fullAccess', 'notPresent'):
            assert node.extras['observable']
        elif node.model_asset.name == 'OS App' and node.name in ('read'):
            assert node.extras['observable']
        else:
            assert not node.extras['observable']

def test_apply_scenario_observability_faulty() -> None:
    """Try different failing cases for observability settings"""

    # Load scenario with no observability specified
    attack_graph, _ = load_scenario(
        path_relative_to_tests(
            'testdata/scenarios/simple_scenario.yml')
    )

    # Wrong key in rule dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property(
            attack_graph,
            'observable',
            {'NotAllowedKey': {'Data': ['read', 'write', 'delete']}},
            assumed_value = 1,
            default_value = 0
        )

    # Correct asset type and attack step
    apply_scenario_node_property(
        attack_graph,
        'observable',
        {'by_asset_type': { 'Application': ['read']}},
        assumed_value = 1,
        default_value = 0
    )

    # Wrong asset type in rule asset type to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property(
            attack_graph,
            'observable',
            {'by_asset_type': {'NonExistingType': ['read']}},
            assumed_value = 1,
            default_value = 0
        )

    # Wrong attack step name in rule asset type to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property(
            attack_graph,
            'observable',
            {'by_asset_type': {'Data': ['nonExistingAttackStep']}},
            assumed_value = 1,
            default_value = 0
        )

    # Correct asset name and attack step
    apply_scenario_node_property(
        attack_graph,
        'observable',
        {'by_asset_name': { 'OS App': ['read']}},
        assumed_value = 1,
        default_value = 0
    )

    # Wrong asset name in rule asset name to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property(
             attack_graph,
            'observable',
            {'by_asset_name': { 'NonExistingName': ['read']}},
            assumed_value = 1,
            default_value = 0
        )

    # Wrong attack step name in rule asset name to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property(
            attack_graph,
            'observable',
            {'by_asset_name': {'OS App': ['nonExistingAttackStep']}},
            assumed_value = 1,
            default_value = 0
        )


def test_load_scenario_false_positive_negative_rate() -> None:
    """Load a scenario with observability settings given and
    make sure observability is applied correctly"""

    # Load scenario with observability specifed
    attack_graph, _ = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/traininglang_fp_fn_scenario.yml')
    )

    # Defined in scenario file
    host_0_access_fp_rate = 0.2
    host_1_access_fp_rate = 0.3
    host_0_access_fn_rate = 0.4
    host_1_access_fn_rate = 0.5
    user_3_compromise_fn_rate = 1.0

    for node in attack_graph.nodes.values():

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
    attack_graph, _ = load_scenario(
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
    apply_scenario_node_property(
        attack_graph, 'false_negative_rate', property_values
    )

    # Make sure all attack steps are observable
    # if no observability settings are given
    for node in attack_graph.nodes.values():
        if node.lg_attack_step.asset.name == 'Data' and node.name == 'read':
            assert node.extras['false_negative_rate'] == 0.5
        elif node.lg_attack_step.asset.name == 'Data' and node.name == 'write':
            assert node.extras['false_negative_rate'] == 0.6
        elif node.lg_attack_step.asset.name == 'Data' and node.name == 'delete':
            assert node.extras['false_negative_rate'] == 0.7
        elif node.model_asset.name == 'OS App' and node.name == 'read':
            assert node.extras['false_negative_rate'] == 0.9
        elif node.lg_attack_step.asset.name == 'Application' and node.name == 'read':
            assert node.extras['false_negative_rate'] == 1.0
        elif node.lg_attack_step.asset.name == 'Application' and node.name == 'fullAccess':
            assert node.extras['false_negative_rate'] == 0.8
        else:
            assert 'false_negative_rate' not in node.extras


def test_apply_scenario_rewards_old_format() -> None:
    """Try different cases for rewards"""

    # Load scenario with no specified
    lang_file = 'tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar'
    model_file = 'tests/testdata/models/simple_test_model.yml'
    scenario = {
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

    with pytest.raises(RuntimeError):
        # Make sure we get error when loading with wrong rewards format
        apply_scenario_to_attack_graph(attack_graph, scenario)
