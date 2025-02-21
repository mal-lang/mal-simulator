"""Test functions that load scenarios"""

import os
import pytest

from malsim.scenario import (
    apply_scenario_node_property_rules,
    load_scenario
)
from malsim.agents import PassiveAgent, BreadthFirstAttacker

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
        (attacker for attacker in attack_graph.attackers.values()
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


def test_load_scenario_no_defender_agent():
    """Make sure we can load a scenario"""

    # Load the scenario
    _, agents = load_scenario(
        path_relative_to_tests(
            './testdata/scenarios/no_defender_agent_scenario.yml'
        )
    )
    assert 'defender' not in [a['name'] for a  in agents]
    assert isinstance(agents[0]['agent'], BreadthFirstAttacker)


def test_load_scenario_agent_class_error():
    """Make sure we get error when loading with wrong class"""

    # Load the scenario
    with pytest.raises(LookupError):
        load_scenario(
            path_relative_to_tests(
                './testdata/scenarios/wrong_agent_classes_scenario.yml'
            )
        )


def test_load_scenario_observability_given():
    """Load a scenario with observability settings given and
    make sure observability is applied correctly"""

    # Load scenario with observability specifed
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


def test_load_scenario_observability_not_given():
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


def test_apply_scenario_observability():
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
    apply_scenario_node_property_rules(attack_graph, 'observable', observability_rules)

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

def test_apply_scenario_observability_faulty():
    """Try different failing cases for observability settings"""

    # Load scenario with no observability specified
    attack_graph, _ = load_scenario(
        path_relative_to_tests(
            'testdata/scenarios/simple_scenario.yml')
    )

    # Wrong key in rule dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property_rules(
            attack_graph,
            'observable',
            {'NotAllowedKey': {'Data': ['read', 'write', 'delete']}}
        )

    # Correct asset type and attack step
    apply_scenario_node_property_rules(
        attack_graph,
        'observable',
        {'by_asset_type': { 'Application': ['read']},
    })

    # Wrong asset type in rule asset type to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property_rules(
            attack_graph,
            'observable',
            {'by_asset_type': {'NonExistingType': ['read']}}
        )

    # Wrong attack step name in rule asset type to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property_rules(
            attack_graph,
            'observable',
            {'by_asset_type': {'Data': ['nonExistingAttackStep']},
        })

    # Correct asset name and attack step
    apply_scenario_node_property_rules(
        attack_graph,
        'observable',
        {'by_asset_name': { 'OS App': ['read']},
    })

    # Wrong asset name in rule asset name to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property_rules(
             attack_graph,
            'observable',
            {'by_asset_name': { 'NonExistingName': ['read']},
        })

    # Wrong attack step name in rule asset name to step dict
    with pytest.raises(AssertionError):
        apply_scenario_node_property_rules(
            attack_graph,
            'observable',
            {'by_asset_name': {'OS App': ['nonExistingAttackStep']}}
        )
