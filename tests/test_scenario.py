"""Test functions that load scenarios"""

import os
import pickle
import pytest
from typing import Any

from maltoolbox.model import Model
from malsim.config.agent_settings import AgentType, AttackerSettings, DefenderSettings
from malsim.config.node_property_rule import NodePropertyRule
from malsim.config.sim_settings import MalSimulatorSettings
from malsim.scenario.scenario import Scenario
from malsim.policies import PassiveAgent, BreadthFirstAttacker

from .conftest import get_node


def path_relative_to_tests(filename: str) -> str:
    """Returns the absolute path of a file in ./tests

    Arguments:
    filename    - filename to append to tests path
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, f'{filename}')


def test_load_scenario() -> None:
    """Make sure we can load a scenario"""

    # Load the scenario
    scenario = Scenario.load_from_file(
        path_relative_to_tests('./testdata/scenarios/simple_scenario.yml')
    )
    assert scenario.attacker_settings['Attacker1'].rewards
    rewards_per_node = scenario.attacker_settings['Attacker1'].rewards
    assert rewards_per_node.by_asset_name
    # Verify rewards were added as defined in './testdata/simple_scenario.yml'
    assert rewards_per_node.by_asset_name['OS App']['notPresent'] == 2
    assert rewards_per_node.by_asset_name['OS App']['supplyChainAuditing'] == 7
    assert rewards_per_node.by_asset_name['Program 1']['notPresent'] == 3
    assert rewards_per_node.by_asset_name['Program 1']['supplyChainAuditing'] == 7
    assert rewards_per_node.by_asset_name['SoftwareVulnerability:4']['notPresent'] == 4
    assert rewards_per_node.by_asset_name['Data:5']['notPresent'] == 1
    assert rewards_per_node.by_asset_name['Credentials:6']['notPhishable'] == 7
    assert rewards_per_node.by_asset_name['Identity:11']['notPresent'] == 3.5

    # Verify attacker entrypoint was added
    attack_step = get_node(scenario.attack_graph, 'OS App:fullAccess')
    attacker1 = scenario.attacker_settings['Attacker1']
    assert isinstance(attacker1, AttackerSettings)
    assert attack_step in attacker1.entry_points

    assert scenario.attacker_settings['Attacker1'].policy == BreadthFirstAttacker
    assert scenario.defender_settings['Defender1'].policy == PassiveAgent


def test_save_scenario(model: Model, tmp_path: Any) -> None:
    """Make sure we can load and save a scenario"""

    # Load the scenario
    scenario = Scenario(
        lang_file=path_relative_to_tests(
            'testdata/langs/org.mal-lang.coreLang-1.0.0.mar'
        ),
        model=model,
        sim_settings=MalSimulatorSettings(),
        agents=(
            AttackerSettings(
                name='Attacker1',
                entry_points=frozenset(),
                rewards=NodePropertyRule(
                    by_asset_type={'Application': {'fullAccess': 1000}}
                ),
            ),
            DefenderSettings(
                name='Defender1',
                observable_steps=NodePropertyRule(
                    by_asset_type={'Application': {'fullAccess': True}}
                ),
                actionable_steps=NodePropertyRule(
                    by_asset_type={'Application': {'fullAccess': True}}
                ),
                rewards=NodePropertyRule(
                    by_asset_type={'Application': {'fullAccess': 1000}}
                ),
                false_negative_rates=NodePropertyRule(
                    by_asset_type={'Application': {'fullAccess': 0.1}}
                ),
                false_positive_rates=NodePropertyRule(
                    by_asset_type={'Application': {'fullAccess': 0.2}}
                ),
            ),
        ),
    )
    save_path = tmp_path / 'saved_scenario.yml'
    scenario.save_to_file(save_path)
    loaded_scenario = Scenario.load_from_file(save_path)
    assert loaded_scenario.to_dict() == scenario.to_dict()


def test_extend_scenario() -> None:
    """Make sure we can extend a scenario"""

    # Load the scenario
    scenario = Scenario.load_from_file(
        path_relative_to_tests(
            './testdata/scenarios/traininglang_scenario_extended.yml'
        )
    )
    num_nodes_with_reward = 0
    assert scenario.defender_settings['Defender1'].rewards
    rewards_per_node = scenario.defender_settings['Defender1'].rewards

    for node in scenario.attack_graph.nodes.values():
        reward = rewards_per_node.value(node, 0)
        if reward:
            # All nodes with reward set should have reward 1
            # Since this is defined in the overriding scenario
            num_nodes_with_reward += 1
            assert reward == 1
    assert num_nodes_with_reward == 7

    # 2 agents are defined in the original scenario
    assert len(scenario.agent_settings) == 2


def test_extend_scenario_deeper() -> None:
    """
    Make sure we can extend a scenario several levels
    and in different sub directory
    """

    # Load the scenario from a sub folder extending another scenario
    scenario = Scenario.load_from_file(
        path_relative_to_tests(
            './testdata/scenarios/sub/traininglang_scenario_extended_again.yml'
        )
    )
    assert scenario.defender_settings['Defender1'].rewards
    rewards_per_node = scenario.defender_settings['Defender1'].rewards
    num_nodes_with_reward = 0
    for node in scenario.attack_graph.nodes.values():
        reward = rewards_per_node.value(node, 0)
        if reward:
            # All nodes with reward set should have reward 1
            # Since this is defined in the extended scenario
            num_nodes_with_reward += 1
            assert reward == 1
    assert num_nodes_with_reward == 7

    # 1 agents are defined in the extended_again scenario
    assert len(scenario.agent_settings) == 1


def test_extend_scenario_override_lang_model() -> None:
    """
    Make sure we can extend a scenario several levels
    and in different sub directory
    """

    # Load the scenario from a sub folder extending another scenario
    # that overrides lang and model
    scenario = Scenario.load_from_file(
        path_relative_to_tests(
            './testdata/scenarios/sub/traininglang_scenario_override_lang_model.yml'
        )
    )
    attackgraph = scenario.attack_graph

    # No reward overrides
    assert scenario.attacker_settings['Attacker1'].rewards
    rewards_per_node = scenario.attacker_settings['Attacker1'].rewards
    assert rewards_per_node[attackgraph.get_node_by_full_name('Host:0:notPresent')] == 2
    assert rewards_per_node[attackgraph.get_node_by_full_name('Host:0:access')] == 4
    assert rewards_per_node[attackgraph.get_node_by_full_name('Host:1:notPresent')] == 7
    assert rewards_per_node[attackgraph.get_node_by_full_name('Host:1:access')] == 5
    assert rewards_per_node[attackgraph.get_node_by_full_name('Data:2:notPresent')] == 8
    assert rewards_per_node[attackgraph.get_node_by_full_name('Data:2:read')] == 5
    assert rewards_per_node[attackgraph.get_node_by_full_name('Data:2:modify')] == 10

    # No agent overrides
    assert len(scenario.agent_settings) == 2


def test_load_scenario_no_defender_agent() -> None:
    """Make sure we can load a scenario"""

    # Load the scenario
    scenario = Scenario.load_from_file(
        path_relative_to_tests('./testdata/scenarios/no_defender_agent_scenario.yml')
    )
    assert 'defender' not in scenario.defender_settings
    assert isinstance(
        scenario.attacker_settings['attacker1'].agent, BreadthFirstAttacker
    )


def test_load_scenario_agent_class_error() -> None:
    """Make sure we get error when loading with wrong class"""

    # Load the scenario
    with pytest.raises(LookupError):
        x = Scenario.load_from_file(
            path_relative_to_tests(
                './testdata/scenarios/wrong_agent_classes_scenario.yml'
            )
        ).agent_settings
        assert x  # to avoid unused variable warning


def test_load_scenario_observability_given() -> None:
    """Load a scenario with observability settings given and
    make sure observability is applied correctly"""

    # Load scenario with observability specified
    scenario = Scenario.load_from_file(
        path_relative_to_tests(
            './testdata/scenarios/simple_filtered_observability_scenario.yml'
        )
    )

    # Make sure only attack steps of name fullAccess
    # part of asset type Application are observable.
    defender_agent = scenario.defender_settings['Defender1']
    assert isinstance(defender_agent, DefenderSettings)
    observable_steps = defender_agent.observable_steps
    assert observable_steps
    is_observable_per_node = observable_steps.per_node(scenario.attack_graph)

    for node in scenario.attack_graph.nodes.values():
        if (
            (
                node.lg_attack_step.asset.name == 'Application'
                and node.name == 'fullAccess'
            )
            or (
                node.lg_attack_step.asset.name == 'Application'
                and node.name == 'supplyChainAuditing'
            )
            or (
                node.model_asset
                and node.model_asset.name == 'Identity:8'
                and node.name == 'assume'
            )
        ):
            assert is_observable_per_node[node.full_name]
        else:
            assert node.full_name not in is_observable_per_node


def test_load_scenario_observability_not_given() -> None:
    """Load a scenario where no observability settings are given"""
    # Load scenario with no observability specifed
    scenario = Scenario.load_from_file(
        path_relative_to_tests('./testdata/scenarios/simple_scenario.yml')
    )
    defender_agent = scenario.defender_settings['Defender1']
    assert isinstance(defender_agent, DefenderSettings)
    assert not defender_agent.observable_steps


def test_apply_scenario_observability() -> None:
    """Try different cases for observability settings"""

    # Load scenario with no observability specified
    scenario = Scenario.load_from_file(
        path_relative_to_tests('testdata/scenarios/simple_scenario.yml')
    )

    # Make Data: read, write, delete observable
    # Make Application: fullAccess, notPresent observable
    observability_rules = {
        'by_asset_type': {
            'Data': ['read', 'write', 'delete'],
            'Application': ['fullAccess', 'notPresent'],
        },
        'by_asset_name': {'OS App': ['read']},
    }

    # Apply observability rules
    observable: NodePropertyRule[bool] | None = NodePropertyRule.from_optional_dict(
        observability_rules
    )
    assert observable
    observable_per_node = observable.per_node(scenario.attack_graph)

    # Make sure all attack steps are observable
    # if no observability settings are given
    for node in scenario.attack_graph.nodes.values():
        if (
            (
                node.lg_attack_step.asset.name == 'Data'
                and node.name
                in (
                    'read',
                    'write',
                    'delete',
                )
            )
            or (
                node.lg_attack_step.asset.name == 'Application'
                and node.name
                in (
                    'fullAccess',
                    'notPresent',
                )
            )
            or (
                node.model_asset
                and node.model_asset.name == 'OS App'
                and node.name in ('read')
            )
        ):
            assert observable_per_node[node.full_name]
        else:
            assert node.full_name not in observable_per_node


def test_apply_scenario_observability_faulty() -> None:
    """Try different failing cases for observability settings"""

    # Load scenario with no observability specified
    # scenario = Scenario.load_from_file(
    #     path_relative_to_tests('testdata/scenarios/simple_scenario.yml')
    # )

    # Wrong key in rule dict
    with pytest.raises(ValueError):
        NodePropertyRule.from_optional_dict(
            {'NotAllowedKey': {'Data': ['read', 'write', 'delete']}},
        )

        # Correct asset type and attack step
        NodePropertyRule.from_optional_dict(
            {'by_asset_type': {'Application': ['read']}},
        )

    # TODO: reintroduce validation

    # # Wrong asset type in rule asset type to step dict
    # with pytest.raises(ValueError):
    #     NodePropertyRule.from_dict(
    #         {'by_asset_type': {'NonExistingType': ['read']}},
    #     ).per_node(scenario.attack_graph)

    # # Wrong attack step name in rule asset type to step dict
    # with pytest.raises(AssertionError):
    #     NodePropertyRule.from_dict(
    #         {'by_asset_type': {'Data': ['nonExistingAttackStep']}},
    #     )

    # # Correct asset name and attack step
    # NodePropertyRule.from_dict({'by_asset_name': {'OS App': ['read']}})

    # # Wrong asset name in rule asset name to step dict
    # with pytest.raises(AssertionError):
    #     NodePropertyRule.from_dict({'by_asset_name': {'NonExistingName': ['read']}})

    # # Wrong attack step name in rule asset name to step dict
    # with pytest.raises(AssertionError):
    #     NodePropertyRule.from_dict(
    #         {'by_asset_name': {'OS App': ['nonExistingAttackStep']}},
    #     )


def test_load_scenario_false_positive_negative_rate() -> None:
    """Load a scenario with observability settings given and
    make sure observability is applied correctly"""

    # Load scenario with observability specifed
    scenario = Scenario.load_from_file(
        path_relative_to_tests('./testdata/scenarios/traininglang_fp_fn_scenario.yml')
    )

    # Defined in scenario file
    host_0_access_fp_rate = 0.2
    host_1_access_fp_rate = 0.3
    host_0_access_fn_rate = 0.4
    host_1_access_fn_rate = 0.5
    user_3_compromise_fn_rate = 1.0

    defender_settings = scenario.defender_settings['defender']

    assert isinstance(defender_settings, DefenderSettings)
    assert defender_settings.false_negative_rates
    assert defender_settings.false_positive_rates
    fpr_per_node = defender_settings.false_positive_rates
    fnr_per_node = defender_settings.false_negative_rates

    for node in scenario.attack_graph.nodes.values():
        if node.full_name == 'Host:0:access':
            # According to scenario file
            assert fpr_per_node[node] == host_0_access_fp_rate
            assert fnr_per_node[node] == host_0_access_fn_rate

        elif node.full_name == 'Host:1:access':
            # According to scenario file
            assert fpr_per_node[node] == host_1_access_fp_rate
            assert fnr_per_node[node] == host_1_access_fn_rate

        elif node.full_name == 'User:3:compromise':
            # According to scenario file
            assert node not in fpr_per_node
            assert fnr_per_node[node] == user_3_compromise_fn_rate

        else:
            # If no rules - don't set fpr/fnr
            assert node not in fpr_per_node
            assert node not in fnr_per_node


def test_apply_scenario_fpr_fnr() -> None:
    """Try different cases for false positives/negatives rates"""

    # Load scenario with no specified
    scenario = Scenario.load_from_file(
        path_relative_to_tests('testdata/scenarios/simple_scenario.yml')
    )

    property_values = {
        'by_asset_type': {
            'Data': {'read': 0.5, 'write': 0.6, 'delete': 0.7},
            'Application': {'fullAccess': 0.8, 'read': 1.0},
        },
        'by_asset_name': {
            'OS App': {
                'read': 0.9  # Has precedence
            }
        },
    }

    # Apply false negative rate rules
    false_negatives_rates: NodePropertyRule[float] | None = (
        NodePropertyRule.from_optional_dict(property_values)
    )
    assert false_negatives_rates
    fnr_per_node = false_negatives_rates.per_node(scenario.attack_graph)

    # Make sure all attack steps are observable
    # if no observability settings are given
    for node in scenario.attack_graph.nodes.values():
        if node.lg_attack_step.asset.name == 'Data' and node.name == 'read':
            assert fnr_per_node[node.full_name] == 0.5
        elif node.lg_attack_step.asset.name == 'Data' and node.name == 'write':
            assert fnr_per_node[node.full_name] == 0.6
        elif node.lg_attack_step.asset.name == 'Data' and node.name == 'delete':
            assert fnr_per_node[node.full_name] == 0.7
        elif (
            node.model_asset
            and node.model_asset.name == 'OS App'
            and node.name == 'read'
        ):
            assert fnr_per_node[node.full_name] == 0.9
        elif node.lg_attack_step.asset.name == 'Application' and node.name == 'read':
            assert fnr_per_node[node.full_name] == 1.0
        elif (
            node.lg_attack_step.asset.name == 'Application'
            and node.name == 'fullAccess'
        ):
            assert fnr_per_node[node.full_name] == 0.8
        else:
            assert node.full_name not in fnr_per_node


def test_apply_scenario_rewards_old_format() -> None:
    """Try different cases for rewards"""

    # Load scenario with no specified
    lang_file = 'tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar'
    model_file = 'tests/testdata/models/simple_test_model.yml'
    scenario_dict: dict[str, Any] = {
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
            'Identity:8:assume': 50,
        },
        # Add entry points to AttackGraph with attacker names
        # and attack step full_names
        'agents': {},
    }

    with pytest.raises(ValueError):
        # Make sure we get error when loading with wrong rewards format
        NodePropertyRule.from_optional_dict(scenario_dict['rewards'])


def test_scenario_pickle(tmp_path: Any) -> None:
    """Make sure we can pickle a scenario"""

    # Load the scenario
    scenario = Scenario.load_from_file(
        path_relative_to_tests('./testdata/scenarios/simple_scenario.yml')
    )

    with open(tmp_path / 'scenario.pkl', 'wb') as f:
        pickle.dump(scenario, f)

    with open(tmp_path / 'scenario.pkl', 'rb') as f:
        loaded_scenario = pickle.load(f)

    assert loaded_scenario.to_dict() == scenario.to_dict()


def test_scenario_advanced_agent_settings() -> None:
    """Verify:
    - scenario loads correctly using new format
    - agent settings are parsed
    - NodePropertyRule objects are created
    - pickling via to_dict()/from_dict() works
    """

    scenario = Scenario.load_from_file(
        path_relative_to_tests(
            './testdata/scenarios/traininglang_scenario_advanced_agent_settings.yml'
        )
    )

    # --- round-trip check ---
    copied = Scenario.from_dict(scenario.to_dict())
    assert scenario.to_dict() == copied.to_dict()

    # --- language and model file paths ---
    assert scenario._lang_file.endswith('langs/org.mal-lang.trainingLang-1.0.0.mar')
    assert scenario._model_file
    assert scenario._model_file.endswith('models/traininglang_model.yml')

    # --- attacker rewards ---
    assert isinstance(scenario.attacker_settings['Attacker1'].rewards, NodePropertyRule)
    attacker_rewards = scenario.attacker_settings['Attacker1'].rewards.by_asset_name
    assert attacker_rewards
    assert attacker_rewards['Host:0']['access'] == 4
    assert attacker_rewards['Host:1']['access'] == 5
    assert attacker_rewards['Data:2']['read'] == 5
    assert attacker_rewards['Data:2']['modify'] == 10

    # --- agents ---

    assert 'Attacker1' in scenario.attacker_settings
    assert 'Defender1' in scenario.defender_settings

    attacker = scenario.attacker_settings['Attacker1']
    defender = scenario.defender_settings['Defender1']

    # -----------------------
    # Attacker1
    # -----------------------
    assert isinstance(attacker, AttackerSettings)
    assert attacker.type == AgentType.ATTACKER

    attack_graph = scenario.attack_graph

    # entry points
    assert {n.full_name for n in attacker.entry_points} == {
        'User:3:phishing',
        'Host:0:connect',
    }

    # policy
    assert attacker.policy is BreadthFirstAttacker

    # actionable_steps
    assert isinstance(attacker.actionable_steps, NodePropertyRule)
    assert attacker.actionable_steps.by_asset_type == {
        'Host': {'authenticate': True, 'connect': True},
        'User': {'compromise': True},
    }

    # observable_steps
    assert attacker.rewards is not None
    assert attacker.rewards.by_asset_name
    assert attacker.rewards.by_asset_name['Host:0']['authenticate'] == 1000

    # -----------------------
    # Defender1
    # -----------------------
    assert isinstance(defender, DefenderSettings)
    assert defender.type == AgentType.DEFENDER

    # actionable / observable
    assert isinstance(defender.actionable_steps, NodePropertyRule)
    assert isinstance(defender.observable_steps, NodePropertyRule)

    assert (
        defender.actionable_steps.by_asset_type
        and defender.actionable_steps.by_asset_type == {'Host': {'notPresent': True}}
    )

    # FN/FP rates
    assert isinstance(defender.false_positive_rates, NodePropertyRule)
    assert isinstance(defender.false_negative_rates, NodePropertyRule)

    assert defender.false_negative_rates.by_asset_type
    assert defender.false_negative_rates.by_asset_type['Host']['access'] == 0.5
    assert defender.false_positive_rates.by_asset_type
    assert defender.false_positive_rates.by_asset_type['Host']['connect'] == 0.5

    # Rewards (defender has none in file)
    assert defender.rewards is not None
    assert defender.rewards.by_asset_name
    assert defender.rewards.by_asset_name['Host:0']['notPresent'] == 100
