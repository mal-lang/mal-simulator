"""Test MalSimulator class"""
from __future__ import annotations
from typing import TYPE_CHECKING
import math

from maltoolbox.attackgraph import AttackGraphNode, AttackGraph
from malsim.mal_simulator import (
    MalSimulator,
    MalSimulatorSettings,
    MalSimDefenderState,
    MalSimAttackerState,
    TTCMode,
    RewardMode
)
from malsim.scenario import load_scenario
from .conftest import get_node

if TYPE_CHECKING:
    from maltoolbox.language import LanguageGraph
    from maltoolbox.model import Model

def test_init(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    MalSimulator(attack_graph)


def test_reset(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    """Make sure attack graph is reset"""
    attack_graph = AttackGraph(corelang_lang_graph, model)

    agent_entry_point = get_node(attack_graph, 'OS App:localConnect')
    attacker_name = "testagent"

    sim = MalSimulator(attack_graph, sim_settings=MalSimulatorSettings(seed=10))

    viability_before = {
        n.full_name: v for n, v in sim._viability_per_node.items()
    }
    necessity_before = {
        n.full_name: v for n, v in sim._necessity_per_node.items()
    }
    enabled_defenses = {
        n.full_name for n in sim._enabled_defenses
    }
    sim.register_attacker(attacker_name, {agent_entry_point})
    assert attacker_name in sim.agent_states
    assert len(sim.agent_states) == 1
    attacker_state = sim.agent_states[attacker_name]
    action_surface_before = {
        n.full_name for n in attacker_state.action_surface
    }

    sim.reset()

    attacker_state = sim.agent_states[attacker_name]
    assert action_surface_before == {
        n.full_name for n in attacker_state.action_surface
    }

    for node, viable in sim._viability_per_node.items():
        # viability is the same after reset
        assert viability_before[node.full_name] == viable

    for node, necessary in sim._necessity_per_node.items():
        # viability is the same after reset
        assert necessity_before[node.full_name] == necessary

    assert enabled_defenses == {
        n.full_name for n in sim._enabled_defenses
    }

    sim.reset()
    attacker_state = sim.agent_states[attacker_name]
    assert action_surface_before == {
        n.full_name for n in attacker_state.action_surface
    }

    # Step with action surface
    sim.step({attacker_name: list(attacker_state.action_surface)})

    # Make sure action surface back to normal
    sim.reset()
    attacker_state = sim.agent_states[attacker_name]
    assert action_surface_before == {
        n.full_name for n in attacker_state.action_surface
    }


def test_register_agent_attacker(
        corelang_lang_graph: LanguageGraph, model: Model
    ) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    agent_name = "attacker1"
    sim.register_attacker(agent_name, set())

    assert agent_name in sim.agent_states
    assert agent_name in sim.agent_states


def test_register_agent_defender(
        corelang_lang_graph: LanguageGraph, model: Model
    ) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    agent_name = "defender1"
    sim.register_defender(agent_name)

    assert agent_name in sim.agent_states
    assert agent_name in sim.agent_states


def test_register_agent_action_surface(
        corelang_lang_graph: LanguageGraph, model: Model
    ) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    agent_name = "defender1"
    sim.register_defender(agent_name)

    defender_state = sim.agent_states[agent_name]
    action_surface = defender_state.action_surface
    for node in action_surface:
        assert node not in sim._enabled_defenses


def test_simulator_initialize_agents(
        corelang_lang_graph: LanguageGraph, model: Model
    ) -> None:
    """Test _initialize_agents"""

    scenario = load_scenario('tests/testdata/scenarios/simple_scenario.yml')
    sim = MalSimulator.from_scenario(scenario, register_agents=False)

    # Register the agents
    attacker_name = "attacker"
    defender_name = "defender"
    sim.register_attacker(attacker_name, set())
    sim.register_defender(defender_name)

    sim.reset()

    assert set(sim.agent_states.keys()) == {attacker_name, defender_name}


def test_get_agents() -> None:
    """Test _get_attacker_agents and _get_defender_agents"""

    scenario = load_scenario('tests/testdata/scenarios/simple_scenario.yml')
    sim = MalSimulator.from_scenario(scenario)
    sim.reset()

    assert [a.name for a in sim._get_attacker_agents()] == ['Attacker1']
    assert [a.name for a in sim._get_defender_agents()] == ['Defender1']


def test_attacker_step(
        corelang_lang_graph: LanguageGraph, model: Model
    ) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = get_node(attack_graph, 'OS App:fullAccess')

    sim = MalSimulator(attack_graph)

    attacker_name = "Test Attacker"
    sim.register_attacker(attacker_name, {entry_point})
    sim.reset()

    attacker_agent = sim._agent_states[attacker_name]
    assert isinstance(attacker_agent, MalSimAttackerState)

    # Can not attack the notPresent step
    defense_step = get_node(attack_graph, 'OS App:notPresent')
    actions, _ = sim._attacker_step(attacker_agent, [defense_step])
    assert not actions

    attack_step = get_node(attack_graph, 'OS App:attemptRead')
    actions, _ = sim._attacker_step(attacker_agent, [attack_step])
    assert actions  == {attack_step}


def test_defender_step(corelang_lang_graph: LanguageGraph, model: Model) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    defender_name = "defender"
    sim.register_defender(defender_name)
    sim.reset()

    defender_agent = sim._agent_states[defender_name]
    assert isinstance(defender_agent, MalSimDefenderState)

    defense_step = get_node(attack_graph, 'OS App:notPresent')
    enabled, made_unviable = sim._defender_step(defender_agent, [defense_step])
    assert enabled ==  {defense_step}
    assert made_unviable

    # Can not defend attack_step
    attack_step = get_node(attack_graph, 'OS App:attemptUseVulnerability')
    assert attack_step
    enabled, made_unviable = sim._defender_step(defender_agent, [attack_step])
    assert enabled == set()
    assert not made_unviable


def test_attacker_step_rewards_cumulative(
        corelang_lang_graph: LanguageGraph, model: Model
    ) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)

    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    attempt_read = get_node(attack_graph, 'OS App:attemptRead')
    access_network_and_conn = get_node(
        attack_graph, 'OS App:accessNetworkAndConnections'
    )

    node_rewards = {
        entry_point: 10.0,
        attempt_read: 100.0,
        access_network_and_conn: 50.4
    }
    sim = MalSimulator(attack_graph, node_rewards=node_rewards)

    attacker_name = "Test Attacker"
    sim.register_attacker(attacker_name, {entry_point})
    sim.reset()

    states = sim.step({attacker_name: [attempt_read]})
    attacker_state = states[attacker_name]
    assert sim.agent_reward(attacker_state.name) == (
        node_rewards[entry_point] + node_rewards[attempt_read]
    )

    states = sim.step({attacker_name: [access_network_and_conn]})
    attacker_state = states[attacker_name]
    assert sim.agent_reward(attacker_state.name) == (
        node_rewards[entry_point]
        + node_rewards[attempt_read]
        + node_rewards[access_network_and_conn]
    )

    # Recording of the simulation
    assert sim.recording == {
        0: {
            attacker_name: [attempt_read]
        },
        1: {
            attacker_name: [access_network_and_conn]
        }
    }

def test_attacker_step_rewards_one_off(
        corelang_lang_graph: LanguageGraph, model: Model
    ) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)

    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    attempt_read = get_node(attack_graph, 'OS App:attemptRead')
    access_network_and_conn = get_node(
        attack_graph, 'OS App:accessNetworkAndConnections'
    )

    node_rewards = {
        entry_point: 10.0,
        attempt_read: 100.0,
        access_network_and_conn: 50.4
    }
    sim = MalSimulator(
        attack_graph,
        node_rewards=node_rewards,
        sim_settings=MalSimulatorSettings(
            attacker_reward_mode=RewardMode.ONE_OFF
        )
    )

    attacker_name = "Test Attacker"
    sim.register_attacker(attacker_name, {entry_point})
    sim.reset()

    sim.step({attacker_name: [attempt_read]})
    assert sim.agent_reward(attacker_name) == node_rewards[attempt_read]

    sim.step({attacker_name: [access_network_and_conn]})
    assert sim.agent_reward(attacker_name) == node_rewards[access_network_and_conn]


def test_defender_step_rewards_cumulative(
        corelang_lang_graph: LanguageGraph, model: Model
    ) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)

    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    attempt_read = get_node(attack_graph, 'OS App:attemptRead')
    not_present = get_node(attack_graph, 'OS App:notPresent')
    access_network_and_conn = get_node(
        attack_graph, 'OS App:accessNetworkAndConnections'
    )

    node_rewards = {
        not_present: 100,
        entry_point: 10.0,
        attempt_read: 105.0,
        access_network_and_conn: 35.04
    }
    sim = MalSimulator(attack_graph, node_rewards=node_rewards)

    defender_name = "defender"
    sim.register_defender(defender_name)
    attacker_name = "Test Attacker" # To be able to step
    sim.register_attacker(attacker_name, {entry_point})
    sim.reset()

    sim.step({
        attacker_name: [attempt_read]
    })
    assert sim.agent_reward(defender_name) == - (
        node_rewards[entry_point] + node_rewards[attempt_read]
    )

    sim.step({
        attacker_name: [access_network_and_conn]
    })
    assert sim.agent_reward(defender_name) == - (
        node_rewards[entry_point]
        + node_rewards[attempt_read]
        + node_rewards[access_network_and_conn]
    )

def test_defender_step_rewards_one_off(
        corelang_lang_graph: LanguageGraph, model: Model
    ) -> None:
    attack_graph = AttackGraph(corelang_lang_graph, model)

    entry_point = get_node(attack_graph, 'OS App:fullAccess')
    attempt_read = get_node(attack_graph, 'OS App:attemptRead')
    not_present = get_node(attack_graph, 'OS App:notPresent')
    access_network_and_conn = get_node(
        attack_graph, 'OS App:accessNetworkAndConnections'
    )

    node_rewards = {
        not_present: 100,
        entry_point: 10.0,
        attempt_read: 105.0,
        access_network_and_conn: 35.04
    }
    sim = MalSimulator(
        attack_graph,
        node_rewards=node_rewards,
        sim_settings=MalSimulatorSettings(
            defender_reward_mode=RewardMode.ONE_OFF
        )
    )

    defender_name = "defender"
    sim.register_defender(defender_name)
    attacker_name = "Test Attacker" # To be able to step
    sim.register_attacker(attacker_name, {entry_point})
    sim.reset()

    states = sim.step({
        attacker_name: [attempt_read]
    })
    defender_state = states[defender_name]
    assert sim.agent_reward(defender_state.name) == - node_rewards[attempt_read]

    states = sim.step({
        attacker_name: [access_network_and_conn]
    })
    defender_state = states[defender_name]
    assert sim.agent_reward(defender_state.name) == - node_rewards[access_network_and_conn]


# TODO: Some of the assert values in this test have changed when updating the
# attacker logic. We should check to see if the behaviour is the new behaviour
# is correct.
def test_agent_state_views_simple(corelang_lang_graph: LanguageGraph, model: Model) -> None:

    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = get_node(attack_graph, 'OS App:fullAccess')

    mss = MalSimulatorSettings(
        seed=13,
        ttc_mode=TTCMode.PER_STEP_SAMPLE
    )
    # Create simulator and register agents
    sim = MalSimulator(attack_graph, sim_settings=mss)
    attacker_name = 'attacker'
    defender_name = 'defender'
    sim.register_attacker(attacker_name, {entry_point})
    sim.register_defender(defender_name)

    # Evaluate the agent state views after reset
    state_views = sim.reset()
    entry_point = get_node(attack_graph, 'OS App:fullAccess')

    pre_enabled_defenses = set(sim._enabled_defenses)

    asv = state_views['attacker']
    dsv = state_views['defender']

    assert asv.step_performed_nodes == {entry_point}
    assert dsv.step_performed_nodes == pre_enabled_defenses

    assert asv.performed_nodes == {entry_point}
    assert dsv.performed_nodes == pre_enabled_defenses

    assert len(asv.action_surface) == 6
    assert set(n.full_name for n in dsv.action_surface) == {
        'Credentials:10:notPhishable',        # Disabled in lang
        'Data:5:notPresent',                  # Disabled in lang
        'Credentials:9:unique',               # Enabled in lang, Disabled in model
        'User:12:noPasswordReuse',            # Enabled in lang, Disabled in model
        'Group:13:notPresent',                # Disabled in lang
        'IDPS 1:notPresent',                  # Disabled in lang
        'OS App:supplyChainAuditing',         # Not set in lang, Disabled by default
        'OS App:notPresent',                  # Disabled in lang
        'Credentials:6:unique',               # Enabled in lang, Disabled in model
        'Program 2:notPresent',               # Disabled in lang
        'Credentials:9:notPhishable',         # Disabled in lang
        'Program 2:supplyChainAuditing',      # Not set in lang, Disabled by default
        'User:12:securityAwareness',          # Not set in lang, Disabled by default
        'Identity:11:notPresent',             # Disabled in lang
        'Credentials:7:notPhishable',         # Disabled in lang
        'Identity:8:notPresent',              # Disabled in lang
        'Credentials:7:unique',               # Enabled in lang, Disabled in model
        'Credentials:6:notPhishable',         # Disabled in lang
        'IDPS 1:supplyChainAuditing',         # Not set in lang, Disabled by default
        'SoftwareVulnerability:4:notPresent', # Disabled in lang
        'Program 1:supplyChainAuditing'       # Not set in lang, Disabled by default
    }

    assert len(dsv.step_action_surface_additions) == len(dsv.action_surface)
    assert len(asv.step_action_surface_additions) == len(asv.action_surface)

    assert asv.step_action_surface_removals == set()
    assert dsv.step_action_surface_removals == set()

    # Save all relvant nodes in variables
    program2_not_present = get_node(attack_graph, 'Program 2:notPresent')
    os_app_attempt_deny = get_node(attack_graph, 'OS App:attemptDeny')
    os_app_success_deny = get_node(attack_graph, 'OS App:successfulDeny')
    os_app_not_present = get_node(attack_graph, 'OS App:notPresent')
    os_app_access_netcon = get_node(attack_graph, 'OS App:accessNetworkAndConnections')
    os_app_spec_access = get_node(attack_graph, 'OS App:specificAccess')

    # Evaluate the agent state views after stepping through an attack step and
    # a defense that will not impact it in any way

    state_views = sim.step({
        'defender': [program2_not_present],
        'attacker': [os_app_attempt_deny]
    })
    asv = state_views['attacker']
    dsv = state_views['defender']
    assert isinstance(asv, MalSimAttackerState)
    assert isinstance(dsv, MalSimDefenderState)

    assert asv.step_performed_nodes == {os_app_attempt_deny}
    assert dsv.step_performed_nodes == {program2_not_present}

    assert asv.performed_nodes == {os_app_attempt_deny, entry_point}
    assert dsv.performed_nodes == pre_enabled_defenses | {program2_not_present}

    assert asv.step_action_surface_additions == {os_app_success_deny}
    assert dsv.step_action_surface_additions == set()
    assert asv.step_action_surface_removals == {os_app_attempt_deny}
    assert os_app_attempt_deny not in asv.action_surface
    assert dsv.step_action_surface_removals == {program2_not_present}
    assert dsv.step_all_compromised_nodes == {os_app_attempt_deny}
    assert len(dsv.step_unviable_nodes) == 47

    # Go through an attack step that already has some children in the attack
    # surface(OS App:accessNetworkAndConnections in this case)
    assert os_app_access_netcon in asv.action_surface
    state_views = sim.step({
        'defender': [],
        'attacker': [os_app_spec_access]
    })
    asv = state_views['attacker']
    dsv = state_views['defender']
    assert isinstance(asv, MalSimAttackerState)
    assert isinstance(dsv, MalSimDefenderState)

    assert asv.step_performed_nodes == {os_app_spec_access}
    assert dsv.step_performed_nodes == set()
    assert os_app_access_netcon in asv.action_surface
    assert os_app_access_netcon not in asv.step_action_surface_additions
    assert dsv.step_action_surface_additions == set()
    assert asv.step_action_surface_removals == {os_app_spec_access}
    assert os_app_spec_access not in asv.action_surface
    assert dsv.step_action_surface_removals == set()
    assert dsv.step_all_compromised_nodes == {os_app_spec_access}
    assert len(dsv.step_unviable_nodes) == 0

    # Evaluate the agent state views after stepping through an attack step and
    # a defense that would prevent it from occurring
    state_views = sim.step({
        'defender': [os_app_not_present],
        'attacker': [os_app_success_deny]
    })
    asv = state_views['attacker']
    dsv = state_views['defender']
    assert isinstance(asv, MalSimAttackerState)
    assert isinstance(dsv, MalSimDefenderState)

    assert asv.step_performed_nodes == set()
    assert dsv.step_performed_nodes == {os_app_not_present}
    assert asv.step_action_surface_additions == set()
    assert dsv.step_action_surface_additions == set()
    assert {a.full_name for a in asv.step_action_surface_removals} == {
        'OS App:accessNetworkAndConnections',
        'OS App:attemptApplicationRespondConnectThroughData',
        'OS App:attemptAuthorizedApplicationRespondConnectThroughData',
        'OS App:attemptModify',
        'OS App:attemptRead',
        'OS App:specificAccessDelete',
        'OS App:specificAccessModify',
        'OS App:specificAccessRead',
        'OS App:successfulDeny',
        'Program 1:localConnect',
        'Program 2:localConnect',
        'IDPS 1:localConnect'
    }
    assert len(asv.step_action_surface_removals) == 12
    assert dsv.step_action_surface_removals == {os_app_not_present}
    assert dsv.step_all_compromised_nodes == set()
    assert len(dsv.step_unviable_nodes) == 63

    # Recording of the simulation
    assert sim.recording == {
        0: {
            'defender': [program2_not_present],
            'attacker': [os_app_attempt_deny]
        },
        1: {
            'defender': [],
            'attacker': [os_app_spec_access]
        },
        2: {
            'defender': [os_app_not_present],
            'attacker': []
        }
    }


def test_step_attacker_defender_action_surface_updates() -> None:
    scenario = load_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml')

    sim = MalSimulator.from_scenario(scenario)
    # Register the agents
    attacker_agent_id = "attacker"
    defender_agent_id = "defender"

    user3_phishing = get_node(sim.attack_graph, 'User:3:phishing')
    host0_connect = get_node(sim.attack_graph, 'Host:0:connect')
    sim.register_attacker(
        attacker_agent_id,
        set([user3_phishing, host0_connect])
    )
    sim.register_defender(defender_agent_id)

    states = sim.reset()

    attacker_agent = states[attacker_agent_id]
    defender_agent = states[defender_agent_id]

    # Run step() with action crafted in test
    attacker_step = sim.attack_graph.get_node_by_full_name('User:3:compromise')
    assert attacker_step in attacker_agent.action_surface

    defender_step = sim.attack_graph.get_node_by_full_name('User:3:notPresent')
    assert defender_step in defender_agent.action_surface

    actions = {
        attacker_agent.name: [attacker_step],
        defender_agent.name: [defender_step]
    }

    states = sim.step(actions)
    attacker_agent = states[attacker_agent_id]
    defender_agent = states[defender_agent_id]

    # Make sure no nodes added to action surface
    assert not attacker_agent.step_action_surface_additions
    assert not defender_agent.step_action_surface_additions

    # Make sure the steps are removed from the action surfaces
    assert attacker_step in attacker_agent.step_action_surface_removals
    assert defender_step in defender_agent.step_action_surface_removals

    assert attacker_step not in attacker_agent.action_surface
    assert defender_step not in defender_agent.action_surface


def test_default_simulator_default_settings_eviction() -> None:
    """Test attacker node eviction using MalSimulatorSettings default"""

    scenario = load_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml'
    )
    sim = MalSimulator.from_scenario(scenario)

    # Register the agents
    attacker_agent_id = "Attacker1"
    defender_agent_id = "Defender1"

    # sim.register_attacker(attacker_agent_id, set())
    # sim.register_defender(defender_agent_id)
    sim.reset()
    attacker_agent = sim.agent_states[attacker_agent_id]
    defender_agent = sim.agent_states[defender_agent_id]

    # Get a step to compromise and its defense parent
    user_3_compromise = get_node(sim.attack_graph, 'User:3:compromise')
    user_3_compromise_defense = next(n for n in user_3_compromise.parents if n.type=='defense')
    assert user_3_compromise not in attacker_agent.performed_nodes
    assert user_3_compromise_defense not in defender_agent.performed_nodes

    # First let the attacker compromise User:3:compromise
    actions = {
        attacker_agent_id: [user_3_compromise],
        defender_agent_id: []
    }
    states = sim.step(actions)

    # Check that the compromise happened and that the defense did not
    assert user_3_compromise in states[attacker_agent_id].performed_nodes
    assert user_3_compromise_defense not in states[defender_agent_id].performed_nodes

    # Now let the defender defend, and the attacker waits
    actions = {
        attacker_agent_id: [],
        defender_agent_id: [user_3_compromise_defense]
    }
    states = sim.step(actions)

    # Verify defense was performed and attacker NOT kicked out
    assert user_3_compromise in states[attacker_agent_id].performed_nodes
    assert user_3_compromise_defense in states[defender_agent_id].performed_nodes


def test_simulator_ttcs() -> None:
    """Create a simulator and check TTCs, then reset and check TTCs again"""

    def get_node(sim: MalSimulator, full_name: str) -> AttackGraphNode:
        node = sim.attack_graph.get_node_by_full_name(full_name)
        assert node
        return node

    scenario = load_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml'
    )
    sim = MalSimulator.from_scenario(
        scenario, sim_settings=MalSimulatorSettings(
            ttc_mode=TTCMode.PER_STEP_SAMPLE
        )
    )

    host_0_notPresent = get_node(sim, "Host:0:notPresent")
    host_0_auth = get_node(sim, "Host:0:authenticate")
    host_0_connect = get_node(sim, "Host:0:connect")
    host_0_access = get_node(sim, "Host:0:access")
    host_1_notPresent = get_node(sim, "Host:1:notPresent")
    host_1_auth = get_node(sim, "Host:1:authenticate")
    host_1_connect = get_node(sim, "Host:1:connect")
    host_1_access = get_node(sim, "Host:1:access")
    data_2_notPresent = get_node(sim, "Data:2:notPresent")
    data_2_read = get_node(sim, "Data:2:read")
    data_2_modify = get_node(sim, "Data:2:modify")
    user_3_notPresent = get_node(sim, "User:3:notPresent")
    user_3_compromise = get_node(sim, "User:3:compromise")
    user_3_phishing = get_node(sim, "User:3:phishing")
    network_3_access = get_node(sim, "Network:3:access")

    expected_bernoullis = {
        host_0_notPresent: math.inf,
        host_0_auth: 1.0,
        host_0_connect: 1.0,
        host_0_access: 1.0,
        host_1_notPresent: math.inf,
        host_1_auth: 1.0,
        host_1_connect: 1.0,
        host_1_access: 1.0,
        data_2_notPresent: math.inf,
        data_2_read: 1.0,
        data_2_modify: 1.0,
        user_3_notPresent: math.inf,
        user_3_compromise: 1.0,
        user_3_phishing: 1.0,
        network_3_access: 1.0
    }

    assert sim._calculated_bernoullis == expected_bernoullis

    sim.reset()

    assert sim._calculated_bernoullis == expected_bernoullis
