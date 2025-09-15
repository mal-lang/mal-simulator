"""
Run a scenario and make sure expected actions are chosen and
expected reward is given to agents.

These tests are to make sure the whole simulator maintains expected behavior.
Determinism, ttcs, agents, action surfaces, step etc.
"""

from malsim.scenario import create_simulator_from_scenario
from malsim.mal_simulator import (
    MalSimulatorSettings,
    TTCMode,
    MalSimDefenderState
)

def test_bfs_vs_bfs_state_and_reward() -> None:
    """
    The point of this test is to see that a specific
    scenario runs deterministically.

    The test creates a simulator, two agents and runs them both with
    BFS Agents against each other.

    It then verifies that rewards and actions performed are what we expected.
    """

    sim, agents = create_simulator_from_scenario(
        "tests/testdata/scenarios/bfs_vs_bfs_network_app_data_scenario.yml",
        sim_settings = MalSimulatorSettings(
            attack_surface_skip_unnecessary=False,
            run_defense_step_bernoullis=False,
            infinity_ttc_attack_step_unviable=False
        )
    )
    defender_agent_name = "defender1"
    attacker_agent_name = "attacker1"

    attacker_agent_info = next(
        agent for agent in agents if agent["name"] == attacker_agent_name
    )
    defender_agent_info = next(
        agent for agent in agents if agent["name"] == defender_agent_name
    )
    attacker_agent = attacker_agent_info["agent"]
    defender_agent = defender_agent_info["agent"]

    total_reward_defender = 0.0
    total_reward_attacker = 0.0

    attacker_actions: list[str] = []
    defender_actions: list[str] = []

    states = sim.reset()

    attacker_state = states[attacker_agent_name]
    defender_state = states[defender_agent_name]

    while not sim.done():
        # Run the simulation until agents are terminated/truncated
        attacker_node = attacker_agent.get_next_action(attacker_state)
        defender_node = defender_agent.get_next_action(defender_state)

        # Step
        actions = {
            defender_agent_name: [defender_node] if defender_node else [],
            attacker_agent_name: [attacker_node] if attacker_node else []
        }
        states = sim.step(actions)
        attacker_state = states[attacker_agent_name]
        defender_state = states[defender_agent_name]
        assert isinstance(defender_state, MalSimDefenderState)

        # If actions were performed, add them to respective list
        if attacker_node and attacker_node in attacker_state.step_performed_nodes:
            attacker_actions.append(attacker_node.full_name)
            assert attacker_node in defender_state.step_all_compromised_nodes

        if defender_node and defender_node in defender_state.step_performed_nodes:
            defender_actions.append(defender_node.full_name)

        total_reward_defender += sim.agent_reward(defender_state.name)
        total_reward_attacker += sim.agent_reward(attacker_state.name)

    assert sim.cur_iter == 44

    # Make sure the actions performed were as expected
    assert attacker_actions == [
        'Internet:attemptReverseReach',
        'Internet:networkForwardingUninspected',
        'Internet:deny',
        'Internet:accessNetworkData',
        'ConnectionRule Internet->Linux '
        'System:attemptConnectToApplicationsUninspected',
        'Internet:reverseReach',
        'Internet:networkForwardingInspected',
        'ConnectionRule Internet->Linux System:attemptAccessNetworksUninspected',
        'ConnectionRule Internet->Linux System:attemptDeny',
        'Internet:attemptEavesdrop',
        'Internet:attemptAdversaryInTheMiddle',
        'ConnectionRule Internet->Linux System:bypassRestricted',
        'ConnectionRule Internet->Linux System:bypassPayloadInspection',
        'ConnectionRule Internet->Linux System:connectToApplicationsUninspected',
        'ConnectionRule Internet->Linux System:attemptReverseReach',
        'ConnectionRule Internet->Linux System:attemptAccessNetworksInspected',
        'ConnectionRule Internet->Linux '
        'System:attemptConnectToApplicationsInspected',
        'ConnectionRule Internet->Linux System:successfulAccessNetworksUninspected',
        'ConnectionRule Internet->Linux System:deny',
        'Internet:bypassEavesdropDefense',
        'Internet:successfulEavesdrop',
        'Internet:bypassAdversaryInTheMiddleDefense',
        'Internet:successfulAdversaryInTheMiddle',
        'ConnectionRule Internet->Linux System:restrictedBypassed',
        'ConnectionRule Internet->Linux System:payloadInspectionBypassed',
        'Linux system:networkConnectUninspected',
        'Linux system:networkConnectInspected',
        'ConnectionRule Internet->Linux System:reverseReach',
        'ConnectionRule Internet->Linux System:successfulAccessNetworksInspected',
        'ConnectionRule Internet->Linux System:connectToApplicationsInspected',
        'ConnectionRule Internet->Linux System:accessNetworksUninspected',
        'Linux system:denyFromNetworkingAsset',
        'Internet:eavesdropDefenseBypassed',
        'Internet:eavesdrop',
        'Internet:adversaryInTheMiddleDefenseBypassed',
        'Internet:adversaryInTheMiddle',
        'Linux system:attemptUseVulnerability',
        'Linux system:networkConnect',
        'Linux system:specificAccessNetworkConnect',
        'Linux system:softwareProductVulnerabilityNetworkAccessAchieved',
        'Linux system:attemptReverseReach',
        'ConnectionRule Internet->Linux System:accessNetworksInspected',
        'Linux system:attemptDeny',
        'Internet:accessInspected'
    ]

    assert defender_actions == [
        'Linux system:notPresent',
        'Linux system:supplyChainAuditing',
        'Internet:networkAccessControl',
        'Internet:eavesdropDefense',
        'Internet:adversaryInTheMiddleDefense',
        'ConnectionRule Internet->Linux System:restricted',
        'ConnectionRule Internet->Linux System:payloadInspection',
        'Secret data:notPresent',
        'SoftwareVuln:notPresent'
    ]
    for step_id in attacker_actions:
        # Make sure that all attacker actions led to compromise
        node = sim.attack_graph.get_node_by_full_name(step_id)
        assert node in attacker_state.performed_nodes

    for step_id in defender_actions:
        # Make sure that all defender actions let to defense enabled
        node = sim.attack_graph.get_node_by_full_name(step_id)
        assert node in defender_state.performed_nodes

    # Verify rewards in latest run and total rewards
    assert sim.agent_reward(attacker_state.name) == 0
    assert sim.agent_reward(defender_state.name) == -50

    assert total_reward_attacker == 0
    assert total_reward_defender == -2200


def test_bfs_vs_bfs_state_and_reward_per_step_ttc() -> None:
    """
    The point of this test is to see that the basic scenario runs
    deterministically with ttcs.

    The test creates a simulator, two agents and runs them both with
    BFS Agents against each other.

    It then verifies that rewards and actions performed are what we expected.
    """

    sim, agents = create_simulator_from_scenario(
        "tests/testdata/scenarios/bfs_vs_bfs_scenario.yml",
        sim_settings = MalSimulatorSettings(
            seed=100,
            ttc_mode=TTCMode.PER_STEP_SAMPLE
        )
    )

    defender_agent_name = "defender1"
    attacker_agent_name = "attacker1"

    attacker_agent_info = next(
        agent for agent in agents if agent["name"] == attacker_agent_name
    )
    defender_agent_info = next(
        agent for agent in agents if agent["name"] == defender_agent_name
    )

    attacker_agent = attacker_agent_info["agent"]
    defender_agent = defender_agent_info["agent"]

    total_reward_defender = 0.0
    total_reward_attacker = 0.0

    attacker_actions = []
    defender_actions = []

    states = sim.reset()
    attacker_state = states[attacker_agent_name]
    defender_state = states[defender_agent_name]

    while not sim.done():
        # Run the simulation until agents are terminated/truncated
        attacker_node = attacker_agent.get_next_action(attacker_state)
        defender_node = defender_agent.get_next_action(defender_state)

        # Step
        actions = {
            defender_agent_name: [defender_node] if defender_node else [],
            attacker_agent_name: [attacker_node] if attacker_node else []
        }
        states = sim.step(actions)
        attacker_state = states[attacker_agent_name]
        defender_state = states[defender_agent_name]
        assert isinstance(defender_state, MalSimDefenderState)

        # If actions were performed, add them to respective list
        if attacker_node and attacker_node in attacker_state.step_performed_nodes:
            attacker_actions.append(attacker_node.full_name)
            assert attacker_node in defender_state.step_all_compromised_nodes

        if defender_node and defender_node in \
                states['defender1'].step_performed_nodes:
            defender_actions.append(defender_node.full_name)

        total_reward_defender += sim.agent_reward(defender_state.name)
        total_reward_attacker += sim.agent_reward(attacker_state.name)

    assert sim.cur_iter == 637

    # Make sure the actions performed were as expected
    assert attacker_actions == [
        'Program 1:attemptApplicationRespondConnectThroughData',
        'Program 1:attemptRead',
        'Program 1:attemptDeny',
        'Program 1:accessNetworkAndConnections',
        'Program 1:attemptModify',
        'Program 1:specificAccess',
        'ConnectionRule:1:attemptAccessNetworksUninspected',
        'ConnectionRule:1:attemptConnectToApplicationsUninspected',
        'ConnectionRule:1:attemptAccessNetworksInspected',
        'ConnectionRule:1:attemptConnectToApplicationsInspected',
        'ConnectionRule:1:bypassRestricted',
        'ConnectionRule:1:successfulAccessNetworksInspected',
        'ConnectionRule:1:connectToApplicationsInspected',
        'ConnectionRule:1:accessNetworksInspected',
        'Program 1:networkConnectInspected',
        'Network:2:accessInspected',
        'Program 1:specificAccessNetworkConnect',
        'Program 1:networkConnect',
        'Network:2:accessNetworkData',
        'Network:2:networkForwardingInspected',
        'Network:2:deny',
        'ConnectionRule:3:attemptConnectToApplicationsInspected',
        'Network:2:attemptEavesdrop',
        'Network:2:attemptAdversaryInTheMiddle',
        'ConnectionRule:3:attemptAccessNetworksInspected',
        'ConnectionRule:1:attemptDeny',
        'ConnectionRule:3:attemptDeny',
        'ConnectionRule:3:bypassPayloadInspection',
        'ConnectionRule:3:bypassRestricted',
        'ConnectionRule:3:connectToApplicationsInspected',
        'Network:2:successfulEavesdrop',
        'Network:2:bypassEavesdropDefense',
        'Network:2:bypassAdversaryInTheMiddleDefense',
        'Network:2:successfulAdversaryInTheMiddle',
        'ConnectionRule:3:successfulAccessNetworksInspected',
        'ConnectionRule:1:deny',
        'ConnectionRule:3:deny',
        'Program 2:networkConnectInspected',
        'Network:2:eavesdrop',
        'Network:2:adversaryInTheMiddle',
        'ConnectionRule:3:accessNetworksInspected',
        'Program 1:denyFromNetworkingAsset',
        'Program 2:denyFromNetworkingAsset',
        'Program 2:specificAccessNetworkConnect',
        'Program 2:networkConnect',
        'Program 2:attemptDeny'
    ]

    assert defender_actions == [
        'Network:2:adversaryInTheMiddleDefense',
        'ConnectionRule:3:payloadInspection',
        'Program 2:supplyChainAuditing',
        'ConnectionRule:3:restricted',
        'Network:2:eavesdropDefense',
        'ConnectionRule:1:payloadInspection',
        'Program 1:notPresent',
        'Network:2:networkAccessControl',
        'Program 1:supplyChainAuditing',
        'Program 2:notPresent',
        'ConnectionRule:1:restricted',
    ]
    for step_id in attacker_actions:
        # Make sure that all attacker actions led to compromise
        node = sim.attack_graph.get_node_by_full_name(step_id)
        assert node in attacker_state.performed_nodes

    for step_id in defender_actions:
        # Make sure that all defender actions let to defense enabled
        node = sim.attack_graph.get_node_by_full_name(step_id)
        assert node in defender_state.performed_nodes

    # Verify rewards in latest run and total rewards
    assert sim.agent_reward(attacker_state.name) == 0
    assert sim.agent_reward(defender_state.name) == -19

    assert total_reward_attacker == 0
    assert total_reward_defender == -11994.0  # It ran for a while


def test_bfs_vs_bfs_state_and_reward_PER_STEP_TRIAL() -> None:

    sim, agents = create_simulator_from_scenario(
        "tests/testdata/scenarios/bfs_vs_bfs_scenario.yml",
        sim_settings = MalSimulatorSettings(
            seed=100,
            ttc_mode=TTCMode.EFFORT_BASED_PER_STEP_SAMPLE
        )
    )

    defender_agent_name = "defender1"
    attacker_agent_name = "attacker1"

    attacker_agent = next(
        agent_info['agent'] for agent_info in agents
        if agent_info["name"] == attacker_agent_name
    )
    defender_agent = next(
        agent_info['agent'] for agent_info in agents
        if agent_info["name"] == defender_agent_name
    )

    total_reward_defender = 0.0
    total_reward_attacker = 0.0

    attacker_actions = []
    defender_actions = []

    states = sim.reset()
    attacker_state = states[attacker_agent_name]
    defender_state = states[defender_agent_name]

    while not sim.done():
        # Run the simulation until agents are terminated/truncated
        attacker_node = attacker_agent.get_next_action(attacker_state)
        defender_node = defender_agent.get_next_action(defender_state)

        # Step
        actions = {
            defender_agent_name: [defender_node] if defender_node else [],
            attacker_agent_name: [attacker_node] if attacker_node else []
        }
        states = sim.step(actions)
        defender_state = states[defender_agent_name]
        attacker_state = states[attacker_agent_name]
        assert isinstance(defender_state, MalSimDefenderState)

        # If actions were performed, add them to respective list
        if attacker_node and attacker_node in attacker_state.step_performed_nodes:
            attacker_actions.append(attacker_node.full_name)
            assert attacker_node in defender_state.step_all_compromised_nodes

        if defender_node and defender_node in defender_state.step_performed_nodes:
            defender_actions.append(defender_node.full_name)

        total_reward_defender += sim.agent_reward(defender_state.name)
        total_reward_attacker += sim.agent_reward(attacker_state.name)

    assert sim.cur_iter == 118

    # Make sure the actions performed were as expected
    assert attacker_actions == [
        'Program 1:attemptApplicationRespondConnectThroughData',
        'Program 1:attemptRead',
        'Program 1:attemptDeny',
        'Program 1:accessNetworkAndConnections',
        'Program 1:attemptModify',
        'Program 1:specificAccess',
        'ConnectionRule:1:attemptAccessNetworksUninspected',
        'ConnectionRule:1:attemptConnectToApplicationsUninspected',
        'ConnectionRule:1:attemptAccessNetworksInspected',
        'ConnectionRule:1:attemptConnectToApplicationsInspected',
        'ConnectionRule:1:bypassRestricted',
        'ConnectionRule:1:successfulAccessNetworksInspected',
        'ConnectionRule:1:connectToApplicationsInspected',
        'ConnectionRule:1:accessNetworksInspected',
        'Program 1:networkConnectInspected',
        'Network:2:accessInspected',
        'Program 1:specificAccessNetworkConnect',
        'Program 1:networkConnect',
        'Network:2:accessNetworkData',
        'Network:2:networkForwardingInspected',
        'Network:2:deny',
        'ConnectionRule:3:attemptConnectToApplicationsInspected',
        'Network:2:attemptEavesdrop',
        'Network:2:attemptAdversaryInTheMiddle',
        'ConnectionRule:3:attemptAccessNetworksInspected',
        'ConnectionRule:1:attemptDeny',
        'ConnectionRule:3:attemptDeny',
        'ConnectionRule:3:bypassPayloadInspection',
        'ConnectionRule:3:bypassRestricted',
        'ConnectionRule:3:connectToApplicationsInspected',
        'Network:2:successfulEavesdrop',
        'Network:2:bypassEavesdropDefense',
        'Network:2:bypassAdversaryInTheMiddleDefense',
        'Network:2:successfulAdversaryInTheMiddle',
        'ConnectionRule:3:successfulAccessNetworksInspected',
        'ConnectionRule:1:deny',
        'ConnectionRule:3:deny',
        'Program 2:networkConnectInspected',
        'Network:2:eavesdrop',
        'Network:2:adversaryInTheMiddle',
        'ConnectionRule:3:accessNetworksInspected',
        'Program 1:denyFromNetworkingAsset',
        'Program 2:denyFromNetworkingAsset',
        'Program 2:specificAccessNetworkConnect',
        'Program 2:networkConnect',
        'Program 2:attemptDeny'
    ]

    assert defender_actions == [
        'Network:2:adversaryInTheMiddleDefense',
        'ConnectionRule:3:payloadInspection',
        'Program 2:supplyChainAuditing',
        'ConnectionRule:3:restricted',
        'Network:2:eavesdropDefense',
        'ConnectionRule:1:payloadInspection',
        'Program 1:notPresent',
        'Network:2:networkAccessControl',
        'Program 1:supplyChainAuditing',
        'Program 2:notPresent',
        'ConnectionRule:1:restricted',
    ]
    for step_id in attacker_actions:
        # Make sure that all attacker actions led to compromise
        node = sim.attack_graph.get_node_by_full_name(step_id)
        assert node in attacker_state.performed_nodes

    for step_id in defender_actions:
        # Make sure that all defender actions let to defense enabled
        node = sim.attack_graph.get_node_by_full_name(step_id)
        assert node in defender_state.performed_nodes

    # Verify rewards in latest run and total rewards
    assert sim.agent_reward(attacker_state.name) == 0
    assert sim.agent_reward(defender_state.name) == -19

    assert total_reward_attacker == 0
    assert total_reward_defender == -2133.0


def test_bfs_vs_bfs_state_and_reward_expected_value_ttc() -> None:

    sim, agents = create_simulator_from_scenario(
        "tests/testdata/scenarios/bfs_vs_bfs_scenario.yml",
        sim_settings = MalSimulatorSettings(
            seed=13,
            ttc_mode=TTCMode.EXPECTED_VALUE
        )
    )

    defender_agent_name = "defender1"
    attacker_agent_name = "attacker1"

    attacker_agent = next(
        agent_info['agent'] for agent_info in agents
        if agent_info["name"] == attacker_agent_name
    )
    defender_agent = next(
        agent_info['agent'] for agent_info in agents
        if agent_info["name"] == defender_agent_name
    )

    total_reward_defender = 0.0
    total_reward_attacker = 0.0

    attacker_actions = []
    defender_actions = []

    states = sim.reset()
    attacker_state = states[attacker_agent_name]
    defender_state = states[defender_agent_name]

    while not sim.done():
        # Run the simulation until agents are terminated/truncated
        attacker_node = attacker_agent.get_next_action(attacker_state)
        defender_node = defender_agent.get_next_action(defender_state)

        # Step
        actions = {
            defender_agent_name: [defender_node] if defender_node else [],
            attacker_agent_name: [attacker_node] if attacker_node else []
        }
        states = sim.step(actions)
        defender_state = states[defender_agent_name]
        attacker_state = states[attacker_agent_name]
        assert isinstance(defender_state, MalSimDefenderState)

        # If actions were performed, add them to respective list
        if attacker_node and attacker_node in attacker_state.step_performed_nodes:
            attacker_actions.append(attacker_node.full_name)
            assert attacker_node in defender_state.step_all_compromised_nodes

        if defender_node and defender_node in defender_state.step_performed_nodes:
            defender_actions.append(defender_node.full_name)

        total_reward_defender += sim.agent_reward(defender_state.name)
        total_reward_attacker += sim.agent_reward(attacker_state.name)

    assert sim.cur_iter == 11

    # Make sure the actions performed were as expected
    assert attacker_actions == [
        'Program 1:attemptApplicationRespondConnectThroughData',
        'Program 1:attemptRead',
        'Program 1:attemptDeny',
        'Program 1:accessNetworkAndConnections',
        'Program 1:attemptModify',
        'Program 1:specificAccess',
        'ConnectionRule:1:attemptAccessNetworksUninspected',
        'ConnectionRule:1:attemptConnectToApplicationsUninspected',
        'ConnectionRule:1:attemptAccessNetworksInspected',
        'ConnectionRule:1:attemptConnectToApplicationsInspected'
    ]

    assert defender_actions == [
        'Network:2:adversaryInTheMiddleDefense',
        'ConnectionRule:3:payloadInspection',
        'Program 2:supplyChainAuditing',
        'ConnectionRule:3:restricted',
        'Network:2:eavesdropDefense',
        'ConnectionRule:1:payloadInspection',
        'Program 1:notPresent',
        'Network:2:networkAccessControl',
        'Program 1:supplyChainAuditing',
        'Program 2:notPresent',
        'ConnectionRule:1:restricted'
    ]
    for step_id in attacker_actions:
        # Make sure that all attacker actions led to compromise
        node = sim.attack_graph.get_node_by_full_name(step_id)
        assert node in attacker_state.performed_nodes

    for step_id in defender_actions:
        # Make sure that all defender actions let to defense enabled
        node = sim.attack_graph.get_node_by_full_name(step_id)
        assert node in defender_state.performed_nodes

    # Verify rewards in latest run and total rewards
    assert sim.agent_reward(attacker_state.name) == 0
    assert sim.agent_reward(defender_state.name) == -19

    assert total_reward_attacker == 0
    assert total_reward_defender == -100.0
