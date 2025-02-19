"""
Run a scenario and make sure expected actions are chosen and
expected reward is given to agents
"""

from malsim.scenario import create_simulator_from_scenario

def test_bfs_vs_bfs_state_and_reward():
    """
    The point of this test is to see that a specific
    scenario runs deterministically.

    The test creates a simulator, two agents and runs them both with
    BFS Agents against each other.

    It then verifies that rewards and actions performed are what we expected.
    """

    sim, agents = create_simulator_from_scenario(
        "tests/testdata/scenarios/bfs_vs_bfs_network_app_data_scenario.yml"
    )
    sim.reset()

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

    total_reward_defender = 0
    total_reward_attacker = 0

    attacker_actions = []
    defender_actions = []

    while True:
        # Run the simulation until agents are terminated/truncated

        # Select attacker node
        attacker_agent_state = sim.agent_states[attacker_agent_info["name"]]
        attacker_node = attacker_agent.get_next_action(attacker_agent_state)

        # Select defender node
        defender_agent_state = sim.agent_states[defender_agent_info["name"]]
        defender_node = defender_agent.get_next_action(defender_agent_state)

        # Step
        actions = {
            defender_agent_name: [defender_node] if defender_node else [],
            attacker_agent_name: [attacker_node] if attacker_node else []
        }
        states = sim.step(actions)

        # If actions were performed, add them to respective list
        if attacker_node and attacker_node in \
                states['attacker1'].step_compromised_nodes:
            attacker_actions.append(attacker_node.full_name)

        if defender_node and defender_node in \
                states['defender1'].step_enabled_defenses:
            defender_actions.append(defender_node.full_name)

        total_reward_defender += defender_agent_state.reward
        total_reward_attacker += attacker_agent_state.reward

        # Break simulation if trunc or term
        if defender_agent_state.terminated or attacker_agent_state.terminated:
            break
        if defender_agent_state.truncated or attacker_agent_state.truncated:
            break

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
        'ConnectionRule Internet->Linux System:attemptConnectToApplicationsInspected',
        'ConnectionRule Internet->Linux System:successfulAccessNetworksUninspected',
        'ConnectionRule Internet->Linux System:deny',
        'Internet:bypassEavesdropDefense',
        'Internet:successfulEavesdrop',
        'Internet:bypassAdversaryInTheMiddleDefense',
        'Internet:successfulAdversaryInTheMiddle',
        'Linux system:networkConnectUninspected',
        'Linux system:networkConnectInspected',
        'ConnectionRule Internet->Linux System:reverseReach',
        'ConnectionRule Internet->Linux System:successfulAccessNetworksInspected',
        'ConnectionRule Internet->Linux System:connectToApplicationsInspected',
        'ConnectionRule Internet->Linux System:accessNetworksUninspected',
        'Linux system:denyFromNetworkingAsset',
        'Internet:eavesdrop',
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
        assert node.is_compromised()

    for step_id in defender_actions:
        # Make sure that all defender actions let to defense enabled
        node = sim.attack_graph.get_node_by_full_name(step_id)
        assert node.is_enabled_defense()

    # Verify rewards in latest run and total rewards
    assert attacker_agent_state.reward == 0
    assert defender_agent_state.reward == -50

    assert total_reward_attacker == 0
    assert total_reward_defender == -2000
