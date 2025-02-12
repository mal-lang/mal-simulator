from malsim.scenario import create_simulator_from_scenario


def test_bfs_vs_bfs_state_and_reward():
    sim, sim_config = create_simulator_from_scenario(
        "tests/testdata/scenarios/bfs_vs_bfs_network_app_data_scenario.yml"
    )
    obs, infos = sim.reset()

    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    defender_agent_id = next(iter(sim.get_defender_agents()), None)

    # Initialize defender and attacker according to classes
    defender_class = sim_config["agents"][defender_agent_id]["agent_class"]
    defender_agent = defender_class({})

    attacker_class = sim_config["agents"][attacker_agent_id]["agent_class"]
    attacker_agent = attacker_class({})

    total_reward_defender = 0
    total_reward_attacker = 0

    attacker_actions = []
    defender_actions = []

    while True:
        defender_action = defender_agent.compute_action_from_dict(
            obs[defender_agent_id], infos[defender_agent_id]["action_mask"]
        )

        attacker_action = attacker_agent.compute_action_from_dict(
            obs[attacker_agent_id], infos[attacker_agent_id]["action_mask"]
        )

        if attacker_action[0]:
            attacker_node = sim.action_to_node(attacker_action)
            attacker_actions.append(attacker_node.full_name)
        if defender_action[0]:
            defender_node = sim.action_to_node(defender_action)
            defender_actions.append(defender_node.full_name)

        actions = {"defender": defender_action, "attacker": attacker_action}
        obs, rewards, terminated, truncated, infos = sim.step(actions)

        total_reward_defender += rewards.get(defender_agent_id, 0)
        total_reward_attacker += rewards.get(attacker_agent_id, 0)

        if terminated[defender_agent_id] or terminated[attacker_agent_id]:
            break

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

    # Verify observations
    for step_fullname in attacker_actions:
        node = sim.attack_graph.get_node_by_full_name(step_fullname)
        if node.is_compromised():
            node_index = sim.node_to_index(node)
            assert obs[defender_agent_id]["observed_state"][node_index]

    for step_fullname in defender_actions:
        node = sim.attack_graph.get_node_by_full_name(step_fullname)
        node_index = sim.node_to_index(node)
        assert obs[defender_agent_id]["observed_state"][node_index]

    assert rewards[attacker_agent_id] == 0
    assert rewards[defender_agent_id] == -50

    assert total_reward_attacker == 0
    assert total_reward_defender == -2000


def test_scenario_step_by_step():
    sim, sim_config = create_simulator_from_scenario(
        "tests/testdata/scenarios/bfs_vs_bfs_network_app_data_scenario.yml"
    )
    sim.reset()

    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    defender_agent_id = next(iter(sim.get_defender_agents()), None)

    attacker_actions = [
        "Internet:attemptReverseReach",
        "Internet:reverseReach",
        "ConnectionRule Internet->Linux System:attemptReverseReach",
        "ConnectionRule Internet->Linux System:reverseReach",
        "Linux system:attemptReverseReach",
        "Linux system:successfulReverseReach",
        "Linux system:reverseReach",
        "Secret data:attemptReverseReach",
        "Secret data:reverseReach",
    ]

    # Make sure attacker can take these steps
    for attacker_action_fn in attacker_actions:
        attacker_node = sim.attack_graph.get_node_by_full_name(attacker_action_fn)
        attacker_action_index = sim.node_to_index(attacker_node)
        actions = {
            defender_agent_id: (0, None),
            attacker_agent_id: (1, attacker_action_index)
        }
        sim.step(actions)
        assert attacker_node.is_compromised()


    # TODO: find out if this fails because is_traversable is not correct
    # But not this one
    # attacker_node = sim.attack_graph.get_node_by_full_name(
    #     "Secret data:read"
    # )
    # attacker_action_index = sim.node_to_index(attacker_node)

    # actions = {
    #     defender_agent_id: (0, None),
    #     attacker_agent_id: (1, attacker_action_index)
    # }

    # sim.step(actions)
    # assert not attacker_node.is_compromised()