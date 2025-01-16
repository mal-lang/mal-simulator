from malsim.scenario import create_simulator_from_scenario


def test_bfs_vs_bfs_state_and_reward():
    sim, sim_config = create_simulator_from_scenario(
        "tests/testdata/scenarios/bfs_vs_bfs_scenario.yml"
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

    attacker = sim.attack_graph.attackers[0]
    attacker_actions = [n.full_name for n in attacker.entry_points]
    defender_actions = [
        n.full_name for n in sim.attack_graph.nodes if n.is_enabled_defense()
    ]

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
        "Credentials:6:attemptCredentialsReuse",
        "Credentials:6:credentialsReuse",
        "Credentials:7:attemptCredentialsReuse",
        "Credentials:6:attemptUse",
        "Credentials:7:credentialsReuse",
        "Credentials:7:attemptUse",
        "Credentials:7:use",
        "Credentials:7:attemptPropagateOneCredentialCompromised",
        "Credentials:7:propagateOneCredentialCompromised",
        "User:12:oneCredentialCompromised",
        "User:12:passwordReuseCompromise",
        "Credentials:9:attemptCredentialsReuse",
        "Credentials:10:attemptCredentialsReuse",
        "Credentials:9:credentialsReuse",
        "Credentials:9:attemptUse",
    ]
    assert defender_actions == [
        "Program 1:notPresent",
        "IDPS 1:effectiveness",
        "Credentials:6:notDisclosed",
        "Credentials:6:notGuessable",
        "Credentials:7:notDisclosed",
        "Credentials:7:notGuessable",
        "Credentials:9:notDisclosed",
        "Credentials:9:notGuessable",
        "Credentials:10:notDisclosed",
        "Credentials:10:notGuessable",
        "Credentials:10:unique",
        "User:12:noRemovableMediaUsage",
        "OS App:notPresent",
        "OS App:supplyChainAuditing",
        "Program 1:supplyChainAuditing",
        "Program 2:notPresent",
        "Program 2:supplyChainAuditing",
        "IDPS 1:notPresent",
        "IDPS 1:supplyChainAuditing",
        "SoftwareVulnerability:4:notPresent",
        "Data:5:notPresent",
        "Credentials:6:unique",
        "Credentials:6:notPhishable",
        "Credentials:7:unique",
        "Credentials:7:notPhishable",
        "Identity:8:notPresent",
    ]

    for step_index in attacker_actions:
        node = sim.attack_graph.nodes[sim._index_to_id[step_index]]
        if node.is_compromised():
            node_index = sim.node_to_index(node)
            assert obs[defender_agent_id]["observed_state"][node_index]

    for step_fullname in defender_actions:
        node = sim.attack_graph.get_node_by_full_name(step_fullname)
        node_index = sim.node_to_index(node)
        assert obs[defender_agent_id]["observed_state"][node_index]

    assert rewards[attacker_agent_id] == 0
    assert rewards[defender_agent_id] == -31

    assert total_reward_attacker == 0
    assert total_reward_defender == -307
