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
        "tests/testdata/scenarios/bfs_vs_bfs_scenario.yml",
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

    attacker = sim.attack_graph.get_attacker_by_id(attacker_agent_info["attacker_id"])
    attacker_actions = [n.full_name for n in attacker.entry_points]
    defender_actions = [
        n.full_name for n in sim.attack_graph.nodes if n.is_enabled_defense()
    ]

    while True:
        # Run the simulation until agents are terminated/truncated

        # Select attacker node
        attacker_agent_state = sim.get_agent_state(attacker_agent_info["name"])
        attacker_node = attacker_agent.get_next_action(attacker_agent_state)

        # Select defender node
        defender_agent_state = sim.get_agent_state(defender_agent_info["name"])
        defender_node = defender_agent.get_next_action(defender_agent_state)

        # Step
        actions = {
            defender_agent_name: [defender_node] or [],
            attacker_agent_name: [attacker_node] or [],
        }
        performed_actions, _ = sim.step(actions)

        # If actions were performed, add them to respective list
        if attacker_node and attacker_node in performed_actions:
            attacker_actions.append(attacker_node.full_name)

        if defender_node and defender_node in performed_actions:
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
    assert defender_agent_state.reward == -31

    assert total_reward_attacker == 0
    assert total_reward_defender == -307
