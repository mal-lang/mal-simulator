from malsim.scenario import load_scenario
from malsim.sims.mal_simulator import MalSimulator

def test():
    attack_graph, conf = load_scenario(
        'tests/testdata/scenarios/simple_scenario.yml'
    )

    # Make alteration to the attack graph attacker
    assert len(attack_graph.attackers) == 1
    attacker = attack_graph.attackers[0]
    assert len(attacker.reached_attack_steps) == 1
    reached_step = attacker.reached_attack_steps[0]
    for child_node in reached_step.children:
        if child_node.type in ('and', 'or'):
            # compromise children of reached step so in the end the
            # attacker will have three reached attack steps where
            # two are children of the first one
            attacker.compromise(child_node)
    assert len(attacker.reached_attack_steps) == 3

    #Create the simulator
    sim = MalSimulator(
        attack_graph.lang_graph, attack_graph.model, attack_graph)

    # Register the agents
    attacker_agent_id = "attacker"
    defender_agent_if = "defender"
    sim.register_attacker(attacker_agent_id, 0)
    sim.register_defender(defender_agent_if)

    obs, _ = sim.reset()

    attacker_agent_id = next(iter(sim.get_attacker_agents()))
    attacker_observation = obs[attacker_agent_id]["observed_state"]
    attacker = sim.attack_graph.attackers[0]

    for node in attacker.reached_attack_steps:
        node_index = sim._id_to_index[node.id]
        node_obs_state = attacker_observation[node_index]
        assert node_obs_state == 1
