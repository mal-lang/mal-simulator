from malsim import MalSimulator, load_scenario, MalSimulatorSettings
from malsim.mal_simulator import TTCMode, MalSimAttackerState
from malsim.agents import TTCSoftMinAttacker

def test_ttc_avoider() -> None:
    """TTC Avoider"""

    scenario_file = (
        "tests/testdata/scenarios/bfs_vs_bfs_scenario.yml"
    )
    scenario = load_scenario(scenario_file)
    sim = MalSimulator(
        scenario.attack_graph,
        sim_settings = MalSimulatorSettings(
            seed=48,
            ttc_mode=TTCMode.PRE_SAMPLE
        ),
    )

    attacker_agent_name = "TTCAvoidingAttacker"
    attacker_agent = TTCSoftMinAttacker({})
    fa = sim.attack_graph.get_node_by_full_name('Program 1:fullAccess')
    assert fa
    sim.register_attacker(attacker_agent_name, {fa})

    states = sim.reset()
    attacker_state = states[attacker_agent_name]

    chosen_nodes = []
    while not sim.done():
        # Run the simulation until agents are terminated/truncated
        assert isinstance(attacker_state, MalSimAttackerState)
        attacker_node = attacker_agent.get_next_action(attacker_state)
        if attacker_node:
            chosen_nodes.append(
                (
                    attacker_node.full_name,
                    sim.node_ttc_value(attacker_node) - attacker_state.num_attempts[attacker_node]
                )
            )

        # Step
        actions = {
            attacker_agent_name: [attacker_node] if attacker_node else []
        }
        states = sim.step(actions)
        attacker_state = states[attacker_agent_name]
