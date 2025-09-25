from malsim import MalSimulator, load_scenario, MalSimulatorSettings
from malsim.mal_simulator import TTCMode, MalSimAttackerState
from malsim.agents import TTCSoftMinAttacker

def test_ttc_avoider() -> None:
    """TTC Avoider"""

    scenario_file = (
        "tests/testdata/scenarios/ttc_lang_scenario.yml"
    )
    scenario = load_scenario(scenario_file)
    sim = MalSimulator(
        scenario.attack_graph,
        sim_settings = MalSimulatorSettings(
            seed=48,
            ttc_mode=TTCMode.PRE_SAMPLE,
            attack_surface_skip_unnecessary=False
        ),
    )

    attacker_agent_name = "TTCAvoidingAttacker"
    attacker_agent = TTCSoftMinAttacker({})
    entry_point = sim.get_node('Net1:easyAccess')
    goal = sim.get_node('DataD:read')
    sim.register_attacker(attacker_agent_name, {entry_point}, {goal})

    states = sim.reset()
    attacker_state = states[attacker_agent_name]

    while not sim.done():
        # Run the simulation until agents are terminated/truncated
        assert isinstance(attacker_state, MalSimAttackerState)
        attacker_node = attacker_agent.get_next_action(attacker_state)

        assert attacker_node
        # Should always pick the easy path or the goal
        assert 'easy' in attacker_node.name or attacker_node == goal

        # Step
        actions = {
            attacker_agent_name: [attacker_node] if attacker_node else []
        }
        states = sim.step(actions)
        attacker_state = states[attacker_agent_name]
