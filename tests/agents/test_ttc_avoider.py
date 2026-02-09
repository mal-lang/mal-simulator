from malsim import MalSimulator, Scenario, MalSimulatorSettings
from malsim.mal_simulator import TTCMode, MalSimAttackerState
from malsim.policies import TTCSoftMinAttacker


def test_ttc_avoider() -> None:
    """TTC Avoider"""

    scenario_file = 'tests/testdata/scenarios/ttc_lang_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim = MalSimulator(
        scenario.attack_graph,
        sim_settings=MalSimulatorSettings(
            seed=48,
            ttc_mode=TTCMode.PRE_SAMPLE,
            attack_surface_skip_unnecessary=False,
        ),
    )

    attacker_agent_name = 'TTCAvoidingAttacker'
    attacker_agent = TTCSoftMinAttacker({'seed': 1})
    entry_point = sim.get_node('Net1:easyAccess')
    goal = sim.get_node('DataD:read')
    sim.register_attacker(attacker_agent_name, {entry_point}, {goal})

    states = sim.agent_states
    attacker_state = states[attacker_agent_name]

    path = list()

    while not sim.done():
        # Run the simulation until agents are terminated/truncated
        assert isinstance(attacker_state, MalSimAttackerState)
        attacker_node = attacker_agent.get_next_action(attacker_state)
        path.append(attacker_node.full_name)
        assert attacker_node
        # Should always pick the easy path or the goal
        assert 'easy' in attacker_node.name or attacker_node == goal

        # Step
        actions = {attacker_agent_name: [attacker_node] if attacker_node else []}
        states = sim.step(actions)
        attacker_state = states[attacker_agent_name]

    assert path == [
        'ComputerA:easyConnect',
        'UserA:easyAssume',
        'Net2:easyAccess',
        'SoftwareA:easyScan',
        'ComputerA:easyAccess',
        'ComputerB:easyConnect',
        'SWVulnA:easyExploit',
        'ComputerC:easyConnect',
        'SoftwareA:easyAccess',
        'Net3:easyAccess',
        'ComputerD:easyConnect',
        'UserD:easyAssume',
        'ComputerD:easyAccess',
        'DataD:easyRead',
        'SoftwareD:easyScan',
        'SWVulnD:easyExploit',
        'SoftwareD:easyAccess',
        'DataD:read'
    ]

def test_ttc_avoider_low_sharpness() -> None:
    """TTC Avoider with low beta/sharpness"""

    scenario_file = 'tests/testdata/scenarios/ttc_lang_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim = MalSimulator(
        scenario.attack_graph,
        sim_settings=MalSimulatorSettings(
            seed=48, ttc_mode=TTCMode.PRE_SAMPLE, attack_surface_skip_unnecessary=False
        ),
    )

    attacker_agent_name = 'TTCAvoidingAttacker'
    attacker_agent = TTCSoftMinAttacker({'beta': 0.1, 'seed': 1})
    entry_point = sim.get_node('Net1:easyAccess')
    goal = sim.get_node('DataD:read')
    sim.register_attacker(attacker_agent_name, {entry_point}, {goal})

    states = sim.agent_states
    attacker_state = states[attacker_agent_name]

    recording = []
    while not sim.done():
        # Run the simulation until agents are terminated/truncated
        assert isinstance(attacker_state, MalSimAttackerState)
        attacker_node = attacker_agent.get_next_action(attacker_state)
        assert attacker_node

        # Step
        actions = {attacker_agent_name: [attacker_node] if attacker_node else []}
        states = sim.step(actions)
        recording.append(attacker_node)
        attacker_state = states[attacker_agent_name]

    # low sharpness will pick both easy and hard
    assert any('easy' in n.name for n in recording)
    assert any('hard' in n.name for n in recording)
