from malsim.mal_simulator import MalSimulator, MalSimulatorSettings, TTCMode
from malsim.agents import get_shortest_path_to
from malsim.scenario import Scenario

import numpy as np


def test_path_finding() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim = MalSimulator.from_scenario(scenario, register_agents=False)
    user_3_phish = sim.get_node('User:3:phishing')
    host_0_connect = sim.get_node('Host:0:connect')
    data_2_read = sim.get_node('Data:2:read')

    sim.register_attacker('path_finder', {host_0_connect, user_3_phish})
    agent_state = sim.agent_states['path_finder']

    path = get_shortest_path_to(
        sim.sim_state.attack_graph,
        list(agent_state.performed_nodes),
        data_2_read,
        ttc_values={n: 1.0 for n in sim.sim_state.attack_graph.nodes.values()},
    )
    assert [n.full_name for n in path] == ['Host:0:access', 'Data:2:read']


def test_path_finding_ttc_lang() -> None:
    scenario_file = 'tests/testdata/scenarios/ttc_lang_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    sim = MalSimulator.from_scenario(
        scenario,
        sim_settings=MalSimulatorSettings(ttc_mode=TTCMode.EXPECTED_VALUE, seed=100),
        register_agents=False,
    )

    entry_point = sim.get_node('Net1:easyAccess')
    goal = sim.get_node('DataD:read')
    ttc_values = {
        n: sim.node_ttc_value(n)
        for n in sim.sim_state.attack_graph.nodes.values()
        if n.type in ('or', 'and')
    }

    assert np.isclose(sum(ttc_values.values()), 2021)
    sim.register_attacker('path_finder', {entry_point}, {goal})

    path = get_shortest_path_to(
        sim.sim_state.attack_graph, [entry_point], goal, ttc_values=ttc_values
    )

    assert path
    for node in path:
        # Should only have picked the low ttc steps
        assert 'easy' in node.full_name or node == goal
