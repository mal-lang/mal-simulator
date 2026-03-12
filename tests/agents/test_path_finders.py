from malsim.mal_simulator import MalSimulator, TTCMode
from malsim.policies import get_shortest_path_to
from malsim.scenario.scenario import Scenario

import numpy as np


def test_path_finding() -> None:
    scenario_file = 'tests/testdata/scenarios/traininglang_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attack_graph = scenario.attack_graph
    data_2_read = attack_graph.get_node_by_full_name('Data:2:read')
    sim = MalSimulator.from_scenario(scenario)

    agent_state = sim.agent_states['path_finder']

    path = get_shortest_path_to(
        sim.sim_state.attack_graph,
        list(agent_state.performed_nodes),
        data_2_read,
        ttc_values=dict.fromkeys(sim.sim_state.attack_graph.nodes.values(), 1.0),
    )
    assert [n.full_name for n in path] == ['Host:0:access', 'Data:2:read']


def test_path_finding_ttc_lang() -> None:
    scenario_file = 'tests/testdata/scenarios/ttc_lang_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    scenario.sim_settings.ttc_mode = TTCMode.EXPECTED_VALUE
    scenario.sim_settings.seed = 100
    attack_graph = scenario.attack_graph

    entry_point = attack_graph.get_node_by_full_name('Net1:easyAccess')
    goal = attack_graph.get_node_by_full_name('DataD:read')
    sim = MalSimulator.from_scenario(scenario)
    ttc_values = {
        n: sim.node_ttc_value(n)
        for n in sim.sim_state.attack_graph.nodes.values()
        if n.type in ('or', 'and')
    }

    assert np.isclose(sum(ttc_values.values()), 2021)

    path = get_shortest_path_to(
        sim.sim_state.attack_graph, [entry_point], goal, ttc_values=ttc_values
    )

    assert path
    for node in path:
        # Should only have picked the low ttc steps
        assert 'easy' in node.full_name or node == goal
