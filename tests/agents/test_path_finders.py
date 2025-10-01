from malsim.mal_simulator import (
    MalSimulator, MalSimulatorSettings, TTCMode
)
from malsim.agents import get_shortest_path_to
from malsim.agents.utils.greedy_a_star.algo import greedy_a_star_attack
from malsim.scenario import load_scenario
from malsim.graph_processing import prune_unviable_and_unnecessary_nodes

def test_path_finding() -> None:
    r"""

    """
    scenario_file = (
        "tests/testdata/scenarios/traininglang_scenario.yml"
    )
    scenario = load_scenario(scenario_file)
    sim = MalSimulator.from_scenario(scenario, register_agents=False)
    user_3_phish = sim.get_node('User:3:phishing')
    host_0_connect = sim.get_node('Host:0:connect')
    data_2_read = sim.get_node('Data:2:read')

    sim.register_attacker('path_finder', {host_0_connect, user_3_phish})
    agent_state = sim.agent_states['path_finder']

    path = get_shortest_path_to(
        sim.attack_graph,
        list(agent_state.performed_nodes),
        data_2_read,
        ttc_values={
            n: 1.0 for n in sim.attack_graph.nodes.values()
        }
    )
    assert [n.full_name for n in path] == ["Host:0:access", "Data:2:read"]


def test_path_finding_ttc_lang() -> None:
    scenario_file = (
        "tests/testdata/scenarios/ttc_lang_scenario.yml"
    )
    scenario = load_scenario(scenario_file)
    sim = MalSimulator.from_scenario(
        scenario,
        sim_settings=MalSimulatorSettings(
            ttc_mode=TTCMode.EXPECTED_VALUE,
            seed=100
        ),
        register_agents=False
    )

    entry_point = sim.get_node('Net1:easyAccess')
    goal = sim.get_node('DataD:read')
    ttc_values = {
        n: sim.node_ttc_value(n)
        for n in sim.attack_graph.nodes.values()
        if n.type in ('or', 'and')
    }

    sim.register_attacker('path_finder', {entry_point}, {goal})

    path = get_shortest_path_to(
        sim.attack_graph,
        [entry_point],
        goal,
        ttc_values=ttc_values
    )

    assert path
    for node in path:
        # Should only have picked the low ttc steps
        assert ('hard' not in node.full_name or node == goal)


def test_sandor_path_finding_ttc_lang() -> None:
    scenario_file = (
        "tests/testdata/scenarios/ttc_lang_scenario.yml"
    )
    scenario = load_scenario(scenario_file)
    sim = MalSimulator.from_scenario(scenario)

    prune_unviable_and_unnecessary_nodes(
        scenario.attack_graph,
        sim._viability_per_node,
        sim._necessity_per_node
    )
    entry_point = sim.get_node('Net1:easyAccess')
    goal = sim.get_node('UserA:easyAssume')

    # Run the greedy a star
    path = greedy_a_star_attack(scenario.attack_graph, entry_point, goal)
    assert path

    # Validate path - TODO: this fails!
    visited = {entry_point}
    curr_node = None
    for curr_node in path:
        if curr_node.type == 'or' and curr_node not in visited:
            assert any(p in visited for p in curr_node.parents), (
                f"Node {curr_node} was reached before any of its parents were"
            )
        elif curr_node.type == 'and' and curr_node not in visited:
            assert all(p in visited for p in curr_node.parents), (
                f"Node {curr_node} was reached before all of its parents"
            )
        visited.add(curr_node)
    assert curr_node == goal
