from maltoolbox.language import LanguageGraph
from malsim.mal_simulator import MalSimulator
from malsim.agents.attackers.path_finding import get_shortest_path_to
from malsim.scenario import load_scenario

def test_path_finder_agent(
        dummy_lang_graph: LanguageGraph
    ) -> None:
    r"""

    """
    scenario_file = (
        "tests/testdata/scenarios/traininglang_scenario.yml"
    )
    scenario = load_scenario(scenario_file)
    sim = MalSimulator.from_scenario(scenario, register_agents=False)
    user_3_phish = sim.attack_graph.get_node_by_full_name('User:3:phishing')
    host_0_connect = sim.attack_graph.get_node_by_full_name('Host:0:connect')
    host_0_access = sim.attack_graph.get_node_by_full_name('Host:0:access')
    net_3_access = sim.attack_graph.get_node_by_full_name('Network:3:access')
    data_2_read = sim.attack_graph.get_node_by_full_name('Data:2:read')

    sim.register_attacker('path_finder', {host_0_access, net_3_access})
    agent_state = sim.agent_states['path_finder']

    found_path, path, ttc_cost = get_shortest_path_to(
        sim.attack_graph,
        list(agent_state.performed_nodes),
        data_2_read,
        ttc_values={
            n:1.0 for n in sim.attack_graph.nodes.values()
        }
    )
    breakpoint()
