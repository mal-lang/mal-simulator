"""
Scenario file / dict is not needed to run a simulation.
All that is needed is an attack graph, but we need to write our own simulation loop.
Not recommended - see other examples.
"""

import pprint

from maltoolbox.language import LanguageGraph
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.model import Model

from malsim.config.agent_settings import AttackerSettings, DefenderSettings
from malsim.mal_simulator import MalSimulator, DefenderState, AttackerState
from malsim.policies import BreadthFirstAttacker, DefendFutureCompromisedDefender


def test_example_no_scenario() -> None:
    file_lang = 'tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar'
    path_to_model = 'tests/testdata/models/basic_model.yml'

    # Everything below this line is done internally
    # when loading a scenario with Scenario.load_from_file
    corelang_graph = LanguageGraph.from_mar_archive(file_lang)
    model = Model.load_from_file(path_to_model, corelang_graph)
    attack_graph = AttackGraph(corelang_graph, model)
    attacker_name = 'MyAttacker'
    defender_name = 'MyDefender'
    sim = MalSimulator(
        attack_graph,
        agents=(
            AttackerSettings(
                name=attacker_name,
                entry_points=frozenset(
                    {attack_graph.get_node_by_full_name('Program 1:fullAccess')}
                ),
            ),
            DefenderSettings(name=defender_name),
        ),
    )

    attacker_policy = BreadthFirstAttacker({})
    defender_policy = DefendFutureCompromisedDefender({})

    states = sim.reset()
    while not sim.done():
        # Step with attacker and defender
        attacker_state: AttackerState = states[attacker_name]  # type: ignore
        attacker_action = attacker_policy.get_next_action(attacker_state)
        defender_state: DefenderState = states[defender_name]  # type: ignore
        defender_action = defender_policy.get_next_action(defender_state)
        states = sim.step(
            {
                attacker_name: [attacker_action] if attacker_action else [],
                defender_name: [defender_action] if defender_action else [],
            }
        )

    pprint.pprint(sim.recording)


if __name__ == '__main__':
    test_example_no_scenario()
