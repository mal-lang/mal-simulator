"""
Scenario can be created as an object directly instead of as a separate file.
"""
from maltoolbox.model import Model
from maltoolbox.language import LanguageGraph

from malsim import MalSimulator, run_simulation
from malsim.scenario import Scenario


def test_scenario_obj_files() -> None:
    scenario = Scenario(
        lang_file='tests/testdata/langs/org.mal-lang.trainingLang-1.0.0.mar',
        model_file='tests/testdata/models/traininglang_model.yml',
        agents={
            'Attacker1': {
                'type': 'attacker',
                'agent_class': 'BreadthFirstAttacker',
                'entry_points': ['User:3:phishing', 'Host:0:connect']
            },
            'Defender1': {
                'type': 'defender',
                'agent_class': 'PassiveAgent'
            }
        }
    )

    scenario.save_to_file('scenario1.yml')

    mal_simulator = MalSimulator.from_scenario(scenario)
    _ = run_simulation(mal_simulator, scenario.agents)


def test_scenario_obj_file_and_dict() -> None:
    """Create a scenario object from a lang graph file and model dictionary"""
    lang_graph = LanguageGraph.load_from_file(
        'tests/testdata/langs/org.mal-lang.trainingLang-1.0.0.mar'
    )
    model = Model.load_from_file(
        'tests/testdata/models/traininglang_model.yml', lang_graph
    )
    scenario = Scenario(
        lang_file='tests/testdata/langs/org.mal-lang.trainingLang-1.0.0.mar',
        model_dict=model.to_dict(),
        agents={
            'Attacker1': {
                'type': 'attacker',
                'agent_class': 'BreadthFirstAttacker',
                'entry_points': ['User:3:phishing', 'Host:0:connect']
            },
            'Defender1': {
                'type': 'defender',
                'agent_class': 'PassiveAgent'
            }
        }
    )

    scenario.save_to_file('scenario2.yml')

    mal_simulator = MalSimulator.from_scenario(scenario)
    _ = run_simulation(mal_simulator, scenario.agents)


def test_scenario_obj_file_and_model() -> None:
    """Create scenario object from a lang file and Model object"""
    lang_graph = LanguageGraph.load_from_file(
        'tests/testdata/langs/org.mal-lang.trainingLang-1.0.0.mar'
    )
    model = Model.load_from_file(
        'tests/testdata/models/traininglang_model.yml', lang_graph
    )
    scenario = Scenario(
        lang_file='tests/testdata/langs/org.mal-lang.trainingLang-1.0.0.mar',
        model=model,
        agents={
            'Attacker1': {
                'type': 'attacker',
                'agent_class': 'BreadthFirstAttacker',
                'entry_points': ['User:3:phishing', 'Host:0:connect']
            },
            'Defender1': {
                'type': 'defender',
                'agent_class': 'PassiveAgent'
            }
        }
    )

    scenario.save_to_file('scenario3.yml')

    mal_simulator = MalSimulator.from_scenario(scenario)
    _ = run_simulation(mal_simulator, scenario.agents)


if __name__ == '__main__':
    test_scenario_obj_files()
    test_scenario_obj_file_and_dict()
    test_scenario_obj_file_and_model()