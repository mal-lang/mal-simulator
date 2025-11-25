"""
Scenario can be created as an object directly instead of as a separate file.
"""

from maltoolbox.model import Model
from maltoolbox.language import LanguageGraph

from malsim import MalSimulator, run_simulation
from malsim.scenario import Scenario, AttackerSettings, DefenderSettings
from malsim.agents import BreadthFirstAttacker, PassiveAgent


def test_scenario_obj_files() -> None:
    scenario = Scenario(
        lang_file='tests/testdata/langs/org.mal-lang.trainingLang-1.0.0.mar',
        model='tests/testdata/models/traininglang_model.yml',
        agent_settings={
            'Attacker1': AttackerSettings(
                name='Attacker1',
                entry_points={'User:3:phishing', 'Host:0:connect'},
                policy=BreadthFirstAttacker,
            ),
            'Defender1': DefenderSettings(name='Defender1', policy=PassiveAgent),
        },
    )

    scenario.save_to_file('scenario1.yml')

    mal_simulator = MalSimulator.from_scenario(scenario)
    _ = run_simulation(mal_simulator, scenario.agent_settings)


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
        model=model.to_dict(),
        agent_settings={
            'Attacker1': AttackerSettings(
                name='Attacker1',
                entry_points={'User:3:phishing', 'Host:0:connect'},
                policy=BreadthFirstAttacker,
            ),
            'Defender1': DefenderSettings(name='Defender1', policy=PassiveAgent),
        },
    )

    scenario.save_to_file('scenario2.yml')

    mal_simulator = MalSimulator.from_scenario(scenario)
    _ = run_simulation(mal_simulator, scenario.agent_settings)


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
        agent_settings={
            'Attacker1': AttackerSettings(
                name='Attacker1',
                entry_points={'User:3:phishing', 'Host:0:connect'},
                policy=BreadthFirstAttacker,
            ),
            'Defender1': DefenderSettings(name='Defender1', policy=PassiveAgent),
        },
    )

    scenario.save_to_file('scenario3.yml')

    mal_simulator = MalSimulator.from_scenario(scenario)
    _ = run_simulation(mal_simulator, scenario.agent_settings)


if __name__ == '__main__':
    test_scenario_obj_files()
    test_scenario_obj_file_and_dict()
    test_scenario_obj_file_and_model()
