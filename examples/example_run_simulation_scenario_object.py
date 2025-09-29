"""
Scenario can be created as an object directly instead of as a separate file.
"""
from malsim import MalSimulator, run_simulation
from malsim.scenario import Scenario

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

mal_simulator = MalSimulator.from_scenario(scenario)
paths = run_simulation(mal_simulator, scenario.agents)
