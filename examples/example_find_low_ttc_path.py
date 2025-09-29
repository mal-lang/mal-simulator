"""
Attacker agents can have goals, and we can use the TTCSoftMinAttacker to find a 'cheap' path to the goal.
"""
from malsim.mal_simulator import (
    MalSimulator,
    MalSimulatorSettings,
    run_simulation,
    TTCMode
)
from malsim.scenario import Scenario

def test_run_scenario_ttc_soft_min_attacker() -> None:
    scenario = Scenario(
        lang_file='tests/testdata/langs/org.mal-lang.trainingLang-1.0.0.mar',
        model_file='tests/testdata/models/traininglang_model.yml',
        agents={
            'Attacker1': {
                'type': 'attacker',
                'agent_class': 'TTCSoftMinAttacker',
                'entry_points': ['User:3:phishing', 'Host:0:connect'],
                'goals': ['Data:2:read']
            },
            'Defender1': {
                'type': 'defender',
                'agent_class': 'PassiveAgent'
            }
        }
    )

    mal_simulator = MalSimulator.from_scenario(
        scenario, sim_settings=MalSimulatorSettings(ttc_mode=TTCMode.EXPECTED_VALUE)
    )
    paths = run_simulation(mal_simulator, scenario.agents)
    print(paths)


if __name__ == '__main__':
    test_run_scenario_ttc_soft_min_attacker()