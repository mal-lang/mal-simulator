"""
Attacker agents can have goals, and we can use the TTCSoftMinAttacker
to find a 'cheap' path to the goal.
"""

from malsim.mal_simulator import (
    MalSimulator,
    MalSimulatorSettings,
    run_simulation,
    TTCMode,
)
from malsim.scenario.scenario import (
    Scenario,
    AttackerSettings,
    DefenderSettings,
    NodePropertyRule,
)

from malsim.policies import TTCSoftMinAttacker, PassiveAgent


def test_run_scenario_ttc_soft_min_attacker() -> None:
    scenario = Scenario(
        lang_file='tests/testdata/langs/org.mal-lang.trainingLang-1.0.0.mar',
        model='tests/testdata/models/traininglang_model.yml',
        agent_settings={
            'Attacker1': AttackerSettings(
                name='Attacker1',
                entry_points={'User:3:phishing', 'Host:0:connect'},
                goals={'Data:2:read'},
                policy=TTCSoftMinAttacker,
                rewards=NodePropertyRule(by_asset_name={'Host:0': {'access': 10}}),
            ),
            'Defender1': DefenderSettings(name='Defender1', policy=PassiveAgent),
        },
    )

    mal_simulator = MalSimulator.from_scenario(
        scenario, sim_settings=MalSimulatorSettings(ttc_mode=TTCMode.EXPECTED_VALUE)
    )
    paths = run_simulation(mal_simulator, scenario.agent_settings)
    print(paths)


if __name__ == '__main__':
    test_run_scenario_ttc_soft_min_attacker()
