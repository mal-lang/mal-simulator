from malsim import MalSimulator, Scenario, MalSimulatorSettings
from malsim.config.agent_settings import AttackerSettings
from malsim.config.sim_settings import AttackSurfaceSettings
from malsim.mal_simulator import TTCMode, AttackerState
from malsim.policies import TTCSoftMinAttacker


def test_ttc_avoider() -> None:
    """TTC Avoider"""

    scenario_file = 'tests/testdata/scenarios/ttc_lang_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attack_graph = scenario.attack_graph
    attacker_agent_name = 'TTCAvoidingAttacker'
    entry_point = attack_graph.get_node_by_full_name('Net1:easyAccess')
    goal = attack_graph.get_node_by_full_name('DataD:read')
    goals = frozenset({goal})
    sim = MalSimulator(
        scenario.attack_graph,
        sim_settings=MalSimulatorSettings(
            seed=48,
            ttc_mode=TTCMode.PRE_SAMPLE,
            attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
        ),
        agents=(
            AttackerSettings(
                name=attacker_agent_name,
                entry_points={entry_point},
                goals=goals,
            ),
        ),
    )
    attacker_agent = TTCSoftMinAttacker({})

    states = sim.agent_states
    attacker_state = states[attacker_agent_name]

    while not sim.done():
        # Run the simulation until agents are terminated/truncated
        assert isinstance(attacker_state, AttackerState)
        attacker_node = attacker_agent.get_next_action(attacker_state)

        assert attacker_node
        # Should always pick the easy path or the goal
        assert 'easy' in attacker_node.name or attacker_node in goals

        # Step
        actions = {attacker_agent_name: [attacker_node] if attacker_node else []}
        states = sim.step(actions)
        attacker_state = states[attacker_agent_name]


def test_ttc_avoider_low_sharpness() -> None:
    """TTC Avoider with low beta/sharpness explores both easy and hard paths."""

    scenario_file = 'tests/testdata/scenarios/ttc_lang_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attack_graph = scenario.attack_graph
    attacker_agent_name = 'TTCAvoidingAttacker'

    num_trials = 20
    easy_count = 0
    hard_count = 0

    for trial in range(num_trials):
        sim = MalSimulator(
            scenario.attack_graph,
            sim_settings=MalSimulatorSettings(
                seed=trial,
                ttc_mode=TTCMode.PRE_SAMPLE,
                attack_surface=AttackSurfaceSettings(skip_unnecessary=False),
            ),
            agents=(
                AttackerSettings(
                    'TTCAvoidingAttacker',
                    entry_points=frozenset(
                        {attack_graph.get_node_by_full_name('Net1:easyAccess')}
                    ),
                    goals=frozenset({attack_graph.get_node_by_full_name('DataD:read')}),
                ),
            ),
        )
        attacker_agent = TTCSoftMinAttacker({'beta': 0.1, 'seed': trial})

        states = sim.agent_states
        attacker_state = states[attacker_agent_name]

        trial_picked_easy = False
        trial_picked_hard = False
        while not sim.done():
            # Run the simulation until agents are terminated/truncated
            assert isinstance(attacker_state, AttackerState)
            attacker_node = attacker_agent.get_next_action(attacker_state)
            assert attacker_node

            trial_picked_easy |= 'easy' in attacker_node.name
            trial_picked_hard |= 'hard' in attacker_node.name

            # Step
            actions = {attacker_agent_name: [attacker_node] if attacker_node else []}
            states = sim.step(actions)
            attacker_state = states[attacker_agent_name]

        easy_count += trial_picked_easy
        hard_count += trial_picked_hard

    min_trials = num_trials * 0.8
    assert easy_count >= min_trials, (
        f'TTC avoider with low sharpness picked an easy step in only '
        f'{easy_count}/{num_trials} trials'
    )
    assert hard_count >= min_trials, (
        f'TTC avoider with low sharpness picked a hard step in only '
        f'{hard_count}/{num_trials} trials'
    )
