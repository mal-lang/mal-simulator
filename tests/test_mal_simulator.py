"""Test MalSimulator class"""

from maltoolbox.attackgraph import AttackGraphNode, AttackGraph, Attacker
from malsim.mal_simulator import MalSimulator

from malsim.scenario import load_scenario

def test_init(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    MalSimulator(attack_graph)


def test_reset(corelang_lang_graph, model):
    """Make sure attack graph is reset"""
    attack_graph = AttackGraph(corelang_lang_graph, model)

    agent_entry_point = attack_graph.get_node_by_full_name(
        'OS App:networkConnectUninspected')

    attacker_name = "testagent"

    attacker = Attacker(
        attacker_name,
        entry_points = {agent_entry_point},
        reached_attack_steps = {agent_entry_point},
        attacker_id = 100
    )

    attack_graph.add_attacker(attacker, attacker.id)

    sim = MalSimulator(attack_graph)

    attack_graph_before = sim.attack_graph
    sim.register_attacker(attacker_name, attacker.id)
    assert attacker.name in sim.agent_states
    assert len(sim.agent_states) == 1

    sim.reset()

    attack_graph_after = sim.attack_graph

    # Make sure agent was added (and not removed)
    assert attacker.name in sim.agent_states
    # Make sure the attack graph is not the same object but identical
    assert id(attack_graph_before) != id(attack_graph_after)

    for node in attack_graph_after.nodes.values():
        # Entry points are added to the nodes after backup is created
        # So they have to be removed for the graphs to be compared as identical
        if 'entrypoint' in node.extras:
            del node.extras['entrypoint']

    assert attack_graph_before._to_dict() == attack_graph_after._to_dict()

def test_register_agent_attacker(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    attack_graph.attach_attackers()
    sim = MalSimulator(attack_graph)

    agent_name = "attacker1"
    sim.register_attacker(agent_name, 0)

    assert agent_name in sim.agent_states
    assert agent_name in sim.agent_states


def test_register_agent_defender(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    agent_name = "defender1"
    sim.register_defender(agent_name)

    assert agent_name in sim.agent_states
    assert agent_name in sim.agent_states


def test_register_agent_action_surface(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    agent_name = "defender1"
    sim.register_defender(agent_name)

    sim._init_agent_action_surfaces()
    action_surface = sim.agent_states[agent_name].action_surface
    for node in action_surface:
        assert node.is_available_defense()


def test_simulator_initialize_agents(corelang_lang_graph, model):
    """Test _initialize_agents"""

    ag, _ = load_scenario('tests/testdata/scenarios/simple_scenario.yml')
    sim = MalSimulator(ag)

    # Register the agents
    attacker_name = "attacker"
    defender_name = "defender"
    sim.register_attacker(attacker_name, 1)
    sim.register_defender(defender_name)

    sim.reset()

    assert set(sim.agent_states.keys()) == {attacker_name, defender_name}


def test_get_agents():
    """Test _get_attacker_agents and _get_defender_agents"""

    ag, _ = load_scenario('tests/testdata/scenarios/simple_scenario.yml')
    sim = MalSimulator(ag)
    sim.reset()

    sim._get_attacker_agents() == ['attacker']
    sim._get_defender_agents() == ['defender']


def test_attacker_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = attack_graph.get_node_by_full_name('OS App:fullAccess')

    attacker = Attacker(
        'attacker1',
        reached_attack_steps = {entry_point},
        entry_points = {entry_point},
        attacker_id = 100
    )
    attack_graph.add_attacker(attacker, attacker.id)
    sim = MalSimulator(attack_graph)

    sim.register_attacker(attacker.name,
        attacker.id)
    sim.reset()
    attacker_agent = sim._agent_states[attacker.name]

    # Can not attack the notPresent step
    defense_step = sim.attack_graph.get_node_by_full_name('OS App:notPresent')
    actions = sim._attacker_step(attacker_agent, {defense_step})
    assert not actions
    assert not attacker_agent.step_action_surface_additions

    attack_step = sim.attack_graph.get_node_by_full_name('OS App:attemptRead')
    sim._attacker_step(attacker_agent, {attack_step})
    assert attacker_agent.step_performed_nodes  == {attack_step}
    assert attacker_agent.step_action_surface_additions == attack_step.children


def test_defender_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    defender_name = "defender"
    sim.register_defender(defender_name)
    sim.reset()

    defender_agent = sim._agent_states[defender_name]
    defense_step = sim.attack_graph.get_node_by_full_name(
        'OS App:notPresent')
    sim._defender_step(defender_agent, {defense_step})
    assert defender_agent.step_performed_nodes == {defense_step}

    # Can not defend attack_step
    attack_step = sim.attack_graph.get_node_by_full_name(
        'OS App:attemptUseVulnerability')
    sim._defender_step(defender_agent, {attack_step})
    assert not defender_agent.step_performed_nodes


def test_agent_state_views_simple(corelang_lang_graph, model):

    def get_node(full_name) -> AttackGraphNode:
        node = sim.attack_graph.get_node_by_full_name(full_name)
        assert node
        return node

    attack_graph = AttackGraph(corelang_lang_graph, model)
    entry_point = attack_graph.get_node_by_full_name('OS App:fullAccess')

    attacker = Attacker(
        'attacker1',
        reached_attack_steps = {entry_point},
        entry_points = set(),
        attacker_id = 100
    )
    attack_graph.add_attacker(attacker, attacker.id)

    # Create simulator and register agents
    sim = MalSimulator(attack_graph)
    attacker_name = 'attacker'
    defender_name = 'defender'
    sim.register_attacker(attacker_name,
        attacker.id)
    sim.register_defender(defender_name)

    # Evaluate the agent state views after reset
    state_views = sim.reset()
    asv = state_views['attacker']
    dsv = state_views['defender']
    assert asv.step_performed_nodes == set()
    assert dsv.step_performed_nodes == set()
    assert len(asv.action_surface) == 6
    assert len(dsv.action_surface) == 21
    assert dsv.step_action_surface_additions == set()
    assert asv.step_action_surface_removals == set()
    assert dsv.step_action_surface_removals == set()

    # Save all relvant nodes in variables
    program2_not_present = get_node('Program 2:notPresent')
    os_app_attempt_deny = get_node('OS App:attemptDeny')
    os_app_success_deny = get_node('OS App:successfulDeny')
    os_app_not_present = get_node('OS App:notPresent')
    os_app_access_netcon = get_node('OS App:accessNetworkAndConnections')
    os_app_spec_access = get_node('OS App:specificAccess')

    # Evaluate the agent state views after stepping through an attack step and
    # a defense that will not impact it in any way

    state_views = sim.step({
        'defender': [program2_not_present],
        'attacker': [os_app_attempt_deny]
    })
    asv = state_views['attacker']
    dsv = state_views['defender']
    assert asv.step_performed_nodes == {os_app_attempt_deny}
    assert dsv.step_performed_nodes == {program2_not_present}
    assert asv.step_action_surface_additions == {os_app_success_deny}
    assert dsv.step_action_surface_additions == set()
    assert asv.step_action_surface_removals == {os_app_attempt_deny}
    assert os_app_attempt_deny not in asv.action_surface
    assert dsv.step_action_surface_removals == {program2_not_present}
    assert dsv.step_all_compromised_nodes == {os_app_attempt_deny}
    assert len(dsv.step_unviable_nodes) == 49

    # Go through an attack step that already has some children in the attack
    # surface(OS App:accessNetworkAndConnections in this case)
    assert os_app_access_netcon in asv.action_surface
    state_views = sim.step({
        'defender': [],
        'attacker': [os_app_spec_access]
    })
    asv = state_views['attacker']
    dsv = state_views['defender']
    assert asv.step_performed_nodes == {os_app_spec_access}
    assert dsv.step_performed_nodes == set()
    assert os_app_access_netcon in asv.action_surface
    assert os_app_access_netcon not in asv.step_action_surface_additions
    assert dsv.step_action_surface_additions == set()
    assert asv.step_action_surface_removals == {os_app_spec_access}
    assert os_app_spec_access not in asv.action_surface
    assert dsv.step_action_surface_removals == set()
    assert dsv.step_all_compromised_nodes == {os_app_spec_access}
    assert len(dsv.step_unviable_nodes) == 0

    # Evaluate the agent state views after stepping through an attack step and
    # a defense that would prevent it from occurring
    state_views = sim.step({
        'defender': [os_app_not_present],
        'attacker': [os_app_success_deny]
    })
    asv = state_views['attacker']
    dsv = state_views['defender']
    assert asv.step_performed_nodes == set()
    assert dsv.step_performed_nodes == {os_app_not_present}
    assert asv.step_action_surface_additions == set()
    assert dsv.step_action_surface_additions == set()
    assert len(asv.step_action_surface_removals) == 12
    assert dsv.step_action_surface_removals == {os_app_not_present}
    assert dsv.step_all_compromised_nodes == set()
    assert len(dsv.step_unviable_nodes) == 55


def test_observe_attacker():
    attack_graph, _ = load_scenario(
        'tests/testdata/scenarios/simple_scenario.yml'
    )

    # Create the simulator
    sim = MalSimulator(attack_graph)

    # Register the agents
    attacker_agent_id = "attacker"
    defender_agent_id = "defender"

    sim.register_attacker(attacker_agent_id, 1)
    sim.register_defender(defender_agent_id)
    sim.reset()

    # Make alteration to the attack graph attacker
    assert len(sim.attack_graph.attackers) == 1
    attacker = next(iter(sim.attack_graph.attackers.values()))
    assert len(attacker.reached_attack_steps) == 1


def test_step_attacker_defender_action_surface_updates():
    ag, _ = load_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml')

    sim = MalSimulator(ag)
    # Register the agents
    attacker_agent_id = "attacker"
    defender_agent_id = "defender"

    sim.register_attacker(attacker_agent_id, 1)
    sim.register_defender(defender_agent_id)

    sim.reset()

    attacker_agent = sim.agent_states[attacker_agent_id]
    defender_agent = sim.agent_states[defender_agent_id]

    # Run step() with action crafted in test
    attacker_step = sim.attack_graph.get_node_by_full_name('User:3:compromise')
    assert attacker_step in attacker_agent.action_surface

    defender_step = sim.attack_graph.get_node_by_full_name('User:3:notPresent')
    assert defender_step in defender_agent.action_surface

    actions = {
        attacker_agent.name: [attacker_step],
        defender_agent.name: [defender_step]
    }

    sim.step(actions)

    # Make sure no nodes added to action surface
    assert not attacker_agent.step_action_surface_additions
    assert not defender_agent.step_action_surface_additions

    # Make sure the steps are removed from the action surfaces
    assert attacker_step in attacker_agent.step_action_surface_removals
    assert defender_step in defender_agent.step_action_surface_removals

    assert attacker_step not in attacker_agent.action_surface
    assert defender_step not in defender_agent.action_surface


def test_default_simulator_default_settings_eviction():
    """Test attacker node eviction using MalSimulatorSettings default"""
    ag, _ = load_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml',
    )

    sim = MalSimulator(ag)

    # Register the agents
    # Register the agents
    attacker_agent_id = "attacker"
    defender_agent_id = "defender"

    sim.register_attacker(attacker_agent_id, 1)
    sim.register_defender(defender_agent_id)
    sim.reset()
    attacker = next(iter(sim.attack_graph.attackers.values()))

    # Get a step to compromise and its defense parent
    user_3_compromise = sim.attack_graph.get_node_by_full_name('User:3:compromise')
    assert attacker not in user_3_compromise.compromised_by
    user_3_compromise_defense = next(n for n in user_3_compromise.parents if n.type=='defense')
    assert not user_3_compromise_defense.is_enabled_defense()

    # First let the attacker compromise User:3:compromise
    actions = {
        attacker_agent_id: [user_3_compromise],
        defender_agent_id: []
    }
    sim.step(actions)

    # Check that the compromise happened and that the defense did not
    assert attacker in user_3_compromise.compromised_by
    assert not user_3_compromise_defense.is_enabled_defense()

    # Now let the defender defend, and the attacker waits
    actions = {
        attacker_agent_id: [],
        defender_agent_id: [user_3_compromise_defense]
    }
    sim.step(actions)

    # Verify defense was performed and attacker NOT kicked out
    assert user_3_compromise_defense.is_enabled_defense()
    assert attacker in user_3_compromise.compromised_by
