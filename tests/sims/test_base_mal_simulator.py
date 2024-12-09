"""Test MalSimulator class"""

from maltoolbox.attackgraph import AttackGraph, Attacker
from malsim.sims.mal_simulator import MalSimulator
from malsim.agents.agent_base import PassiveDefender, PassiveAttacker, AgentType
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
        entry_points=[agent_entry_point],
        reached_attack_steps=[agent_entry_point]
    )

    attack_graph.add_attacker(attacker, attacker.id)

    attacker_agent = PassiveAttacker(attacker.name, attacker.id)
    sim = MalSimulator(attack_graph)

    attack_graph_before = sim.attack_graph
    sim.register_agent(attacker_agent)
    assert attacker.name in sim.agents_dict
    assert len(sim.agents) == 1

    sim.reset()

    attack_graph_after = sim.attack_graph

    # Make sure agent was added (and not removed)
    assert attacker.name in [a.name for a in sim.agents]
    # Make sure the attack graph is not the same object but identical
    assert id(attack_graph_before) != id(attack_graph_after)

    for node in attack_graph_after.nodes:
        # Entry points are added to the nodes after backup is created
        # So they have to be removed for the graphs to be compared as identical
        if 'entrypoint' in node.extras:
            del node.extras['entrypoint']

    assert attack_graph_before._to_dict() == attack_graph_after._to_dict()

def test_register_agent_attacker(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    attacker = 1
    agent = PassiveAttacker("attacker1", attacker)
    sim.register_agent(agent)

    assert agent.name in sim.agents_dict
    assert agent.name in [a.name for a in sim.agents]


def test_register_agent_defender(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    agent = PassiveDefender("defender1")
    sim.register_agent(agent)

    assert agent.name in sim.agents_dict
    assert agent.name in [a.name for a in sim.agents]


def test_simulator_initialize_agents(corelang_lang_graph, model):
    """Test _initialize_agents"""

    ag, _ = load_scenario('tests/testdata/scenarios/simple_scenario.yml')
    sim = MalSimulator(ag)

    # Register the agents
    attacker = PassiveAttacker("attacker", 1)
    defender = PassiveDefender("defender")
    sim.register_agent(attacker)
    sim.register_agent(defender)

    sim.reset()

    assert set(sim.agents_dict.keys()) == {'defender', 'attacker'}


def test_get_agents():
    """Test get_attacker_agents and get_defender_agents"""

    ag, _ = load_scenario('tests/testdata/scenarios/simple_scenario.yml')
    sim = MalSimulator(ag)
    sim.reset()

    sim.get_attacker_agents() == ['attacker']
    sim.get_defender_agents() == ['defender']


def test_attacker_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)

    attacker = Attacker('attacker1', id=0)
    attack_graph.add_attacker(attacker, attacker.id)
    sim = MalSimulator(attack_graph)

    attacker_agent = PassiveAttacker(attacker.name, attacker.id)
    sim.register_agent(attacker_agent)
    sim.reset()
    attacker_agent = sim.agents_dict[attacker.name]

    # Can not attack the notPresent step
    defense_step = attack_graph.get_node_by_full_name('OS App:notPresent')
    actions = sim._attacker_step(attacker_agent, [defense_step])
    assert not actions

    # Can attack the attemptUseVulnerability step!
    attack_step = attack_graph.get_node_by_full_name('OS App:attemptUseVulnerability')
    actions = sim._attacker_step(attacker_agent, [attack_step])
    assert actions == [attack_step]


def test_defender_step(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(attack_graph)

    defender_name = "defender"
    defender_agent = PassiveDefender(defender_name)
    sim.register_agent(defender_agent)
    sim.reset()

    defense_step = attack_graph.get_node_by_full_name(
        'OS App:notPresent')
    actions, _ = sim._defender_step(defender_agent, [defense_step])
    assert actions == [defense_step]

    # Can not defend attack_step
    attack_step = attack_graph.get_node_by_full_name(
        'OS App:attemptUseVulnerability')
    actions, _ = sim._defender_step(defender_agent, [attack_step])
    assert not actions


def test_observe_attacker():
    attack_graph, _ = load_scenario(
        'tests/testdata/scenarios/simple_scenario.yml'
    )

    # Create the simulator
    sim = MalSimulator(attack_graph)

    # Register the agents
    attacker_agent_id = "attacker"
    attacker_agent = PassiveAttacker(attacker_agent_id, 1)
    defender_agent_id = "defender"
    defender_agent = PassiveDefender(defender_agent_id)

    sim.register_agent(attacker_agent)
    sim.register_agent(defender_agent)
    sim.reset()

    # Make alteration to the attack graph attacker
    assert len(sim.attack_graph.attackers) == 1
    attacker = sim.attack_graph.attackers[0]
    assert len(attacker.reached_attack_steps) == 1


def test_agents_registered():

    ag, _ = load_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml')
    sim = MalSimulator(ag)

    # Register the agents
    attacker_agent_id = "attacker"
    attacker_agent = PassiveAttacker(attacker_agent_id, ag.attackers[0].id)
    defender_agent_id = "defender"
    defender_agent = PassiveDefender(defender_agent_id)

    sim.register_agent(attacker_agent)
    sim.register_agent(defender_agent)

    # We need to reinitialize to initialize agents
    sim.reset()
    # assert set(obs.keys()) == {attacker_name, defender_name}
    # assert set(infos.keys()) == {attacker_name, defender_name}


def test_step_attacker_defender_action_surface_updates():
    ag, _ = load_scenario(
        'tests/testdata/scenarios/traininglang_scenario.yml')

    sim = MalSimulator(ag)
    # Register the agents
    attacker_agent_id = "attacker"
    attacker_agent = PassiveAttacker(attacker_agent_id, ag.attackers[0].id)
    defender_agent_id = "defender"
    defender_agent = PassiveDefender(defender_agent_id)

    sim.register_agent(attacker_agent)
    sim.register_agent(defender_agent)

    attacker_agent = next(iter(sim.get_attacker_agents()))
    defender_agent = next(iter(sim.get_defender_agents()))

    sim.reset()

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
    attacker_agent = PassiveAttacker(attacker_agent_id, ag.attackers[0].id)
    defender_agent_id = "defender"
    defender_agent = PassiveDefender(defender_agent_id)

    sim.register_agent(attacker_agent)
    sim.register_agent(defender_agent)
    sim.reset()
    attacker = sim.attack_graph.attackers[0]

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
