from collections import Counter
from typing import Any
from malsim.scenario.scenario import Scenario
from maltoolbox.attackgraph import AttackGraph, AttackGraphNode
from maltoolbox.language import LanguageGraph
from maltoolbox.model import Model
from malsim.config.agent_settings import AttackerSettings, DefenderSettings
from malsim.config.node_property_rule import NodePropertyRule
from malsim.mal_simulator import MalSimulator, MalSimDefenderState
from malsim.policies import (
    DefendCompromisedDefender,
    DefendFutureCompromisedDefender,
    RandomAgent,
)
import numpy as np


def test_defend_compromised_defender(dummy_lang_graph: LanguageGraph) -> None:
    r"""
    node0 -------+---------|
     |   node1   |  node2  |
     |   /    \  |  /   \  |
    node3      node4    node5

    """
    dummy_and_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyAndAttackStep'
    ]
    dummy_defense_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyDefenseAttackStep'
    ]

    dummy_model = Model('Dummy model', dummy_lang_graph)
    dummy_asset0 = dummy_model.add_asset('DummyAsset')
    dummy_asset1 = dummy_model.add_asset('DummyAsset')
    dummy_asset2 = dummy_model.add_asset('DummyAsset')
    dummy_asset3 = dummy_model.add_asset('DummyAsset')
    dummy_asset4 = dummy_model.add_asset('DummyAsset')
    dummy_asset5 = dummy_model.add_asset('DummyAsset')

    # Create attack graph with nodes
    ag = AttackGraph(dummy_lang_graph)
    node0 = ag.add_node(
        lg_attack_step=dummy_and_attack_step, model_asset=dummy_asset0, node_id=0
    )
    node1 = ag.add_node(
        lg_attack_step=dummy_defense_attack_step, model_asset=dummy_asset1, node_id=1
    )
    node2 = ag.add_node(
        lg_attack_step=dummy_defense_attack_step, model_asset=dummy_asset2, node_id=2
    )
    node3 = ag.add_node(
        lg_attack_step=dummy_and_attack_step, model_asset=dummy_asset3, node_id=3
    )
    node4 = ag.add_node(
        lg_attack_step=dummy_and_attack_step, model_asset=dummy_asset4, node_id=4
    )
    node5 = ag.add_node(
        lg_attack_step=dummy_and_attack_step, model_asset=dummy_asset5, node_id=5
    )

    # Connect nodes (Node1 -> Node3, Node4, Node5)
    node0.children.add(node3)
    node3.parents.add(node0)
    node0.children.add(node4)
    node4.parents.add(node0)
    node0.children.add(node5)
    node5.parents.add(node0)

    # Connect nodes (Node1 -> Node3, Node4)
    node1.children.add(node3)
    node3.parents.add(node1)
    node1.children.add(node4)
    node4.parents.add(node1)

    # Connect nodes (Node2 -> Node4, Node5)
    node2.children.add(node4)
    node4.parents.add(node2)
    node2.children.add(node5)
    node5.parents.add(node2)

    sim = MalSimulator(
        ag,
        agents=(
            AttackerSettings(name='bfs', entry_points=frozenset({node4})),
            DefenderSettings(name='def_comp'),
        ),
    )

    # Set up a defender
    agent_state = sim.agent_states['def_comp']

    # Configure BreadthFirstAttacker
    agent_config = {'seed': 42, 'randomize': False}
    defender_ai = DefendCompromisedDefender(agent_config)

    # Should pick cheapest one
    sim.agent_settings['def_comp'].rewards = NodePropertyRule.from_attack_step_dict(
        {node1: 100, node2: 10}
    )
    # Get next action
    assert isinstance(agent_state, MalSimDefenderState)
    action_node = defender_ai.get_next_action(agent_state)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node2.id

    # Should pick cheapest one
    sim.agent_settings['def_comp'].rewards = NodePropertyRule.from_attack_step_dict(
        {node1: 10, node2: 100}
    )

    # Get next action
    action_node = defender_ai.get_next_action(agent_state)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node1.id


def test_defend_future_compromised_defender(dummy_lang_graph: LanguageGraph) -> None:
    r"""
    node0 -------+-----------|
     |   node1   |    node2  |
     |   /    \  |    /   \  |
    node3      node4  |   node5
                 |    |
                 \    /
                 node 6
    """

    dummy_and_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyAndAttackStep'
    ]
    dummy_defense_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyDefenseAttackStep'
    ]

    # Create attack graph with nodes
    ag = AttackGraph(dummy_lang_graph)
    node0 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=0)
    node1 = ag.add_node(lg_attack_step=dummy_defense_attack_step, node_id=1)
    node2 = ag.add_node(lg_attack_step=dummy_defense_attack_step, node_id=2)
    node3 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=3)
    node4 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=4)
    node5 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=5)
    node6 = ag.add_node(lg_attack_step=dummy_and_attack_step, node_id=6)

    # Connect nodes (Node1 -> Node3, Node4, Node5)
    node0.children.add(node3)
    node3.parents.add(node0)
    node0.children.add(node4)
    node4.parents.add(node0)
    node0.children.add(node5)
    node5.parents.add(node0)

    # Connect nodes (Node1 -> Node3, Node4)
    node1.children.add(node3)
    node3.parents.add(node1)
    node1.children.add(node4)
    node4.parents.add(node1)

    # Connect nodes (Node2 -> Node5, Node6)
    node2.children.add(node5)
    node5.parents.add(node2)
    node2.children.add(node6)
    node6.parents.add(node2)

    # Connect nodes (Node4 -> Node6)
    node4.children.add(node6)
    node6.parents.add(node4)

    sim = MalSimulator(
        ag,
        agents=(
            AttackerSettings(name='bfs', entry_points=frozenset({node4})),
            DefenderSettings(name='def_future_comp'),
        ),
    )

    agent_state = sim.agent_states['def_future_comp']

    # Configure BreadthFirstAttacker
    agent_config = {'seed': 42, 'randomize': False}
    defender_ai = DefendFutureCompromisedDefender(agent_config)

    # Should pick node 2 either way
    assert isinstance(agent_state, MalSimDefenderState)
    action_node = defender_ai.get_next_action(agent_state)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node2.id


def test_random_agent(dummy_lang_graph: LanguageGraph) -> None:
    dummy_or_attack_step = dummy_lang_graph.assets['DummyAsset'].attack_steps[
        'DummyOrAttackStep'
    ]

    # Create attack graph with nodes
    ag = AttackGraph(dummy_lang_graph)
    node0 = ag.add_node(lg_attack_step=dummy_or_attack_step, node_id=0)
    node1 = ag.add_node(lg_attack_step=dummy_or_attack_step, node_id=1)
    node2 = ag.add_node(lg_attack_step=dummy_or_attack_step, node_id=2)
    node3 = ag.add_node(lg_attack_step=dummy_or_attack_step, node_id=3)

    # Connect node0 to all other nodes
    node0.children.add(node1)
    node1.parents.add(node0)
    node0.children.add(node2)
    node2.parents.add(node0)
    node0.children.add(node3)
    node3.parents.add(node0)

    sim = MalSimulator(
        ag,
        agents=(
            AttackerSettings(name='random_attacker', entry_points=frozenset({node0})),
        ),
    )

    # Set up an attacker
    attacker_name = 'random_attacker'

    agent_config: dict[str, Any] = {}
    attacker_ai = RandomAgent(agent_config)
    actions = []
    for _ in range(1_500):
        state = sim.reset()[attacker_name]
        action_node = attacker_ai.get_next_action(state)
        actions.append(action_node)
    counter = Counter(actions)
    counts = np.array(list(counter.values()))
    probs = counts / counts.sum()
    assert np.allclose(probs, 1 / len(counter), atol=0.04)

    agent_config = {'wait_prob': 0.05}
    attacker_ai = RandomAgent(agent_config)
    actions = []
    for _ in range(1_500):
        state = sim.reset()[attacker_name]
        action_node = attacker_ai.get_next_action(state)
        actions.append(action_node)
    counter = Counter(actions)
    wait_prob = counter[None] / len(actions)
    assert abs(wait_prob - agent_config['wait_prob']) < 0.04


def test_random_agent_wait_prob() -> None:
    scenario = Scenario.load_from_file(
        'tests/testdata/scenarios/simple_random_defender.yml'
    )
    sim = MalSimulator.from_scenario(scenario)
    defender_name = 'Defender'
    wait_prob = scenario.defender_settings[defender_name].config['wait_prob']
    defender_ai = scenario.defender_settings[defender_name].agent
    assert isinstance(defender_ai, RandomAgent)

    actions = []
    for _ in range(100):
        state = sim.reset()
        while len(state[defender_name].action_surface) > 0:
            action = defender_ai.get_next_action(state[defender_name])
            actions.append(action)
            step_actions: list[AttackGraphNode] = [action] if action else []
            state = sim.step({defender_name: step_actions})

    counter = Counter(actions)
    wait_count = counter[None]
    wait_freq = wait_count / len(actions)
    assert abs(wait_freq - wait_prob) < 0.01
