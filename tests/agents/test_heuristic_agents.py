from unittest.mock import MagicMock
from maltoolbox.attackgraph import AttackGraphNode, Attacker
from maltoolbox.language import LanguageGraph
from malsim.mal_simulator import MalSimAgentStateView
from malsim.agents import (
    DefendCompromisedDefender,
    DefendFutureCompromisedDefender
)

def test_defend_compromised_defender(
        dummy_lang_graph: LanguageGraph
    ) -> None:
    r"""
            node1          node2
            /    \         /   \
        node3       node4        node5

    """
    dummy_or_attack_step = (
        dummy_lang_graph.assets['DummyAsset']
        .attack_steps['DummyOrAttackStep']
    )

    # Create nodes
    node1 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=1)
    node2 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=2)

    node3 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=3)
    node4 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=4)
    node5 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=5)

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node1.children.add(node3)
    node3.parents.add(node1)
    node1.children.add(node4)
    node4.parents.add(node1)

    node2.children.add(node4)
    node4.parents.add(node2)
    node2.children.add(node5)
    node5.parents.add(node5)

    # Set up an attacker
    attacker = Attacker(name="TestAttacker")
    attacker.compromise(node4)

    # Set up a mock MalSimAgentState
    agent = MagicMock()
    agent.action_surface = [node1, node2]

    # Set up MalSimAgentStateView
    agent_view = MalSimAgentStateView(agent)

    # Configure BreadthFirstAttacker
    agent_config = {"seed": 42, "randomize": False}
    defender_ai = DefendCompromisedDefender(agent_config)

    # Should pick cheapest one
    node1.extras['reward'] = 100
    node2.extras['reward'] = 10

    # Get next action
    action_node = defender_ai.get_next_action(agent_view)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node2.id

    # Should pick cheapest one
    node1.extras['reward'] = 10
    node2.extras['reward'] = 100

    # Get next action
    action_node = defender_ai.get_next_action(agent_view)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node1.id


def test_defend_future_compromised_defender(
        dummy_lang_graph: LanguageGraph
    ) -> None:
    r"""
            node1              node2
            /    \             /     \
        node3      node4       |    node5
                      |        |
                       \      /
                         node 6
    """

    dummy_or_attack_step = (
        dummy_lang_graph.assets['DummyAsset']
        .attack_steps['DummyOrAttackStep']
    )

    # Create nodes
    node1 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=1)
    node2 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=2)

    node3 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=3)
    node4 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=4)
    node5 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=5)
    node6 = AttackGraphNode(lg_attack_step=dummy_or_attack_step, node_id=6)

    # Connect nodes (Node1 -> Node2 -> Node3 -> Node4)
    node1.children.add(node3)
    node3.parents.add(node1)
    node1.children.add(node4)
    node4.parents.add(node1)

    node2.children.add(node6)
    node6.parents.add(node2)
    node2.children.add(node5)
    node5.parents.add(node5)

    node4.children.add(node6)
    node6.parents.add(node4)

    # Set up an attacker
    attacker = Attacker(name="TestAttacker")
    attacker.compromise(node4)

    # Set up a mock MalSimAgentState
    agent = MagicMock()
    agent.action_surface = [node1, node2]

    # Set up MalSimAgentStateView
    agent_view = MalSimAgentStateView(agent)

    # Configure BreadthFirstAttacker
    agent_config = {"seed": 42, "randomize": False}
    defender_ai = DefendFutureCompromisedDefender(agent_config)

    # Should pick node 2 either way
    action_node = defender_ai.get_next_action(agent_view)
    assert action_node is not None, "Action node shouldn't be None"
    assert action_node.id == node2.id
