"""Tests for analyzers"""

from malsim.config.sim_settings import MalSimulatorSettings
from maltoolbox.language import LanguageGraph
from maltoolbox.attackgraph import AttackGraph

from malsim.mal_simulator.graph_state import GraphState
from malsim.mal_simulator.graph_utils import node_is_blocked
from malsim.mal_simulator.simulator_state import MalSimulatorState

# Tests


def test_node_is_blocked(dummy_lang_graph: LanguageGraph) -> None:
    """Make sure nodes defended by parents are blocked"""

    impossible_attack_steps = set()
    attack_graph = AttackGraph(dummy_lang_graph)
    dummy_attack_steps = dummy_lang_graph.assets['DummyAsset'].attack_steps

    # Prepare types
    exist_attack_step_type = dummy_attack_steps['DummyExistAttackStep']
    not_exist_attack_step_type = dummy_attack_steps['DummyNotExistAttackStep']
    defense_step_type = dummy_attack_steps['DummyDefenseAttackStep']
    and_attack_step_type = dummy_attack_steps['DummyAndAttackStep']
    or_attack_step_type = dummy_attack_steps['DummyOrAttackStep']

    # exists, existance_status = False -> blocks child
    exist_node = attack_graph.add_node(exist_attack_step_type)
    exist_node.existence_status = False

    # notExists, existence_status = True -> blocks child
    not_exist_node = attack_graph.add_node(not_exist_attack_step_type)
    not_exist_node.existence_status = True

    # Defense -> blocks child when enabled
    defense_step_node = attack_graph.add_node(defense_step_type)

    and_node_blocked_by_defense = attack_graph.add_node(and_attack_step_type)
    and_node_blocked_by_defense.parents.add(defense_step_node)

    and_node_blocked_by_exist = attack_graph.add_node(and_attack_step_type)
    and_node_blocked_by_exist.parents.add(exist_node)

    and_node_blocked_by_not_exist = attack_graph.add_node(and_attack_step_type)
    and_node_blocked_by_not_exist.parents.add(not_exist_node)

    or_node_blocked_by_defense = attack_graph.add_node(or_attack_step_type)
    or_node_blocked_by_defense.parents.add(defense_step_node)

    or_node_blocked_by_exist = attack_graph.add_node(or_attack_step_type)
    or_node_blocked_by_exist.parents.add(exist_node)

    or_node_blocked_by_not_exist = attack_graph.add_node(or_attack_step_type)
    or_node_blocked_by_not_exist.parents.add(not_exist_node)

    # Or node with any parent that is not blocking will not be blocked
    or_node_not_blocked = attack_graph.add_node(or_attack_step_type)
    or_node_not_blocked.parents.add(
        and_node_blocked_by_defense
    )  # the and node is blocked but not the or node

    impossible_and_node = attack_graph.add_node(and_attack_step_type)
    impossible_attack_steps.add(impossible_and_node)
    impossible_attack_steps.add(impossible_and_node)

    # Make sure unviable
    enabled_defenses = {defense_step_node}

    sim_state = MalSimulatorState(
        attack_graph=attack_graph,
        settings=MalSimulatorSettings(),
        graph_state=GraphState({}, enabled_defenses, impossible_attack_steps, {}),
        enabled_defenses=enabled_defenses,
    )

    assert node_is_blocked(sim_state, and_node_blocked_by_defense)
    assert node_is_blocked(sim_state, and_node_blocked_by_exist)
    assert node_is_blocked(sim_state, and_node_blocked_by_not_exist)
    assert node_is_blocked(sim_state, or_node_blocked_by_defense)
    assert node_is_blocked(sim_state, or_node_blocked_by_exist)
    assert node_is_blocked(sim_state, or_node_blocked_by_not_exist)
    assert not node_is_blocked(sim_state, or_node_not_blocked)
    assert node_is_blocked(sim_state, impossible_and_node)
