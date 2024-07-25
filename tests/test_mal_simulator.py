"""Test MalSimulator class"""
from maltoolbox.attackgraph import AttackGraph
from malsim.sims.mal_simulator import MalSimulator

def test_malsimulator(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    MalSimulator(corelang_lang_graph, model, attack_graph)


def test_malsimulator_num_assets(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    assert (
        sim.num_assets == len(corelang_lang_graph.assets) + sim.offset
    )


def test_malsimulator_num_step_names(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    assert (
        sim.num_step_names == len(
            corelang_lang_graph.attack_steps
        ) + sim.offset
    )


def test_malsimulator_asset_type(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    step = attack_graph.nodes[0]
    assert (
        sim.asset_type(step) == \
            sim._asset_type_to_index[step.asset.type] + sim.offset
    )

def test_malsimulator_step_name(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    step = attack_graph.nodes[0]
    assert (
        sim.step_name(step) == \
            sim._step_name_to_index[step.asset.type + ":" + step.name] + sim.offset
    )


def test_malsimulator_asset_id(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    step = attack_graph.nodes[0]
    assert sim.asset_id(step) == int(step.asset.id)


def test_malsimulator_create_blank_observation(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    num_objects = len(attack_graph.nodes)
    blank_observation = sim.create_blank_observation()
    assert len(blank_observation['is_observable']) == num_objects
    assert len(blank_observation['observed_state']) == num_objects
    assert len(blank_observation['remaining_ttc']) == num_objects
    assert len(blank_observation['asset_type']) == num_objects
    assert len(blank_observation['asset_id']) == num_objects
    assert len(blank_observation['step_name']) == num_objects
    expected_num_edges = sum([1 for step in attack_graph.nodes
                                for child in step.children] +
                                # We expect all defenses again (reversed)
                             [1 for step in attack_graph.nodes
                                for child in step.children
                                if step.type == "defense"])
    assert len(blank_observation['edges']) == expected_num_edges


def test_malsimulator_format_info(corelang_lang_graph, model):
    """Make sure format info works as expected"""
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)

    # Preparations of info to send to _format_info
    can_wait = {"attacker": 0, "defender": 1}
    infos = {}
    agent = "attacker1"
    agent_type = "attacker"
    can_act = 1
    available_actions = [0] * len(attack_graph.nodes)
    available_actions[0] = 1  # Only first action is available

    infos[agent] = {
        "action_mask": (
            [can_wait[agent_type], can_act],
            available_actions
        )
    }
    formatted = sim._format_info(infos[agent])
    assert formatted == "Can act? Yes\n0\n"

    # Add an action and change 'can_act' to false
    available_actions[1] = 1  # Also second action is available
    can_act = False
    infos[agent] = {
        "action_mask": (
            [can_wait[agent_type], can_act],
            available_actions
        )
    }
    formatted = sim._format_info(infos[agent])
    assert formatted == "Can act? No\n0\n1\n"


def test_malsimulator_observation_space(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    sim.observation_space()  # TODO: agent param needed?

def test_malsimulator_action_space(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    sim.action_space()


def test_malsimulator_reset(corelang_lang_graph, model):
    attack_graph = AttackGraph(corelang_lang_graph, model)
    sim = MalSimulator(corelang_lang_graph, model, attack_graph)
    sim.reset()

# def test_malsimulator_init(corelang_lang_graph, model):
#     attack_graph = AttackGraph(corelang_lang_graph, model)
#     MalSimulator(corelang_lang_graph, model, attack_graph)


# def test_malsimulator_register_attacker(corelang_lang_graph, model):
#     attack_graph = AttackGraph(corelang_lang_graph, model)
#     MalSimulator(corelang_lang_graph, model, attack_graph)


# def test_malsimulator_register_defender(corelang_lang_graph, model):
#     attack_graph = AttackGraph(corelang_lang_graph, model)
#     MalSimulator(corelang_lang_graph, model, attack_graph)


# def test_malsimulator_attacker_step(corelang_lang_graph, model):
#     attack_graph = AttackGraph(corelang_lang_graph, model)
#     MalSimulator(corelang_lang_graph, model, attack_graph)


# def test_malsimulator_defender_step(corelang_lang_graph, model):
#     attack_graph = AttackGraph(corelang_lang_graph, model)
#     MalSimulator(corelang_lang_graph, model, attack_graph)


# def test_malsimulator_observe_attacker(corelang_lang_graph, model):
#     attack_graph = AttackGraph(corelang_lang_graph, model)
#     MalSimulator(corelang_lang_graph, model, attack_graph)


# def test_malsimulator_observe_defender(corelang_lang_graph, model):
#     attack_graph = AttackGraph(corelang_lang_graph, model)
#     MalSimulator(corelang_lang_graph, model, attack_graph)


# def test_malsimulator_observe_and_reward(corelang_lang_graph, model):
#     attack_graph = AttackGraph(corelang_lang_graph, model)
#     MalSimulator(corelang_lang_graph, model, attack_graph)


# def test_malsimulator_update_viability_with_eviction(corelang_lang_graph, model):
#     attack_graph = AttackGraph(corelang_lang_graph, model)
#     MalSimulator(corelang_lang_graph, model, attack_graph)


# def test_malsimulator_step(corelang_lang_graph, model):
#     attack_graph = AttackGraph(corelang_lang_graph, model)
#     MalSimulator(corelang_lang_graph, model, attack_graph)
