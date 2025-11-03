from maltoolbox.language import LanguageGraphAssociation
from maltoolbox.attackgraph import AttackGraphNode
from malsim.envs.graph.serialization import LangSerializer
from malsim.envs.graph.mal_spaces import (
    MALObs,
    MALObsInstance,
    Step,
    Asset,
    Association,
    LogicGate,
    MALObsAttackerActionSpace,
    MALObsDefenderActionSpace,
    MALAttackerObs,
    MALDefenderObs,
)
from malsim.envs.graph.utils import attacker_state2graph, defender_state2graph, create_full_obs, full_obs2attacker_obs, full_obs2defender_obs
from malsim.scenario import Scenario, AgentType
from malsim.mal_simulator import MalSimulator, MalSimAttackerState, MalSimDefenderState
import numpy as np

def test_mal_obs() -> None:

    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_attack_step_types=True
    )
    obs_space = MALObs(serializer, use_logic_gates=True)
    sim = MalSimulator.from_scenario(scenario)

    states = sim.reset()
    obs = create_full_obs(sim, serializer, use_logic_gates=True)
    assert obs in obs_space
    for _ in range(10):
        actions = {}
        for agent_name, state in states.items():
            if len(state.action_surface) > 0:
                actions[agent_name] = [list(state.action_surface)[0]]
            else:
                actions[agent_name] = []
        states = sim.step(actions)
        obs = create_full_obs(sim, serializer, use_logic_gates=True)
        assert obs in obs_space

def test_attacker_obs() -> None:
    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    attacker_name = next(agent['name'] for agent in scenario.agents if agent['type'] == AgentType.ATTACKER)
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_attack_step_types=True
    )
    sim = MalSimulator.from_scenario(scenario)
    AG = scenario.attack_graph
    full_obs = create_full_obs(sim, serializer, use_logic_gates=True)
    attacker_obs_space = MALAttackerObs(serializer, use_logic_gates=True)
    attacker_state = sim.reset()[attacker_name]
    assert isinstance(attacker_state, MalSimAttackerState)
    attacker_obs = full_obs2attacker_obs(full_obs, attacker_state, see_defense_steps=False)
    assert attacker_obs in attacker_obs_space

    while not sim.agent_is_terminated(attacker_name):
        attacker_state = sim.step({attacker_name: [list(attacker_state.action_surface)[0]]})[attacker_name]
        assert isinstance(attacker_state, MalSimAttackerState)
        attacker_obs = full_obs2attacker_obs(full_obs, attacker_state, see_defense_steps=False)
        assert attacker_obs in attacker_obs_space

        visible_assets = {node.model_asset for node in attacker_state.performed_nodes if node.model_asset} | {node.model_asset for node in attacker_state.action_surface if node.model_asset}

        for idx in range(len(attacker_obs.steps.id)):
            node = AG.nodes[attacker_obs.steps.id[idx]]
            if serializer.split_attack_step_types and node.model_asset:
                assert attacker_obs.steps.type[idx] == serializer.attack_step_type[(node.model_asset.type, node.name)]
            else:
                assert attacker_obs.steps.type[idx] == serializer.attack_step_type[(node.name,)]
            assert attacker_obs.steps.logic_class[idx] == serializer.attack_step_class[node.type]
            assert attacker_obs.steps.tags[idx] == serializer.attack_step_tag[node.tags[0] if len(node.tags) > 0 else None]
            assert attacker_obs.steps.compromised[idx] == sim.node_is_compromised(node)
            assert attacker_obs.steps.attempts is not None and attacker_obs.steps.attempts[idx] == attacker_state.num_attempts.get(node, 0)
            assert attacker_obs.steps.action_mask[idx] == sim.node_is_traversable(attacker_state.performed_nodes, node)
            assert node.model_asset in visible_assets
            children = {AG.nodes[attacker_obs.steps.id[child_idx]] for child_idx in attacker_obs.step2step[:, attacker_obs.step2step[0] == idx][1]}
            assert all(child in node.children for child in children)
            assert all(child.model_asset in visible_assets for child in children)
            parents = {AG.nodes[attacker_obs.steps.id[parent_idx]] for parent_idx in attacker_obs.step2step[:, attacker_obs.step2step[1] == idx][0]}
            assert all(parent in node.parents for parent in parents)
            assert all(parent.model_asset in visible_assets for parent in parents)


def test_defender_obs() -> None:
    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    defender_name = next(agent['name'] for agent in scenario.agents if agent['type'] == AgentType.DEFENDER)
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_attack_step_types=True
    )
    sim = MalSimulator.from_scenario(scenario)
    full_obs = create_full_obs(sim, serializer, use_logic_gates=True)
    defender_obs_space = MALDefenderObs(serializer, use_logic_gates=True)
    defender_state = sim.reset()[defender_name]
    assert isinstance(defender_state, MalSimDefenderState)
    defender_obs = full_obs2defender_obs(full_obs, defender_state)
    assert defender_obs in defender_obs_space

    while len(defender_state.action_surface) > 0:
        defender_state = sim.step({defender_name: [list(defender_state.action_surface)[0]]})[defender_name]
        assert isinstance(defender_state, MalSimDefenderState)
        defender_obs = full_obs2defender_obs(full_obs, defender_state)
        assert defender_obs in defender_obs_space

def test_jsonable() -> None:
    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    attacker = next(agent for agent in scenario.agents if agent['type'] == AgentType.ATTACKER)
    agent_name = attacker['name']
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_attack_step_types=True
    )
    obs_space = MALObs(serializer, use_logic_gates=False)
    sim = MalSimulator.from_scenario(scenario)

    state = sim.reset()[agent_name]
    assert isinstance(state, MalSimAttackerState)
    obs = attacker_state2graph(state, serializer, use_logic_gates=False)
    assert obs in obs_space

    jsonable = obs_space.to_jsonable([obs])
    obs_from_jsonable = obs_space.from_jsonable(jsonable)[0]
    assert obs_from_jsonable in obs_space

    attacker_action_space = MALObsAttackerActionSpace(sim)
    attacker_obs_idx = attacker_action_space.sample(obs.steps.action_mask)
    jsonable = attacker_action_space.to_jsonable([attacker_obs_idx])
    attacker_obs_idx_from_jsonable = attacker_action_space.from_jsonable(jsonable)[0]
    assert attacker_obs_idx_from_jsonable == attacker_obs_idx

    defender_action_space = MALObsDefenderActionSpace(sim)
    defender_obs_idx = defender_action_space.sample()
    jsonable = defender_action_space.to_jsonable([defender_obs_idx])
    defender_obs_idx_from_jsonable = defender_action_space.from_jsonable(jsonable)[0]
    assert defender_obs_idx_from_jsonable == defender_obs_idx