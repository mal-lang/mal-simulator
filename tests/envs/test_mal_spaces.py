from malsim.envs.graph.serialization import LangSerializer
from malsim.envs.graph.mal_spaces import (
    AssetThenAttackerAction,
    AssetThenDefenderAction,
    AttackerActionThenAsset,
    MALObs,
    ActionThenAsset,
    MALObsAttackStepSpace,
    MALAttackerObs,
    MALDefenderObs,
    AssetThenAction,
)
from malsim.envs.graph.utils import (
    create_full_obs,
    full_obs2attacker_obs,
    full_obs2defender_obs,
)
from malsim.scenario import Scenario, AgentType
from malsim.mal_simulator import MalSimulator, MalSimAttackerState, MalSimDefenderState


def test_mal_obs() -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_step_types=True
    )
    obs_space = MALObs(serializer)
    sim = MalSimulator.from_scenario(scenario)

    states = sim.reset()
    obs = create_full_obs(sim, serializer)
    assert obs in obs_space
    for _ in range(10):
        actions = {}
        for agent_name, state in states.items():
            if len(state.action_surface) > 0:
                actions[agent_name] = [list(state.action_surface)[0]]
            else:
                actions[agent_name] = []
        states = sim.step(actions)
        obs = create_full_obs(sim, serializer)
        assert obs in obs_space


def test_attacker_obs() -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attacker_name = next(
        agent.name
        for agent in scenario.agent_settings.values()
        if agent.type == AgentType.ATTACKER
    )
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_step_types=True
    )
    sim = MalSimulator.from_scenario(scenario)
    AG = scenario.attack_graph
    full_obs = create_full_obs(sim, serializer)
    attacker_obs_space = MALAttackerObs(serializer)
    attacker_state = sim.reset()[attacker_name]
    assert isinstance(attacker_state, MalSimAttackerState)
    attacker_obs = full_obs2attacker_obs(full_obs, attacker_state, serializer)
    assert attacker_obs in attacker_obs_space

    while not sim.agent_is_terminated(attacker_name):
        attacker_state = sim.step(
            {attacker_name: [list(attacker_state.action_surface)[0]]}
        )[attacker_name]
        assert isinstance(attacker_state, MalSimAttackerState)
        attacker_obs = full_obs2attacker_obs(full_obs, attacker_state, serializer)
        assert attacker_obs in attacker_obs_space

        visible_assets = {
            node.model_asset
            for node in attacker_state.performed_nodes
            if node.model_asset
        } | {
            node.model_asset
            for node in attacker_state.action_surface
            if node.model_asset
        }

        for idx in range(len(attacker_obs.steps.id)):
            node = AG.nodes[attacker_obs.steps.id[idx]]
            if serializer.split_step_types and node.model_asset:
                assert (
                    attacker_obs.steps.type[idx]
                    == serializer.step_type2attacker_step_type[
                        serializer.step_type[(node.model_asset.type, node.name)]
                    ]
                )
            else:
                assert (
                    attacker_obs.steps.type[idx]
                    == serializer.step_type2attacker_step_type[
                        serializer.step_type[(node.name,)]
                    ]
                )
            assert (
                attacker_obs.steps.logic_class[idx] == serializer.step_class[node.type]
            )
            assert (
                attacker_obs.steps.tags[idx]
                == serializer.step_tag[node.tags[0] if len(node.tags) > 0 else None]
            )
            assert attacker_obs.steps.compromised[idx] == sim.node_is_compromised(node)
            assert (
                attacker_obs.steps.attempts is not None
                and attacker_obs.steps.attempts[idx]
                == attacker_state.num_attempts.get(node, 0)
            )
            assert attacker_obs.steps.action_mask[idx] == (
                node in attacker_state.action_surface
            )
            assert node.model_asset in visible_assets
            children = {
                AG.nodes[attacker_obs.steps.id[child_idx]]
                for child_idx in attacker_obs.step2step[
                    :, attacker_obs.step2step[0] == idx
                ][1]
            }
            assert all(child in node.children for child in children)
            assert all(child.model_asset in visible_assets for child in children)
            parents = {
                AG.nodes[attacker_obs.steps.id[parent_idx]]
                for parent_idx in attacker_obs.step2step[
                    :, attacker_obs.step2step[1] == idx
                ][0]
            }
            assert all(parent in node.parents for parent in parents)
            assert all(parent.model_asset in visible_assets for parent in parents)


def test_defender_obs() -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    defender_name = next(
        agent.name
        for agent in scenario.agent_settings.values()
        if agent.type == AgentType.DEFENDER
    )
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_step_types=True
    )
    sim = MalSimulator.from_scenario(scenario)
    full_obs = create_full_obs(sim, serializer)
    defender_obs_space = MALDefenderObs(serializer)
    defender_state = sim.reset()[defender_name]
    assert isinstance(defender_state, MalSimDefenderState)
    defender_obs = full_obs2defender_obs(full_obs, defender_state, serializer)
    assert defender_obs in defender_obs_space

    while len(defender_state.action_surface) > 0:
        defender_state = sim.step(
            {defender_name: [list(defender_state.action_surface)[0]]}
        )[defender_name]
        assert isinstance(defender_state, MalSimDefenderState)
        defender_obs = full_obs2defender_obs(full_obs, defender_state, serializer)
        assert defender_obs in defender_obs_space


def test_jsonable() -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attacker = next(
        agent
        for agent in scenario.agent_settings.values()
        if agent.type == AgentType.ATTACKER
    )
    agent_name = attacker.name
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_step_types=True
    )
    obs_space = MALObs(serializer)
    sim = MalSimulator.from_scenario(scenario)

    state = sim.reset()[agent_name]
    assert isinstance(state, MalSimAttackerState)
    obs = create_full_obs(sim, serializer)
    assert obs in obs_space

    jsonable = obs_space.to_jsonable([obs])
    obs_from_jsonable = obs_space.from_jsonable(jsonable)[0]
    assert obs_from_jsonable in obs_space

    attacker_action_space = MALObsAttackStepSpace(sim)
    attacker_obs = full_obs2attacker_obs(obs, state, serializer)
    attacker_obs_idx = attacker_action_space.sample(attacker_obs.steps.action_mask)
    jsonable = attacker_action_space.to_jsonable([attacker_obs_idx])
    attacker_obs_idx_from_jsonable = attacker_action_space.from_jsonable(jsonable)[0]
    assert attacker_obs_idx_from_jsonable == attacker_obs_idx


def test_asset_then_attacker_action() -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attacker_name = next(
        agent.name
        for agent in scenario.agent_settings.values()
        if agent.type == AgentType.ATTACKER
    )
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_step_types=True
    )
    sim = MalSimulator.from_scenario(scenario)
    model = sim.sim_state.attack_graph.model
    assert model is not None, 'Attack graph needs to have a model attached to it'
    full_obs = create_full_obs(sim, serializer)

    attacker_obs_space = MALAttackerObs(serializer)
    attacker_asset_action_space = AssetThenAttackerAction(model, serializer)

    attacker_state = sim.reset()[attacker_name]
    assert isinstance(attacker_state, MalSimAttackerState)
    attacker_obs = full_obs2attacker_obs(full_obs, attacker_state, serializer)
    assert attacker_obs in attacker_obs_space

    i = 0
    while not sim.agent_is_terminated(attacker_name) and i < 100:
        asset_mask, action_mask = attacker_asset_action_space.mask(attacker_obs)
        asset_idx, action = attacker_asset_action_space.sample(
            mask=(asset_mask, action_mask)
        )
        asset = model.assets[attacker_obs.assets.id[asset_idx]]
        action_name = next(
            key
            for key, val in serializer.attacker_step_type.items()
            if val == int(action)
        )[-1]
        asset_action = f'{asset.name}:{action_name}'
        attacker_state = sim.step({attacker_name: [asset_action]})[attacker_name]
        assert isinstance(attacker_state, MalSimAttackerState)
        attacker_obs = full_obs2attacker_obs(full_obs, attacker_state, serializer)
        assert attacker_obs in attacker_obs_space
        i += 1


def test_asset_then_defender_action() -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    defender_name = next(
        agent.name
        for agent in scenario.agent_settings.values()
        if agent.type == AgentType.DEFENDER
    )
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_step_types=True
    )
    sim = MalSimulator.from_scenario(scenario)
    model = sim.sim_state.attack_graph.model
    assert model is not None, 'Attack graph needs to have a model attached to it'
    full_obs = create_full_obs(sim, serializer)

    defender_obs_space = MALDefenderObs(serializer)
    defender_action_asset_space = AssetThenDefenderAction(model, serializer)

    defender_state = sim.reset()[defender_name]
    assert isinstance(defender_state, MalSimDefenderState)
    defender_obs = full_obs2defender_obs(full_obs, defender_state, serializer)
    assert defender_obs in defender_obs_space

    i = 0
    while len(defender_state.action_surface) > 0 and i < 100:
        asset_mask, action_mask = defender_action_asset_space.mask(defender_obs)
        asset_idx, action = defender_action_asset_space.sample(
            mask=(asset_mask, action_mask)
        )
        asset = model.assets[defender_obs.assets.id[asset_idx]]
        action_name = next(
            key
            for key, val in serializer.defender_step_type.items()
            if val == int(action)
        )[-1]
        asset_action = f'{asset.name}:{action_name}'
        defender_state = sim.step({defender_name: [asset_action]})[defender_name]
        assert isinstance(defender_state, MalSimDefenderState)
        defender_obs = full_obs2defender_obs(full_obs, defender_state, serializer)
        assert defender_obs in defender_obs_space
        i += 1


def test_attacker_action_then_asset() -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    attacker_name = next(
        agent.name
        for agent in scenario.agent_settings.values()
        if agent.type == AgentType.ATTACKER
    )
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_step_types=True
    )
    sim = MalSimulator.from_scenario(scenario)
    model = sim.sim_state.attack_graph.model
    assert model is not None, 'Attack graph needs to have a model attached to it'
    full_obs = create_full_obs(sim, serializer)

    attacker_obs_space = MALAttackerObs(serializer)
    attacker_action_asset_space = AttackerActionThenAsset(model, serializer)

    attacker_state = sim.reset()[attacker_name]
    assert isinstance(attacker_state, MalSimAttackerState)
    attacker_obs = full_obs2attacker_obs(full_obs, attacker_state, serializer)
    assert attacker_obs in attacker_obs_space

    i = 0
    while not sim.agent_is_terminated(attacker_name) and i < 100:
        action_mask, asset_mask = attacker_action_asset_space.mask(attacker_obs)
        action, asset_idx = attacker_action_asset_space.sample(
            mask=(action_mask, asset_mask)
        )
        asset = model.assets[attacker_obs.assets.id[asset_idx]]
        action_name = next(
            key
            for key, val in serializer.attacker_step_type.items()
            if val == int(action)
        )[-1]
        asset_action = f'{asset.name}:{action_name}'
        attacker_state = sim.step({attacker_name: [asset_action]})[attacker_name]
        assert isinstance(attacker_state, MalSimAttackerState)
        attacker_obs = full_obs2attacker_obs(full_obs, attacker_state, serializer)
        assert attacker_obs in attacker_obs_space
        i += 1


def test_defender_action_then_asset() -> None:
    scenario_file = 'tests/testdata/scenarios/simple_scenario.yml'
    scenario = Scenario.load_from_file(scenario_file)
    defender_name = next(
        agent.name
        for agent in scenario.agent_settings.values()
        if agent.type == AgentType.DEFENDER
    )
    serializer = LangSerializer(
        scenario.lang_graph, split_assoc_types=False, split_step_types=True
    )
    sim = MalSimulator.from_scenario(scenario)
    model = sim.sim_state.attack_graph.model
    assert model is not None, 'Attack graph needs to have a model attached to it'
    full_obs = create_full_obs(sim, serializer)

    defender_obs_space = MALDefenderObs(serializer)
    defender_action_asset_space = ActionThenAsset(model, serializer)

    defender_state = sim.reset()[defender_name]
    assert isinstance(defender_state, MalSimDefenderState)
    defender_obs = full_obs2defender_obs(full_obs, defender_state, serializer)
    assert defender_obs in defender_obs_space

    i = 0
    while len(defender_state.action_surface) > 0 and i < 100:
        action_mask, asset_mask = defender_action_asset_space.mask(defender_obs)
        action, asset_idx = defender_action_asset_space.sample(
            mask=(action_mask, asset_mask)
        )
        asset = model.assets[defender_obs.assets.id[asset_idx]]
        action_name = next(
            key
            for key, val in serializer.defender_step_type.items()
            if val == int(action)
        )[-1]
        asset_action = f'{asset.name}:{action_name}'
        defender_state = sim.step({defender_name: [asset_action]})[defender_name]
        assert isinstance(defender_state, MalSimDefenderState)
        defender_obs = full_obs2defender_obs(full_obs, defender_state, serializer)
        assert defender_obs in defender_obs_space
        i += 1
