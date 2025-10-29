from maltoolbox.language import LanguageGraphAssociation
from maltoolbox.attackgraph import AttackGraphNode
from malsim.envs.serialization import LangSerializer
from malsim.envs.mal_spaces import (
    MALObs,
    MALObsInstance,
    AttackStep,
    Asset,
    Association,
    LogicGate,
    attacker_state2graph,
    AttackGraphNodeSpace,
)
from malsim.scenario import Scenario, AgentType
from malsim.mal_simulator import MalSimulator, MalSimAttackerState
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

    def state2instance(sim: MalSimulator) -> MALObsInstance:
        def get_total_attempts(node: AttackGraphNode) -> int:
            return sum(state.num_attempts.get(node, 0) 
                for state in sim._get_attacker_agents()
            )
        def is_traversable_by_any(node: AttackGraphNode) -> bool:
            return any(sim.node_is_traversable(state.performed_nodes, node) 
                for state in sim._get_attacker_agents()
            )
        sorted_attack_steps = [
            sim.attack_graph.nodes[node_id]
            for node_id in sorted(sim.attack_graph.nodes.keys())
        ]
        attack_steps = AttackStep(
            type=np.array([
                serializer.attack_step_type[(node.model_asset.type, node.name)]
                for node in sorted_attack_steps if node.model_asset
            ]),
            id=np.array([node.id for node in sorted_attack_steps]),
            logic_class=np.array([
                serializer.attack_step_class[node.type]
                for node in sorted_attack_steps
            ]),
            tags=np.array([
                serializer.attack_step_tag[node.tags[0] if len(node.tags) > 0 else None]
                for node in sorted_attack_steps
            ]),
            compromised=np.array([
                sim.node_is_compromised(node) for node in sorted_attack_steps
            ]),
            attempts=np.array([
                get_total_attempts(node) for node in sorted_attack_steps
            ]),
            traversable=np.array([
                is_traversable_by_any(node) for node in sorted_attack_steps
            ]),
        )
        step2step = {
            (sorted_attack_steps.index(node), sorted_attack_steps.index(child))
            for node in sorted_attack_steps for child in node.children if child in sorted_attack_steps
        }
        
        assert sim.attack_graph.model, "Attack graph needs to have a model attached to it"
        sorted_assets = [
            sim.attack_graph.model.assets[asset_id]
            for asset_id in sorted(sim.attack_graph.model.assets.keys())
        ]
        assets = Asset(
            type=np.array([serializer.asset_type[asset.type] for asset in sorted_assets]),
            id=np.array([asset.id for asset in sorted_assets]),
        )
        step2asset = {(
            sorted_attack_steps.index(node), sorted_assets.index(node.model_asset))
            for node in sorted_attack_steps if node.model_asset in sorted_assets
        }

        associations: list[tuple[LanguageGraphAssociation, int, int]] = []
        for asset in sorted_assets:
            for fieldname, other_assets in asset.associated_assets.items():
                assoc = asset.lg_asset.associations[fieldname]
                for other_asset in other_assets:
                    if (
                        (assoc, asset.id, other_asset.id) not in associations 
                        and (assoc, other_asset.id, asset.id) not in associations
                    ):
                        associations.append((assoc, asset.id, other_asset.id))
        sorted_associations = sorted(associations, key=lambda x: (x[0].name, x[1], x[2]))
        association = Association(
            type=np.array([
                serializer.association_type[assoc.name,]
                for (assoc, _, _) in sorted_associations
            ])
        )
        assoc2asset: set[tuple[int, int]] = set()
        for i, (assoc, asset1_id, asset2_id) in enumerate(sorted_associations):
            assoc2asset.add((i, asset1_id))
            assoc2asset.add((i, asset2_id))

        sorted_and_or_steps = sorted(
            [node for node in sorted_attack_steps if node.type in ('and', 'or')],
            key=lambda x: x.id
        )
        logic_gates = LogicGate(
            type=np.array([(0 if node.type == 'and' else 1) for node in sorted_and_or_steps])
        )
        step2logic = set()
        logic2step = set()
        for logic_gate_id, node in enumerate(sorted_and_or_steps):
            logic2step.add((logic_gate_id, node.id))
            for child in node.children:
                step2logic.add((child.id, logic_gate_id))
            

        return MALObsInstance(
            time=np.int64(sim.cur_iter),
            assets=assets,
            attack_steps=attack_steps,
            associations=association,
            logic_gates=logic_gates,
            step2asset=np.array(list(zip(*step2asset))),
            step2step=np.array(list(zip(*step2step))),
            assoc2asset=np.array(list(zip(*assoc2asset))),
            logic2step=np.array(list(zip(*logic2step))),
            step2logic=np.array(list(zip(*step2logic))),
        )

    states = sim.reset()
    assert state2instance(sim) in obs_space
    for _ in range(10):
        actions = {}
        for agent_name, state in states.items():
            if len(state.action_surface) > 0:
                actions[agent_name] = [list(state.action_surface)[0]]
            else:
                actions[agent_name] = []
        states = sim.step(actions)
        assert state2instance(sim) in obs_space

def test_obs_creation() -> None:
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

    while not sim.agent_is_terminated(agent_name):
        actions = {agent_name: [next(iter(state.action_surface))]}
        state = sim.step(actions)[agent_name]
        assert isinstance(state, MalSimAttackerState)
        assert attacker_state2graph(state, serializer, use_logic_gates=False) in obs_space

def test_node_space() -> None:
    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    attacker = next(agent for agent in scenario.agents if agent['type'] == AgentType.ATTACKER)
    agent_name = attacker['name']
    sim = MalSimulator.from_scenario(scenario)
    action_space = AttackGraphNodeSpace(sim.attack_graph)

    state = sim.reset()[agent_name]

    while not sim.agent_is_terminated(agent_name):
        action = next(iter(state.action_surface))
        assert action.id in action_space
        actions = {agent_name: [action]}
        state = sim.step(actions)[agent_name]

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