from maltoolbox.language import LanguageGraphAssociation
from maltoolbox.attackgraph import AttackGraphNode
from malsim.envs.serialization import LangSerializer
from malsim.envs.mal_spaces import MALObs, MALObsInstance, AttackStep, Assets, Association, LogicGate
from malsim.scenario import Scenario
from malsim.mal_simulator import MalSimulator
import numpy as np

def test_mal_obs() -> None:

    scenario_file = (
        "tests/testdata/scenarios/simple_scenario.yml"
    )
    scenario = Scenario.load_from_file(scenario_file)
    serializer = LangSerializer(scenario.lang_graph, split_assoc_types=False, split_attack_step_types=True)
    obs_space = MALObs(serializer)
    sim = MalSimulator(scenario.attack_graph)

    def state2instance(sim: MalSimulator) -> MALObsInstance:
        def get_total_attempts(node: AttackGraphNode) -> int:
            return sum(sim.agent_states[agent].num_attempts[node] 
                for agent in sim.agent_states.keys() 
                if hasattr(sim.agent_states[agent], 'num_attempts')
            )
        def is_traversable_by_any(node: AttackGraphNode) -> bool:
            return any(sim.node_is_traversable(sim.agent_states[agent].performed_nodes, node) 
                for agent in sim.agent_states.keys() 
                if hasattr(sim.agent_states[agent], 'performed_nodes')
            )
        sorted_attack_steps = [sim.attack_graph.nodes[node_id] for node_id in sorted(sim.attack_graph.nodes.keys())]
        attack_steps = AttackStep(
            type=np.array([serializer.attack_step_type[node.model_asset.type][node.name] for node in sorted_attack_steps]),
            logic_class=np.array([serializer.attack_step_class[node.type] for node in sorted_attack_steps]),
            tags=np.array([serializer.attack_step_tag[node.tags[0] if len(node.tags) > 0 else None] for node in sorted_attack_steps]),
            compromised=np.array([sim.node_is_compromised(node) for node in sorted_attack_steps]),
            attempts=np.array([get_total_attempts(node) for node in sorted_attack_steps]),
            traversable=np.array([is_traversable_by_any(node) for node in sorted_attack_steps]),
        )
        step2step = {(node.id, child.id) for node in sorted_attack_steps for child in node.children}
        
        sorted_assets = [sim.attack_graph.model.assets[asset_id] for asset_id in sorted(sim.attack_graph.model.assets.keys())]
        assets = Assets(type=np.array([serializer.asset_type[asset.type] for asset in sorted_assets]))
        step2asset = {(node.id, node.model_asset.id) for node in sorted_attack_steps}

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
        associations = Association(
            type=np.array([serializer.association_type[assoc.name] for (assoc, _, _) in sorted_associations])
        )
        assoc2asset: set[tuple[int, int]] = set()
        for i, (assoc, asset1_id, asset2_id) in enumerate(sorted_associations):
            assoc2asset.add((i, asset1_id))
            assoc2asset.add((i, asset2_id))

        sorted_and_or_steps = sorted([node for node in sorted_attack_steps if node.type in ('and', 'or')], key=lambda x: x.id)
        logic_gates = LogicGate(type=np.array([(0 if node.type == 'and' else 1) for node in sorted_and_or_steps]))
        step2logic = set()
        logic2step = set()
        for logic_gate_id, node in enumerate(sorted_and_or_steps):
            logic2step.add((logic_gate_id, node.id))
            for child in node.children:
                step2logic.add((child.id, logic_gate_id))
            

        return MALObsInstance(
            time=sim.cur_iter,
            assets=assets,
            attack_steps=attack_steps,
            associations=associations,
            logic_gates=logic_gates,
            step2asset=np.array(list(zip(*step2asset))),
            step2step=np.array(list(zip(*step2step))),
            assoc2asset=np.array(list(zip(*assoc2asset))),
            logic2step=np.array(list(zip(*logic2step))),
            step2logic=np.array(list(zip(*step2logic))),
        )

    states = sim.reset()
    assert obs_space.contains(state2instance(sim))
    for _ in range(10):
        actions = {}
        for agent_name, state in states.items():
            actions[agent_name] = [state.action_surface[0]]
        states = sim.step(actions)
        assert obs_space.contains(state2instance(sim))
