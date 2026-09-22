import contextlib
from pathlib import Path
import random

from maltoolbox.language import LanguageGraph
from maltoolbox.model import Model, ModelAsset
from maltoolbox.attackgraph import AttackGraph
import yaml
from malsim.config.agent_settings import AttackerSettings
from malsim.policies.random_agent import RandomAgent
from malsim.scenario import Scenario


if __name__ == '__main__':
    test_data_dir = Path('tests/testdata/')

    lang_file = test_data_dir / 'langs' / 'dynamic_remove_many_assoc.mal'
    lang_graph = LanguageGraph.load_from_file(str(lang_file))

    model = Model('Dynamic Remove Many Assoc', lang_graph)
    god = model.add_asset(asset_type='GodAsset', name='God')

    object_type = lang_graph.assets['Object']
    num_objects = random.randint(20, 40)
    objects = []
    for i in range(num_objects):
        new_object: ModelAsset = model.add_asset(asset_type='Object', name=f'Object{i}')

        with contextlib.suppress(IndexError):
            new_object.add_associated_assets('parent', {objects[-1]})

        new_object.add_associated_assets('metas', set(objects))
        objects.append(new_object)

        a = model.add_asset(asset_type='A', name=f'A{i}')
        b = model.add_asset(asset_type='B', name=f'B{i}')
        a.add_associated_assets('b', {b})
        c = model.add_asset(asset_type='C', name=f'C{i}')
        b.add_associated_assets('c', {c})
        d = model.add_asset(asset_type='D', name=f'D{i}')
        c.add_associated_assets('d', {d})

        god.add_associated_assets('a', {a})
        god.add_associated_assets('b', {b})
        god.add_associated_assets('c', {c})
        god.add_associated_assets('d', {d})
    god.add_associated_assets('objects', set(objects))

    attack_graph = AttackGraph(lang_graph, model)

    attacker_settings = AttackerSettings(
        name='Attacker',
        entry_points={attack_graph.get_node_by_full_name('Object0:reach')},
        policy=RandomAgent,
    )
    scenario = Scenario(
        lang_file=str(lang_file),
        model=model,
        agents=[attacker_settings],
    )
    scenario_dir = test_data_dir / 'scenarios'
    scenario_file = scenario_dir / 'dynamic_remove_many_assoc_scenario.yml'
    scenario.save_to_file(scenario_file)

    scenario_dict = yaml.safe_load(scenario_file.read_text())
    scenario_dict['lang_file'] = scenario_dict['lang_file'].replace(
        'tests/testdata', '..'
    )
    scenario_dict['agents']['Attacker']['entry_points'] = ['Object0:reach']
    scenario_file.write_text(yaml.dump(scenario_dict, sort_keys=False))
