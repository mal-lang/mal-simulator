from os import path
import pytest
from malsim.sims.mal_simulator import MalSimulator

from maltoolbox.language import specification
from maltoolbox.language import classes_factory
from maltoolbox.attackgraph import attackgraph as malattackgraph
from maltoolbox.language import languagegraph as mallanguagegraph
from maltoolbox import model as malmodel

model_file_name='example_model.json'
attack_graph_file_name=path.join('tmp','attack_graph.json')
lang_file_name='org.mal-lang.coreLang-1.0.0.mar'


@pytest.fixture(scope="session", name="env")
def fixture_env()-> MalSimulator:

    lang_file = lang_file_name
    lang_spec = specification.load_language_specification_from_mar(lang_file)
    specification.save_language_specification_to_json(lang_spec,
        "lang_spec.json")
    lang_classes_factory = classes_factory.LanguageClassesFactory(lang_spec)
    lang_classes_factory.create_classes()

    lang_graph = mallanguagegraph.LanguageGraph(lang_spec)

    model = malmodel.Model("Test Model", lang_spec, lang_classes_factory)
    model.load_from_file(model_file_name)

    attack_graph = malattackgraph.AttackGraph(lang_spec, model)
    attack_graph.attach_attackers(model)
    attack_graph.save_to_file(attack_graph_file_name)

    env = MalSimulator(lang_graph,
        model,
        attack_graph,
        max_iter=1000)

    env.register_attacker("attacker", 0)
    env.register_defender("defender")

    return env

