import logging

from maltoolbox.language import classes_factory
from maltoolbox.language import specification
from maltoolbox.language import languagegraph as mallanguagegraph
from maltoolbox.attackgraph import attackgraph as malattackgraph
from maltoolbox.model import model as malmodel

from pettingzoo.test import parallel_api_test

from malpzsim.sims.mal_petting_zoo_simulator import MalPettingZooSimulator

logger = logging.getLogger(__name__)

null_action = (0, None)

lang_file = "org.mal-lang.coreLang-1.0.0.mar"
lang_spec = specification.load_language_specification_from_mar(lang_file)
specification.save_language_specification_to_json(lang_spec, "lang_spec.json")
lang_classes_factory = classes_factory.LanguageClassesFactory(lang_spec)
lang_classes_factory.create_classes()

lang_graph = mallanguagegraph.LanguageGraph()
lang_graph.generate_graph(lang_spec)

model = malmodel.Model("Test Model", lang_spec, lang_classes_factory)
model.load_from_file("example_model.json")

attack_graph = malattackgraph.AttackGraph()
attack_graph.generate_graph(lang_spec, model)
attack_graph.attach_attackers(model)
attack_graph.save_to_file("tmp/attack_graph.json")

env = MalPettingZooSimulator(lang_graph, model, attack_graph, max_iter=5)

env.register_attacker("attacker", 0)
env.register_defender("defender")

logger.debug("Run Parrallel API test.")
parallel_api_test(env, num_cycles=50)

env.close()
