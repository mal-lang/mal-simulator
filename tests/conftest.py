from os import path
import pytest
from malsim.sims.mal_simulator import MalSimulator

from maltoolbox.wrappers import create_attack_graph

model_file_name='tests/example_model.yml'
attack_graph_file_name=path.join('tmp','attack_graph.json')
lang_file_name='tests/org.mal-lang.coreLang-1.0.0.mar'


@pytest.fixture(scope="session", name="env")
def fixture_env()-> MalSimulator:

    attack_graph = create_attack_graph(lang_file_name, model_file_name)
    lang_graph = attack_graph.lang_graph

    model = attack_graph.model

    attack_graph.save_to_file(attack_graph_file_name)

    env = MalSimulator(lang_graph,
        model,
        attack_graph,
        max_iter=1000)

    env.register_attacker("attacker", 0)
    env.register_defender("defender")

    return env

