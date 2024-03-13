from gymnasium.spaces.space import Space
from pettingzoo.utils.env import ParallelEnv

from maltoolbox.language import classes_factory
from maltoolbox.language import specification
from maltoolbox.language import languagegraph as mallanguagegraph
from maltoolbox.attackgraph import attackgraph as malattackgraph
from maltoolbox.model import model as malmodel

from malpzsim.sims.mal_petting_zoo_simulator import MalPettingZooSimulator
from typing import Any


class LazyWrapper(ParallelEnv):
    def __init__(self, **kwargs):
        lang_file = kwargs.pop("lang_file")
        model_file = kwargs.pop("model_file")
        agents = kwargs.pop("agents", {})
        lang_spec = specification.load_language_specification_from_mar(str(lang_file))
        lang_classes_factory = classes_factory.LanguageClassesFactory(lang_spec)
        lang_classes_factory.create_classes()

        lang_graph = mallanguagegraph.LanguageGraph()
        lang_graph.generate_graph(lang_spec)

        model = malmodel.Model("Test Model", lang_spec, lang_classes_factory)
        model.load_from_file(model_file)

        attack_graph = malattackgraph.AttackGraph()
        attack_graph.generate_graph(lang_spec, model)
        attack_graph.attach_attackers(model)

        sim = MalPettingZooSimulator(lang_graph, model, attack_graph, **kwargs)

        for agent_class, agent_id in agents.items():
            if agent_class == "attacker":
                sim.register_attacker(agent_id, 0)
            elif agent_class == "defender":
                sim.register_defender(agent_id)

        # TODO - This is a temporary fix to set the rewards for the nodes in the attack graph



        self.sim = sim

    def step(
        self, actions: dict
    ) -> tuple[
        dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]
    ]:
        return self.sim.step(actions)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        result = self.sim.reset(seed, options)
        sim = self.sim
        sim.attack_graph.get_node_by_id("Application:0:notPresent").reward = 2
        sim.attack_graph.get_node_by_id("Application:0:supplyChainAuditing").reward = 7
        sim.attack_graph.get_node_by_id("Application:1:notPresent").reward = 3
        sim.attack_graph.get_node_by_id("Application:1:supplyChainAuditing").reward = 7
        sim.attack_graph.get_node_by_id("SoftwareVulnerability:2:notPresent").reward = 4
        sim.attack_graph.get_node_by_id("Data:3:notPresent").reward = 1
        sim.attack_graph.get_node_by_id("Credentials:4:notPhishable").reward = 7
        sim.attack_graph.get_node_by_id("Identity:5:notPresent").reward = 3.5
        sim.attack_graph.get_node_by_id("ConnectionRule:6:restricted").reward = 4
        sim.attack_graph.get_node_by_id("ConnectionRule:6:payloadInspection").reward = 3
        sim.attack_graph.get_node_by_id("Application:7:notPresent").reward = 2
        sim.attack_graph.get_node_by_id("Application:7:supplyChainAuditing").reward = 7

        sim.attack_graph.get_node_by_id("Application:0:fullAccess").reward = 5
        sim.attack_graph.get_node_by_id("Application:1:fullAccess").reward = 2
        sim.attack_graph.get_node_by_id("Identity:5:assume").reward = 2
        sim.attack_graph.get_node_by_id("Application:7:fullAccess").reward = 6
        return result
    
    def observation_space(self, agent: Any) -> Space:
        return self.sim.observation_space(agent)

    def action_space(self, agent: Any) -> Space:
        return self.sim.action_space(agent)
