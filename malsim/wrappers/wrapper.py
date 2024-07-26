from typing import Any

from gymnasium.spaces.space import Space
from pettingzoo.utils.env import ParallelEnv

from maltoolbox.language import LanguageClassesFactory, LanguageGraph
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.model import Model

from malsim.sims.mal_simulator import MalSimulator


class LazyWrapper(ParallelEnv):
    def __init__(self, **kwargs):
        lang_file = kwargs.pop("lang_file")
        model_file = kwargs.pop("model_file")
        if "attack_graph_file" in kwargs:
            attack_graph_file = kwargs.pop("attack_graph_file")
        else:
            attack_graph_file = ""
        agents = kwargs.pop("agents", {})
        lang_graph = LanguageGraph.from_mar_archive(
            str(lang_file))
        lang_classes_factory = LanguageClassesFactory(lang_graph)

        model = Model.load_from_file(model_file, lang_classes_factory)

        if attack_graph_file != "":
            # If we were provided with an attack graph file we load it.
            attack_graph = AttackGraph.load_from_file(attack_graph_file, model=model)
            attack_graph.model = model
            attack_graph.attach_attackers()
        else:
            # Otherwise we generate the attack graph based on the model
            # provided.
            attack_graph = AttackGraph(lang_graph, model)
            attack_graph.attach_attackers()

        sim = MalSimulator(lang_graph,
            model,
            attack_graph,
            **kwargs)

        for agent_class, agent_id in agents.items():
            if agent_class == "attacker":
                sim.register_attacker(agent_id, 0)
            elif agent_class == "defender":
                sim.register_defender(agent_id)

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
        return result


    def observation_space(self, agent: Any) -> Space:
        return self.sim.observation_space(agent)


    def action_space(self, agent: Any) -> Space:
        return self.sim.action_space(agent)
