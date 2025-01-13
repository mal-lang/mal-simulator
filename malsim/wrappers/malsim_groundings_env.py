from __future__ import annotations
from typing import TYPE_CHECKING
import logging

import argparse

# mal_sim.malsim_model comes from gitr.sys.kth.se/jaknyb/RDDLGraphWrapper
# TODO: add to this repo where it fits in
from mal_sim.malsim_model import MalSimDefenderModel

from .base_classes import MalSimEnv

if TYPE_CHECKING:
    from ..sims import MalSimulator

logger = logging.getLogger(__name__)

class MalSimGroundingsEnv(MalSimEnv):

    def __init__(self, sim: MalSimulator):
        super().__init__(sim)
        self.obs_model = MalSimDefenderModel(self.sim)

        # Will be initialized by _init_step/association_grounding_values
        self._grounding_name_to_object = {}
        self.step_groundings = self._step_grounding_values()

    def _association_grounding_values(self):
        assoc_groundings = {}
        for assoc in self.sim.model.associations:
            grounding_name = \
                self.obs_model.assoc_to_grounding_name(assoc)

            self._grounding_name_to_object[grounding_name] = assoc
            assoc_groundings[grounding_name] = True
        return assoc_groundings

    def _step_grounding_values(self):
        step_groundings = {}
        for step in self.sim.attack_graph.nodes:
            grounding_name = \
                self.obs_model.attack_step_node_to_grounding_name(step)

            self._grounding_name_to_object[grounding_name] = step
            step_groundings[grounding_name] = False

            if step.is_compromised() or step.is_enabled_defense():
                step_groundings[grounding_name] = True
        return step_groundings

    def reset(self):
        super().reset()
        self.step_groundings = self._step_grounding_values()
        all_groundings = (
            self.step_groundings | self._association_grounding_values()
        )
        return all_groundings

    def step(self, steps: dict[str, dict[str, bool]]):
        """Takes steps in grounding format, runs simulator step
        and returns current state in groundings format"""

        steps_to_perform = {}
        for agent_name, groundings in steps.items():
            for grounding_name, value in groundings.items():
                if value != self.step_groundings[grounding_name]:
                    step_object = \
                        self._grounding_name_to_object[grounding_name]
                    steps_to_perform\
                        .setdefault(agent_name, []).append(step_object)

        enabled, disabled = self.sim.step(steps_to_perform)

        # Enable enabled steps in groundings dict
        for step in enabled:
            grounding_name = \
                self.obs_model.attack_step_node_to_grounding_name(step)
            self.step_groundings[grounding_name] = True

        # Optionally disable disabled steps in groundings dict
        if self.sim.sim_settings.uncompromise_untraversable_steps:
            for step in disabled:
                logger.info("Step %s disabled", step.full_name)
                grounding_name = \
                    self.obs_model.attack_step_node_to_grounding_name(step)
                self.step_groundings[grounding_name] = False

        return self.step_groundings

if __name__ == "__main__":
    from ..scenario import create_simulator_from_scenario
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('scenario_file')
    args = arg_parser.parse_args()
    sim, _ = create_simulator_from_scenario(args.scenario_file)
    env = MalSimGroundingsEnv(sim)
    groundings = env.step({})
