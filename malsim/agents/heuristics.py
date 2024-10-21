import logging
from typing import Optional

from .agent_base import MalSimulatorAgent

logger = logging.getLogger(__name__)

class TripWireDefender(MalSimulatorAgent):
    """A defender that defends compromised assets using notPresent"""

    def __init__(self, agent_config: dict, **kwargs):
        """Agent needs the simulator to map between observation and graph"""
        self.sim = kwargs.get('simulator')

    def compute_action_from_dict(self, _, __) -> tuple[int, Optional[int]]:
        """Return an action which blocks a reached attack step"""

        # Assume that there is just one attacker registered
        attacker = self.sim.attack_graph.attackers[0]

        for attack_step in attacker.reached_attack_steps:
            asset = attack_step.asset
            defense_step_name = f'{asset.name}:notPresent'
            defense_step_node = self.sim.attack_graph.get_node_by_full_name(
                defense_step_name
            )

            if defense_step_node and defense_step_node.is_available_defense():
                # Return a defense that disables compromised asset
                return (1, self.sim._id_to_index[defense_step_node.id])

        return (0, None)
