from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
import logging
import math

import numpy as np

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ..mal_simulator import MalSimDefenderState

logger = logging.getLogger(__name__)

class DefendCompromisedDefender:
    """A defender that defends compromised assets using notPresent"""

    def __init__(self, agent_config: dict[str, Any], **_: Any):
        # Seed and rng not currently used
        seed = (
            agent_config["seed"]
            if agent_config.get("seed")
            else np.random.SeedSequence().entropy
        )
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get("randomize")
            else None
        )
        self.compromised_nodes: set[AttackGraphNode] = set()

    def get_next_action(
        self, agent_state: MalSimDefenderState, **kwargs: Any
    ) -> Optional[AttackGraphNode]:

        """Return an action that disables a compromised node"""

        self.compromised_nodes |= agent_state.step_all_compromised_nodes

        selected_node_cost = math.inf
        selected_node = None

        # To make it deterministic
        possible_choices = list(agent_state.action_surface)
        possible_choices.sort(key=lambda n: n.id)

        for node in possible_choices:

            if node in self.compromised_nodes:
                continue

            node_cost = node.extras.get('reward', 0)

            # Strategy:
            # - Enabled the cheapest defense node
            #   that has compromised child nodes
            if node_cost < selected_node_cost:

                node_has_compromised_child = (
                    any(
                        child_node in self.compromised_nodes
                        for child_node in node.children
                    )
                )

                if node_has_compromised_child:
                    selected_node = node
                    selected_node_cost = node_cost

        return selected_node


class DefendFutureCompromisedDefender:
    """A defender that defends compromised assets using notPresent"""

    def __init__(self, agent_config: dict[str, Any], **_: Any):
        # Seed and rng not currently used
        seed = (
            agent_config["seed"]
            if agent_config.get("seed")
            else np.random.SeedSequence().entropy
        )
        self.rng = (
            np.random.default_rng(seed)
            if agent_config.get("randomize")
            else None
        )
        self.compromised_nodes: set[AttackGraphNode] = set()

    def get_next_action(
        self, agent_state: MalSimDefenderState, **kwargs: Any
    ) -> Optional[AttackGraphNode]:

        """Return an action that disables a compromised node"""

        self.compromised_nodes |= agent_state.step_all_compromised_nodes

        selected_node_cost = math.inf
        selected_node = None

        # To make it deterministic
        possible_choices = list(agent_state.action_surface)
        possible_choices.sort(key=lambda n: n.id)

        for node in possible_choices:

            if node in self.compromised_nodes:
                continue

            node_cost = node.extras.get('reward', 0)

            # Strategy:
            # - Enabled the cheapest defense node
            #   that has a non compromised child
            #   that has a compromised parent.
            if node_cost < selected_node_cost:

                node_has_child_that_can_be_compromised = (
                    any(
                        any(
                            p in self.compromised_nodes
                            for p in child_node.parents
                        )
                        and child_node not in self.compromised_nodes
                        for child_node in node.children
                    )
                )

                if node_has_child_that_can_be_compromised:
                    selected_node = node
                    selected_node_cost = node_cost

        return selected_node
