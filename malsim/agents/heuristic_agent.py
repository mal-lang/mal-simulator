from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import logging
import math

import numpy as np

from .decision_agent import DecisionAgent

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ..mal_simulator import MalSimAgentStateView

logger = logging.getLogger(__name__)

class DefendCompromisedDefender(DecisionAgent):
    """A defender that defends compromised assets using notPresent"""

    def __init__(self, agent_config, **_):
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

    def get_next_action(
        self, agent_state: MalSimAgentStateView, **kwargs
    ) -> Optional[AttackGraphNode]:

        """Return an action that disables a compromised node"""

        selected_node_cost = math.inf
        selected_node = None

        # To make it deterministic
        possible_choices = list(agent_state.action_surface)
        possible_choices.sort(key=lambda n: n.id)

        for node in possible_choices:

            if node.is_enabled_defense():
                continue

            node_cost = node.extras.get('reward', 0)

            # Strategy:
            # - Enabled the cheapest defense node
            #   that has compromised child nodes
            if node_cost < selected_node_cost:

                node_has_compromised_child = (
                    any(
                        child_node.is_compromised()
                        for child_node in node.children
                    )
                )

                if node_has_compromised_child:
                    selected_node = node
                    selected_node_cost = node_cost

        return selected_node


class DefendFutureCompromisedDefender(DecisionAgent):
    """A defender that defends compromised assets using notPresent"""

    def __init__(self, agent_config, **_):
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

    def get_next_action(
        self, agent_state: MalSimAgentStateView, **kwargs
    ) -> Optional[AttackGraphNode]:

        """Return an action that disables a compromised node"""

        selected_node_cost = math.inf
        selected_node = None

        # To make it deterministic
        possible_choices = list(agent_state.action_surface)
        possible_choices.sort(key=lambda n: n.id)

        for node in possible_choices:

            if node.is_enabled_defense():
                continue

            node_cost = node.extras.get('reward', 0)

            # Strategy:
            # - Enabled the cheapest defense node
            #   that has a non compromised child
            #   that has a compromised parent.
            if node_cost < selected_node_cost:

                node_has_child_that_can_be_compromised = (
                    any(
                        any(p.is_compromised() for p in child_node.parents)
                        and not child_node.is_compromised()
                        for child_node in node.children
                    )
                )

                if node_has_child_that_can_be_compromised:
                    selected_node = node
                    selected_node_cost = node_cost

        return selected_node
