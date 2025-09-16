from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, Any

from .decision_agent import DecisionAgent

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ..mal_simulator import MalSimAgentState

logger = logging.getLogger(__name__)

class KeyboardAgent(DecisionAgent):
    """An agent that makes decisions by asking user for keyboard input"""

    def __init__(self, _: Any, **kwargs: Any):
        super().__init__(**kwargs)
        logger.info("Creating KeyboardAgent")

    def get_next_action(
            self,
            agent_state: MalSimAgentState,
            **kwargs: Any
        ) -> Optional[AttackGraphNode]:
        """Compute action from action_surface"""

        def valid_action(user_input: str) -> bool:
            if user_input == "":
                return True

            try:
                node = int(user_input)
            except ValueError:
                return False

            return 0 <= node <= len(agent_state.action_surface)

        def get_action_object(user_input: str) -> Optional[int]:
            node = int(user_input) if user_input != "" else None
            return node

        if not agent_state.action_surface:
            print("No actions to pick for defender")
            return None

        index_to_node = dict(enumerate(agent_state.action_surface))
        user_input = "xxx"
        while not valid_action(user_input):
            print("Available actions:")
            print(
                "\n".join(
                    [f"{i}. {n.full_name}" for i, n in index_to_node.items()]
                )
            )
            print("Enter action or leave empty to wait:")
            user_input = input("> ")

            if not valid_action(user_input):
                print("Invalid action.")

        index = get_action_object(user_input)
        print(
            f"Selected action: {index_to_node[index].full_name}"
            if index is not None else 'wait'
        )

        return index_to_node[index] if index is not None else None
