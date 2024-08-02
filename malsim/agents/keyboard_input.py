import numpy as np
import logging

AGENT_ATTACKER = "attacker"
AGENT_DEFENDER = "defender"

logger = logging.getLogger(__name__)

null_action = (0, None)


class KeyboardAgent:
    def __init__(self, vocab):
        logger.debug("Create Keyboard agent.")
        self.vocab = vocab

    def compute_action_from_dict(self, obs: dict, mask: tuple) -> tuple:
        def valid_action(user_input: str) -> bool:
            if user_input == "":
                # Empty string interpreted as 'do nothing'
                return True

            if not user_input.isnumeric():
                return False

            action_id = int(user_input)
            if action_id < 0 or action_id >= len(action_strings):
                # action_id not within length of action_strings
                return False

            action_name = action_strings[action_id]
            action = associated_action.get(action_name, None)

            if action is None:
                # action_name not in associated_action dict
                return False

            if action == 0:
                return True  # wait is always valid

            # Return true if action_id is within available_actions length
            return 0 <= action_id < len(available_actions)

        def get_action_object(user_input: str) -> tuple:

            if user_input == "":
                action_id = None
                action = 0
            else:
                action_id = int(user_input)
                action = associated_action[action_strings[action_id]]

            return action_id, action

        available_actions = np.flatnonzero(mask[1])

        action_strings = [self.vocab[i] for i in available_actions]
        associated_action = {i: 1 for i in action_strings}
        action_strings += ["wait"]
        associated_action["wait"] = 0

        user_input = "xxx"
        while not valid_action(user_input):
            print("Available actions:")
            print("\n".join([f"{i}. {a}" for i, a in enumerate(action_strings)]))
            print("Enter action or leave empty to wait:")
            user_input = input("> ")

            if not valid_action(user_input):
                print("Invalid action.")

        node, a = get_action_object(user_input)
        print(
            f"Selected action: {action_strings[node] if node is not None else 'wait'}"
        )

        return (a, available_actions[node] if a != 0 else -1)
