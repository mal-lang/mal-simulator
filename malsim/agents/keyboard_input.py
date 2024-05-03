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
                return True

            try:
                node = int(user_input)
            except ValueError:
                return False

            try:
                a = associated_action[action_strings[node]]
            except IndexError:
                return False

            if a == 0:
                return True  # wait is always valid
            return node < len(available_actions) and node >= 0

        def get_action_object(user_input: str) -> tuple:
            node = int(user_input) if user_input != "" else None
            action = associated_action[action_strings[node]] if user_input != "" else 0
            return node, action

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
