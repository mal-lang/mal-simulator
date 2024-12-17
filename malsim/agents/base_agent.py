class MalSimAgent():
    pass

class MalSimAttackerAgent():
    pass

class MalSimDefenderAgent():
    pass

class PassiveAgent():
    def __init__(self, info):
        return

    def get_next_action(self, state) -> tuple[int, int]:
        return (0, None)
