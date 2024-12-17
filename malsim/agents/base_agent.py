class MalSimAgent():
    pass

class MalSimAttackerAgent():
    pass

class MalSimDefenderAgent():
    pass

class PassiveAgent():
    def __init__(self, info):
        return

    def get_next_actions(self, state) -> list[tuple[int, int]]:
        return []
