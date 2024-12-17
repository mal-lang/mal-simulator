class MalSimAgent():
    pass

class MalSimAttackerAgent():
    pass

class MalSimDefenderAgent():
    pass


class PassiveAttacker():
    def __init__(self, info):
        return

    def get_next_actions(self, observation, mask) -> list[tuple[int, int]]:
        return []

class PassiveDefender():
    def __init__(self, info):
        return

    def get_next_actions(self, observation, mask) -> list[tuple[int, int]]:
        return []
