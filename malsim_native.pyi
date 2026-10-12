class AttackGraphIndex:
    def __init__(
        self,
        ids: list[int],
        kinds: list[int],
        existence_status: list[bool],
        necessary: list[bool],
        impossible: list[bool],
        parents: list[list[int]],
    ) -> None: ...
    def filter_traversable(
        self,
        candidate_ids: list[int],
        performed_ids: list[int],
        enabled_defense_ids: list[int],
    ) -> list[int]: ...
    def is_traversable_single(
        self,
        node_id: int,
        performed_ids: list[int],
        enabled_defense_ids: list[int],
    ) -> bool: ...
