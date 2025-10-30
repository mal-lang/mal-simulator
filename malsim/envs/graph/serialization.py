from typing import Any
from maltoolbox.language import LanguageGraph

class LangSerializer:
    """Serializer for LanguageGraph"""

    def __init__(
        self,
        lang: LanguageGraph,
        split_assoc_types: bool = False,
        split_attack_step_types: bool = False
    ):
        """
        Arguments:
        ----------
        - lang: LanguageGraph to serialize
        - split_assoc_types: If True, split association types for each asset type pair
        - split_attack_step_types: If True, split attack step types for each asset type
        """
        self._lang = lang
        self.split_assoc_types = split_assoc_types
        self.split_attack_step_types = split_attack_step_types

        # Maps asset_type_name to index
        # TODO: Remove abstract classes?
        self.asset_type: dict[str, int] = {
            asset_type: i for i, asset_type
            in enumerate(sorted(self._lang.assets.keys()))
        }

        self.association_type: dict[tuple[str, ...], int] = {}
        if split_assoc_types:
            # Map from (assoc_name, left_asset_type, right_asset_type) to idx
            type_idx = 0
            for association in sorted(self._lang.associations, key=lambda x: x.name):
                assoc_name = association.name
                left_asset_type = association.left_field.asset.name
                right_asset_type = association.right_field.asset.name
                assoc_type_key = (assoc_name, left_asset_type, right_asset_type)
                if assoc_type_key in self.association_type:
                    raise ValueError(f"Can not have more than one key {assoc_type_key}")
                self.association_type[assoc_type_key] = type_idx
        else:
            # Map from (assoc_name,) to idx
            self.association_type = {
                (association.name,): i
                for i, association in enumerate(
                    sorted(self._lang.associations, key=lambda x: x.name)
                )
            }

        # TODO: Decide if we want to serialize
        # LanguageGraphAssociationField attributes minimum and maximum

        all_attack_steps = sorted([
            attack_step for asset in self._lang.assets.values()
            for attack_step in asset.attack_steps.values()
        ], key=lambda x: x.name)

        # NOTE: This looks odd, but the attack step name is the type of the action
        self.attack_step_type: dict[tuple[str, ...], int] = {}
        if split_attack_step_types:
            # Map from (asset_name, attack_step_name) to idx
            for idx, attack_step in enumerate(all_attack_steps):
                attack_step_key = (attack_step.asset.name, attack_step.name)
                if attack_step_key in self.attack_step_type:
                    raise ValueError(f"Can not have more than one key {attack_step_key}")
                self.attack_step_type[attack_step_key] = idx
        else:
            # Map from (attack_step_name) to idx
            all_attack_step_names: set[str] = set(
                attack_step.name for attack_step in all_attack_steps
            )
            self.attack_step_type = {
                (attack_step_name,): i
                for i, attack_step_name in enumerate(all_attack_step_names)
            }

        # NOTE: The actual logic-class of the attack step
        all_attack_step_classes = set(
            attack_step.type for attack_step in all_attack_steps
        )
        # Map from attack step class name to idx
        self.attack_step_class = {
            class_name: i for i, class_name in enumerate(all_attack_step_classes)
        }

        # NOTE: Add None tag for steps without tags
        all_attack_step_tags: list[Any] = [None] + list(
            sorted(
                set(tag for attack_step in all_attack_steps for tag in attack_step.tags)
            )
        )
        self.attack_step_tag: dict[str | None, int] = {
            tag: i for i, tag in enumerate(all_attack_step_tags)
        }
