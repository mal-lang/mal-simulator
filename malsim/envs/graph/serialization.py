from typing import Any
from maltoolbox.attackgraph import AttackGraphNode
from maltoolbox.language import LanguageGraph, LanguageGraphAttackStep
import numpy as np
class LangSerializer:
    """Serializer for LanguageGraph"""

    def __init__(
        self,
        lang: LanguageGraph,
        split_assoc_types: bool = False,
        split_step_types: bool = False
    ):
        """
        Arguments:
        ----------
        - lang: LanguageGraph to serialize
        - split_assoc_types: If True, split association types for each asset type pair
        - split_step_types: If True, split attack step types for each asset type
        """
        self._lang = lang
        self.split_assoc_types = split_assoc_types
        self.split_step_types = split_step_types

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

        all_steps = sorted([
            step for asset in self._lang.assets.values()
            for step in asset.attack_steps.values()
        ], key=lambda x: x.name)
        all_steps_attacker_sorting = list(filter(lambda x: x.type in ("and", "or"), all_steps)) + list(filter(lambda x: x.type in ("defense", "exist", "notExist"), all_steps))
        all_steps_defender_sorting = list(filter(lambda x: x.type == "defense", all_steps)) + list(filter(lambda x: x.type in ("and", "or", "exist", "notExist"), all_steps))

        # NOTE: This looks odd, but the step name is the type of the action
        self.step_type: dict[tuple[str, ...], int] = {}
        # Different indexing for attacker and defender
        # This is needed by some action spaces
        self.attacker_step_type: dict[tuple[str, ...], int] = {}
        self.defender_step_type: dict[tuple[str, ...], int] = {}
        if split_step_types:
            # Map from (asset_name, step_name) to idx
            for idx, step in enumerate(all_steps):
                step_key = (step.asset.name, step.name)
                if step_key in self.step_type:
                    raise ValueError(f"Can not have more than one key {step_key}")
                self.step_type[step_key] = idx
            for idx, step in enumerate(all_steps_attacker_sorting):
                step_key = (step.asset.name, step.name)
                if step_key in self.attacker_step_type:
                    raise ValueError(f"Can not have more than one key {step_key}")
                self.attacker_step_type[step_key] = idx
            for idx, step in enumerate(all_steps_defender_sorting):
                step_key = (step.asset.name, step.name)
                if step_key in self.defender_step_type:
                    raise ValueError(f"Can not have more than one key {step_key}")
                self.defender_step_type[step_key] = idx
        else:
            # Map from (step_name) to idx
            all_step_names: set[str] = set(
                step.name for step in all_steps
            )
            self.step_type = {
                (step_name,): i
                for i, step_name in enumerate(all_step_names)
            }
            
            attacker_seen_step_names: set[str] = set()
            idx = 0
            for step in all_steps_attacker_sorting:
                if step.name not in attacker_seen_step_names:
                    attacker_seen_step_names.add(step.name)
                    self.attacker_step_type[(step.name,)] = idx
                    idx += 1

            defender_seen_step_names: set[str] = set()
            idx = 0
            for step in all_steps_defender_sorting:
                if step.name not in defender_seen_step_names:
                    defender_seen_step_names.add(step.name)
                    self.defender_step_type[(step.name,)] = idx
                    idx += 1

        self.step_type2attacker_step_type = np.zeros((len(self.step_type),), dtype=np.int64)
        for step_key, idx in self.step_type.items():
            self.step_type2attacker_step_type[idx] = self.attacker_step_type[step_key]

        self.step_type2defender_step_type = np.zeros((len(self.step_type),), dtype=np.int64)
        for step_key, idx in self.step_type.items():
            self.step_type2defender_step_type[idx] = self.defender_step_type[step_key]

        # NOTE: The actual logic-class of the step
        all_step_classes = set(
            step.type for step in all_steps
        )
        # Map from step class name to idx
        self.step_class = {
            class_name: i for i, class_name in enumerate(all_step_classes)
        }

        # NOTE: Add None tag for steps without tags
        all_step_tags: list[Any] = [None] + list(
            sorted(
                set(tag for step in all_steps for tag in step.tags)
            )
        )
        self.step_tag: dict[str | None, int] = {
            tag: i for i, tag in enumerate(all_step_tags)
        }

