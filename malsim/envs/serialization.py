from maltoolbox.language import LanguageGraph

class LangSerializer:
    """Serializer for LanguageGraph"""

    def __init__(self, lang: LanguageGraph, split_assoc_types: bool = False, split_attack_step_types: bool = False):
        """
        Args:
            lang: LanguageGraph to serialize
            split_assoc_types: If True, split association types into separate types for each asset type
        """
        self._lang = lang

        # TODO: Remove abstract classes?
        self.asset_type = {
            asset_type: i for i, asset_type in enumerate(sorted(self._lang.assets.keys()))
        }

        if split_assoc_types:
            self.association_type = {}
            type_idx = 0
            for association in sorted(self._lang.associations, key=lambda x: x.name):
                left_asset_type = association.left_field.asset.name
                right_asset_type = association.right_field.asset.name
                if association.name not in self.association_type:
                    self.association_type[association.name] = {}
                if left_asset_type not in self.association_type[association.name]:
                    self.association_type[association.name][left_asset_type] = {}
                if right_asset_type not in self.association_type[association.name][left_asset_type]:
                    self.association_type[association.name][left_asset_type][right_asset_type] = type_idx
                type_idx += 1

        else:
            self.association_type = {
                association.name: i for i, association in enumerate(sorted(self._lang.associations, key=lambda x: x.name))
            }

        # TODO: Decide if we want to serialize  
        # LanguageGraphAssociationField attributes minimum and maximum

        all_attack_steps = sorted([attack_step
                            for asset in self._lang.assets.values()
                            for attack_step in asset.attack_steps.values()], key=lambda x: x.name)
        # NOTE: This looks odd, but the attack step name is the type of the action
        if split_attack_step_types:
            self.attack_step_type: dict[str, dict[str, int]] = {}
            type_idx = 0
            for attack_step in all_attack_steps:
                if attack_step.asset.name not in self.attack_step_type:
                    self.attack_step_type[attack_step.asset.name] = {}
                if attack_step.name not in self.attack_step_type[attack_step.asset.name]:
                    self.attack_step_type[attack_step.asset.name][attack_step.name] = type_idx
                type_idx += 1
        else:
            self.attack_step_type: dict[str, int] = {
                attack_step.name: i for i, attack_step in enumerate(all_attack_steps)
            }
        # NOTE: The actual logic-class of the attack step
        all_attack_step_classes = set(attack_step.type for attack_step in all_attack_steps)
        self.attack_step_class = {class_name: i for i, class_name in enumerate(all_attack_step_classes)}

        # NOTE: Add None tag for steps without tags
        all_attack_step_tags = [None] + sorted(list(tag for attack_step in all_attack_steps for tag in attack_step.tags))
        self.attack_step_tag = {tag: i for i, tag in enumerate(all_attack_step_tags)}        
        