from maltoolbox.language import LanguageGraph
from malsim.envs.serialization import LangSerializer

def test_serializer() -> None:

    scenario_file = (
        "tests/testdata/langs/org.mal-lang.coreLang-1.0.0.mar"
    )
    lang = LanguageGraph.load_from_file(scenario_file)
    serializer = LangSerializer(lang, split_assoc_types=False, split_attack_step_types=False)

    asset_keys = set(lang.assets.keys())
    serializer_keys = set(serializer.asset_type.keys())
    assert asset_keys == serializer_keys, "Not all asset keys are serialized"
    values = list(serializer.asset_type.values())
    assert len(values) == len(set(values)), "Asset type integer values are not unique"
    assert all(isinstance(v, int) for v in values), "Not all asset types map to integers"

    assoc_names = set(assoc.name for assoc in lang.associations)
    assert set(serializer.association_type.keys()) == assoc_names, "association_type key mismatch"
    assoc_values = list(serializer.association_type.values())
    assert len(assoc_values) == len(set(assoc_values)), "association_type integer values are not unique"
    assert all(isinstance(v, int) for v in assoc_values), "Not all association types map to integers"

    all_attack_steps = sorted(
        [attack_step
            for asset in lang.assets.values()
            for attack_step in asset.attack_steps.values()],
        key=lambda x: x.name)
    attack_step_names = set(a.name for a in all_attack_steps)
    assert set(serializer.attack_step_type.keys()) == attack_step_names, "attack_step_type key mismatch"
    attack_step_values = list(serializer.attack_step_type.values())
    assert len(attack_step_values) == len(set(attack_step_values)), "attack_step_type integer values are not unique"
    assert all(isinstance(v, int) for v in attack_step_values), "Not all attack step types map to integers"

    attack_step_types = set(a.type for a in all_attack_steps)
    assert set(serializer.attack_step_class.keys()) == attack_step_types, "attack_step_class key mismatch"
    attack_step_class_values = list(serializer.attack_step_class.values())
    assert len(attack_step_class_values) == len(set(attack_step_class_values)), "attack_step_class integer values are not unique"
    assert all(isinstance(v, int) for v in attack_step_class_values), "Not all attack step classes map to integers"

    attack_step_tags = set(tag for attack_step in all_attack_steps for tag in attack_step.tags).union({None})
    assert set(serializer.attack_step_tag.keys()) == attack_step_tags, "attack_step_tag key mismatch"
    tag_values = list(serializer.attack_step_tag.values())
    assert len(tag_values) == len(set(tag_values)), "attack_step_tag integer values are not unique"
    assert all(isinstance(v, int) for v in tag_values), "Not all attack step tags map to integers"

    serializer = LangSerializer(lang, split_assoc_types=True, split_attack_step_types=True)

    # Test for split_assoc_types=True, split_attack_step_types=True
    # This should build nested dicts for association_type and attack_step_type

    # Check association_type structure
    for assoc in lang.associations:
        assoc_dict = serializer.association_type.get(assoc.name)
        assert isinstance(assoc_dict, dict), f"association_type[{assoc.name}] should be dict"
        left_type = assoc.left_field.asset.name
        right_type = assoc.right_field.asset.name
        assert left_type in assoc_dict, (
            f"association_type[{assoc.name}] missing left asset type {left_type}"
        )
        assert right_type in assoc_dict[left_type], (
            f"association_type[{assoc.name}][{left_type}] missing right asset type {right_type}"
        )
        id_val = assoc_dict[left_type][right_type]
        assert isinstance(id_val, int), (
            f"Final value in association_type[{assoc.name}][{left_type}][{right_type}] "
            "should be int"
        )

    # Check that all (assoc name, left type, right type) combos are unique
    seen_assoc_ids = set()
    for assoc_dict in serializer.association_type.values():
        for left_dict in assoc_dict.values():
            for right_val in left_dict.values():
                assert right_val not in seen_assoc_ids, "Duplicate assoc_type id found"
                seen_assoc_ids.add(right_val)

    # Check attack_step_type structure
    all_attack_steps = sorted(
        [attack_step
            for asset in lang.assets.values()
            for attack_step in asset.attack_steps.values()],
        key=lambda x: x.name)
    for attack_step in all_attack_steps:
        asset_name = attack_step.asset.name
        attack_name = attack_step.name
        assert asset_name in serializer.attack_step_type, (
            f"attack_step_type missing asset {asset_name}"
        )
        asset_dict = serializer.attack_step_type[asset_name]
        assert attack_name in asset_dict, (
            f"attack_step_type[{asset_name}] missing attack step {attack_name}"
        )
        attack_id = asset_dict[attack_name]
        assert isinstance(attack_id, int), (
            f"attack_step_type[{asset_name}][{attack_name}] should be int"
        )

    # Check that all (asset_type, attack_step_name) combos are unique ids
    seen_attackstep_ids = set()
    for asset_dict in serializer.attack_step_type.values():
        for val in asset_dict.values():
            assert val not in seen_attackstep_ids, "Duplicate attack_step_type id found"
            seen_attackstep_ids.add(val)
