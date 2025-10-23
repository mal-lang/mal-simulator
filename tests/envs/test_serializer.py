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

    attack_step_tags = set(tag for attack_step in all_attack_steps for tag in attack_step.tags)
    assert set(serializer.attack_step_tag.keys()) == attack_step_tags, "attack_step_tag key mismatch"
    tag_values = list(serializer.attack_step_tag.values())
    assert len(tag_values) == len(set(tag_values)), "attack_step_tag integer values are not unique"
    assert all(isinstance(v, int) for v in tag_values), "Not all attack step tags map to integers"