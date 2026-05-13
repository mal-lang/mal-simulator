from malsim.config.agent_settings import AttackerSettings


def test_attacker_settings_single_entry_points_set() -> None:
    settings = AttackerSettings(
        name='Attacker1',
        entry_points=frozenset({'User:3:phishing', 'Host:0:connect'}),
        goals=frozenset({'Data:2:read'}),
    )
    assert settings.entry_points == frozenset({'User:3:phishing', 'Host:0:connect'})


def test_attacker_settings_single_entry_points_set_in_list() -> None:
    settings = AttackerSettings(
        name='Attacker1',
        entry_points=frozenset({'User:3:phishing', 'Host:0:connect'}),
        goals=frozenset({'Data:2:read'}),
    )
    assert settings.entry_points == frozenset(
        ({'User:3:phishing', 'Host:0:connect'}),
    )


def test_attacker_settings_multiple_entry_points_sets_in_list() -> None:
    settings = AttackerSettings(
        name='Attacker1',
        entry_points=({'User:3:phishing', 'Host:0:connect'}, {'User:4:phishing'}),
        goals=frozenset({'Data:2:read'}),
    )
    assert settings.entry_points == (
        {'User:3:phishing', 'Host:0:connect'},
        {'User:4:phishing'},
    )
