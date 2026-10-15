from collections.abc import Set
from typing import Any


from malsim.config.agent_settings import AgentType, AttackerSettings, DefenderSettings

from malsim.config.node_property_rule import NodePropertyRule
from malsim.config.sim_settings import RewardMode
from malsim.policies import (
    BreadthFirstAttacker,
    DepthFirstAttacker,
    KeyboardAgent,
    PassiveAgent,
    DefendCompromisedDefender,
    DefendFutureCompromisedDefender,
    RandomAgent,
    TTCSoftMinAttacker,
    ShortestPathAttacker,
)

policy_name_to_class = {
    'KeyboardAgent': KeyboardAgent,
    'PassiveAgent': PassiveAgent,
    'DepthFirstAttacker': DepthFirstAttacker,
    'BreadthFirstAttacker': BreadthFirstAttacker,
    'TTCSoftMinAttacker': TTCSoftMinAttacker,
    'ShortestPathAttacker': ShortestPathAttacker,
    'DefendCompromisedDefender': DefendCompromisedDefender,
    'DefendFutureCompromisedDefender': DefendFutureCompromisedDefender,
    'RandomAgent': RandomAgent,
}


def _validate_agent_dict(d: dict[str, Any]) -> dict[str, Any]:
    required_keys = {'type', 'policy'}
    allowed_keys = {
        'type',
        'policy',
        'agent_class',
        'config',
        'rewards',
        'false_positive_rates',
        'false_negative_rates',
        'observable_steps',
        'actionable_steps',
        'entry_points',
        'goals',
        'reward_mode',
        'ttc_overrides',
    }
    illegal_keys = d.keys() - allowed_keys
    missing_keys = required_keys - d.keys()

    if illegal_keys:
        raise ValueError(f'Illegal keys in agent scenario settings: {illegal_keys}')

    if illegal_keys:
        raise ValueError(f'Missing keys in agent scenario settings: {missing_keys}')

    return d


def _load_entry_points(d: Any) -> tuple[Set[str], ...] | Set[str]:
    if d is None:
        return frozenset()
    elif isinstance(d, set):
        return frozenset(d)
    elif isinstance(d, (list, tuple)):
        # Can be a list of strings or a list of sets/lists with strings
        if all(isinstance(ep, str) for ep in d):
            return frozenset(d)
        elif all(isinstance(ep, (set, list)) for ep in d):
            return tuple(frozenset(ep) for ep in d)
        else:
            raise ValueError(
                'entry_points list must contain either '
                'all strings or all sets/lists of strings'
            )
    else:
        raise ValueError(
            f'entry_points must be a set or list of strings, got {type(d)}'
        )


def agent_settings_from_dict(
    name: str,
    d: dict[str, Any],
) -> AttackerSettings[str] | DefenderSettings:
    """Load agent settings from a dict"""

    d = _validate_agent_dict(d)
    agent_type = AgentType(d['type'])

    # Resolve policy class if provided
    policy = None
    policy_name = d.get('policy') or d.get('agent_class')
    if policy_name:
        if policy_name not in policy_name_to_class:
            raise LookupError(
                f"Policy class '{policy_name}' not supported. "
                f'Must be one of: {list(policy_name_to_class.keys())}'
            )
        policy = policy_name_to_class[policy_name]

    config = d.get('config', {})

    if agent_type == AgentType.ATTACKER:
        return AttackerSettings(
            name=name,
            entry_points=_load_entry_points(d.get('entry_points')),
            goals=frozenset(d.get('goals', [])),
            ttc_dists=NodePropertyRule.from_optional_dict(d.get('ttc_overrides')),
            policy=policy,
            actionable_steps=NodePropertyRule.from_optional_dict(
                d.get('actionable_steps')
            ),
            rewards=NodePropertyRule.from_optional_dict(d.get('rewards')),
            config=config,
            reward_mode=RewardMode[d.get('reward_mode', 'CUMULATIVE')],
        )

    # Defender
    return DefenderSettings(
        name=name,
        policy=policy,
        observable_steps=NodePropertyRule.from_optional_dict(d.get('observable_steps')),
        actionable_steps=NodePropertyRule.from_optional_dict(d.get('actionable_steps')),
        rewards=NodePropertyRule.from_optional_dict(d.get('rewards')),
        false_positive_rates=NodePropertyRule.from_optional_dict(
            d.get('false_positive_rates')
        ),
        false_negative_rates=NodePropertyRule.from_optional_dict(
            d.get('false_negative_rates')
        ),
        config=config,
        reward_mode=RewardMode[d.get('reward_mode', 'CUMULATIVE')],
    )
