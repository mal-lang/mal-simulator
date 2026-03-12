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
            entry_points=frozenset(d['entry_points']),
            goals=frozenset(d.get('goals', [])),
            ttc_dists=NodePropertyRule.from_optional_dict(d.get('ttc_overrides')),
            policy=policy,
            actionable_steps=NodePropertyRule.from_optional_dict(
                d.get('actionable_steps')
            ),
            rewards=NodePropertyRule.from_optional_dict(d.get('rewards'))
            config=config,
            reward_mode=RewardMode[d.get('reward_mode', 'CUMULATIVE')],
        )

    # Defender
    return DefenderSettings(
        name=name,
        policy=policy,
        observable_steps=NodePropertyRule.from_optional_dict(d.get('observable_steps')),
        actionable_steps=NodePropertyRule.from_optional_dict(d.get('actionable_steps')),
        rewards=NodePropertyRule.from_optional_dict(d.get('rewards')) or global_rewards,
        false_positive_rates=NodePropertyRule.from_optional_dict(
            d.get('false_positive_rates')
        ),
        false_negative_rates=NodePropertyRule.from_optional_dict(
            d.get('false_negative_rates')
        ),
        config=config,
        reward_mode=RewardMode[d.get('reward_mode', 'CUMULATIVE')],
    )
