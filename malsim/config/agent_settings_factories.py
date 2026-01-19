from typing import Any

from malsim.config.agent_settings import AgentType, AttackerSettings, DefenderSettings

from malsim.config.node_property_rule import NodePropertyRule
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


def agent_settings_from_dict(
    name: str,
    d: dict[str, Any],
) -> AttackerSettings | DefenderSettings:
    """Load agent settings from a dict"""

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
            entry_points=set(d['entry_points']),
            goals=set(d.get('goals', [])),
            ttc_overrides=NodePropertyRule.from_optional_dict(d.get('ttc_overrides')),
            policy=policy,
            actionable_steps=NodePropertyRule.from_optional_dict(
                d.get('actionable_steps')
            ),
            rewards=NodePropertyRule.from_optional_dict(d.get('rewards')),
            config=config,
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
    )
