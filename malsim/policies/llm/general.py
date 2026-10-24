from __future__ import annotations
from collections.abc import Callable, Set
from dataclasses import dataclass, replace
from enum import Enum
import logging
import random
from abc import ABC, abstractmethod

from collections import deque
from typing import ClassVar, TYPE_CHECKING, Any

from ..decision_agent import DecisionAgent

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ...mal_simulator import AgentState

logger = logging.getLogger(__name__)


class Observation(Enum):
    ACTION_SURFACE = 'action_surface'


@dataclass
class LLMAgentConfig:
    api_key: str
    observation_mode: Observation = Observation.ACTION_SURFACE
    seed: int | None = None


def parse_llm_agent_config(config_dict: dict[str, Any]) -> LLMAgentConfig:
    """Parse a dict into an AgentConfig, validating the fields."""
    try:
        api_key = config_dict['api_key']
    except KeyError as e:
        raise ValueError("Missing required field: 'api_key'") from e
    try:
        observation_mode = Observation(
            config_dict.get('observation_mode', 'action_surface')
        )
    except ValueError as e:
        raise ValueError(
            f'Invalid observation_mode: {config_dict.get("observation_mode")}. '
            f'Valid options are: {[ao.value for ao in Observation]}'
        ) from e
    seed = config_dict.get('seed')
    if seed is not None and not isinstance(seed, int):
        raise ValueError(f'Seed must be an integer or None, got {type(seed)}')
    return LLMAgentConfig(api_key=api_key, observation_mode=observation_mode, seed=seed)


class LLMAgent(DecisionAgent, ABC):
    """A Breadth-First agent, with possible randomization at each level."""

    # A human-friendly name for the agent.
    name = 'LLM Agent'

    _default_settings: ClassVar[LLMAgentConfig] = LLMAgentConfig(
        observation_mode=Observation.ACTION_SURFACE,
        seed=None,
    )

    @abstractmethod
    def client_setup(self) -> None:
        """Set up the LLM client. This method should be implemented by subclasses."""

    def __init__(self, agent_config: LLMAgentConfig | dict[str, Any]) -> None:
        """Initialize am LLM agent.

        Args:
            agent_config: Dict with settings to override defaults
        """
        config = (
            parse_llm_agent_config(agent_config)
            if isinstance(agent_config, dict)
            else agent_config
        )

        settings = replace(self._default_settings, **config.__dict__)
