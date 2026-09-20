from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
import json
import logging

from typing import ClassVar, TYPE_CHECKING, Any

from maltoolbox import config

from ..decision_agent import DecisionAgent

if TYPE_CHECKING:
    from maltoolbox.attackgraph import AttackGraphNode
    from ...mal_simulator import AgentState

try:
    from openai import OpenAI
except ModuleNotFoundError as exc:
    if exc.name != 'openai':
        raise

    raise ImportError(
        'OpenAI agent requires the OpenAI package. '
        "Install it with: pip install 'mal-simulator[openai]'"
    ) from exc

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = (
    'You are an agent in a simulated environment. '
    'Based on the current state of the environment, provide the next action to take.'
)

ACTION_RESPONSE_FORMAT: dict[str, Any] = {
    'type': 'json_schema',
    'name': 'action_selection',
    'schema': {
        'type': 'object',
        'properties': {
            'action': {
                'type': 'string',
                'description': 'The name of the chosen action from the valid actions list.',
            }
        },
        'required': ['action'],
        'additionalProperties': False,
    },
    'strict': True,
}


class Observation(Enum):
    NONE = 'none'
    ACTION_SURFACE = 'action_surface'


@dataclass
class OpenAIAgentConfig:
    api_key: str
    observation_mode: Observation = Observation.NONE
    allow_wait: bool = True
    seed: int | None = None

    # OpenAI Settings
    model: str
    instructions: str | None = DEFAULT_INSTRUCTIONS


def parse_agent_config(config_dict: dict[str, Any]) -> OpenAIAgentConfig:
    """Parse a dict into an AgentConfig, validating the fields."""
    try:
        api_key = config_dict['api_key']
    except KeyError as e:
        raise ValueError("Missing required field: 'api_key'") from e
    try:
        observation_mode = Observation(config_dict.get('observation_mode', 'none'))
    except ValueError as e:
        raise ValueError(
            f'Invalid observation_mode: {config_dict.get("observation_mode")}. '
            f'Valid options are: {[ao.value for ao in Observation]}'
        ) from e
    seed = config_dict.get('seed')
    if seed is not None and not isinstance(seed, int):
        raise ValueError(f'Seed must be an integer or None, got {type(seed)}')
    allow_wait = config_dict.get('allow_wait', True)
    if not isinstance(allow_wait, bool):
        raise ValueError(f'allow_wait must be a boolean, got {type(allow_wait)}')

    # OpenAI Settings
    model = config_dict.get('model')
    if not isinstance(model, str):
        raise ValueError(
            'Missing or invalid required field: "model"'
            f' (expected str, got {type(model)})'
        )
    instructions = config_dict.get('instructions', DEFAULT_INSTRUCTIONS)
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError('Invalid field: "instructions". Must be a string or None.')
    return OpenAIAgentConfig(
        api_key=api_key,
        observation_mode=observation_mode,
        allow_wait=allow_wait,
        seed=seed,
        model=model,
        instructions=instructions,
    )


class OpenAIAgent(DecisionAgent):
    """An agent driven by an LLM, with different prompts and observations."""

    name = 'OpenAI LLM Agent'

    _default_settings: ClassVar[OpenAIAgentConfig] = OpenAIAgentConfig(
        observation_mode=Observation.NONE,
        seed=None,
        instructions=DEFAULT_INSTRUCTIONS,
        allow_wait=True,
    )

    def __init__(self, agent_config: OpenAIAgentConfig | dict[str, Any]) -> None:
        """Initialize am LLM agent.

        Args:
            agent_config: Dict with settings to override defaults
        """
        self.config = (
            parse_agent_config(agent_config)
            if isinstance(agent_config, dict)
            else agent_config
        )

        settings = replace(self._default_settings, **self.config.__dict__)

        self.client = OpenAI(api_key=settings.api_key)

        self.config.instructions = (
            '\nOutput the action in JSON format: {"action": "<action_name>"}'
        )

        self.history: list[dict[str, Any]] = []

    def get_next_action(
        self, agent_state: AgentState, **kwargs: Any
    ) -> AttackGraphNode | None:
        """Receive the next action according to the LLM."""

        nodes_by_name: dict[str, AttackGraphNode | None] = {
            node.full_name: node for node in agent_state.action_surface
        }
        if self.config.allow_wait:
            nodes_by_name['WAIT'] = None

        input = [
            {'role': 'user', 'content': f'Valid actions:\n{"\n".join(nodes_by_name)}'}
        ]

        response = self.client.responses.create(
            model=self.config.model,
            instructions=self.config.instructions,
            input=input,
            text={'format': ACTION_RESPONSE_FORMAT},
        )

        try:
            parsed = json.loads(response.output_text)
            action_name = parsed['action']
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.error(
                'Failed to parse action from LLM response: %r', response.output_text
            )
            return None

        node = nodes_by_name.get(action_name)
        if node is None:
            logger.error(
                f'LLM chose action %r which is not in the action surface', action_name
            )
        return node
