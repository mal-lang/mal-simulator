"""Agent classes and helper methods compatible with the KTH MAL Simulator."""
from collections.abc import Iterable
import enum
import functools
import random
from typing import TypeAlias

from absl import logging
from malsim import agents
from malsim import mal_simulator as sim
from maltoolbox import attackgraph
from maltoolbox import language
import numpy as np
from typing_extensions import TypedDict

import .policies


class AgentProfile(enum.Enum):
  """Enumeration of supported labels for produced synthetic logs.

  These are enumerated mostly for avoiding typos and ambiguity in what labels
  can be used. Other than being included in every generated log under the "type"
  key, these labels have no other effect.
  """

  BENIGN = "benign"
  MALICIOUS = "malicious"
  USER = "user"
  ATTACKER = "attacker"

  def __str__(self) -> str:
    return self.name


class EventLogger:
  """Collects logs from using agents during attack simulations."""

  def __init__(
      self,
      attack_graph: attackgraph.AttackGraph,
      agent_class_name: str,
      agent_profile_name: str,
  ):
    """Initialize an event logger to produce synthetic event logs.

    Args:
      attack_graph: An attack graph to lookup assets in.
      agent_class_name: The name of the agent whose actions will be logged.
      agent_profile_name: The profile of the agent whose actions will be logged.
    """
    self.attack_graph = attack_graph
    self.agent_class_name = agent_class_name
    self.agent_profile_name = agent_profile_name

    self.logs: list[dict] = []

  @functools.cached_property
  def subasset_names(self) -> dict[str, set[str]]:
    """A mapping from asset to all related subassets, including self."""
    subasset_names: dict[str, set[str]] = {}
    for asset in self.attack_graph.lang_graph.assets.values():
      subasset_names[asset.name] = {
          subasset.name for subasset in asset.sub_assets
      }

    return subasset_names

  def collect_logs(
      self, attack_step: attackgraph.AttackGraphNode, current_step: int
  ) -> None:
    """Generates logs and saves them to self.logs.

    Args:
      attack_step: The attack step to be performed that will trigger the event
        log.
      current_step: The simulation step when the attack step is being
        compromised.
    """
    for _, detector in attack_step.detectors.items():
      labeled_assets = self._get_context_assets(
          detector,
          self.attack_graph.attackers[0].reached_attack_steps,
          attack_step,
      )

      self.logs.append({
          "timestep": current_step,
          "_detector": detector.name,
          "asset": attack_step.model_asset.name,
          "attack_step": attack_step.name,
          "agent": self.agent_class_name,
          "agent_profile": self.agent_profile_name,
          **labeled_assets,
      })

      logging.debug(
          "Detector %s triggered on %s", detector.name, attack_step.full_name
      )

  def _get_context_assets(
      self,
      detector: language.Detector,
      reached_steps: Iterable[attackgraph.AttackGraphNode],
      attack_step: attackgraph.AttackGraphNode,
  ) -> dict[str, str]:
    """Finds model assets that satisfy the detector's context.

    Only model assets that have at least one attack step already compromised are
    considered as possible candidates for the detector context as well as the
    current asset that holds the attack step whose detectors are being resolved.
    If multiple model assets of the type defined in the detector context, the
    last one compromised is selected.

    Args:
      detector: The detector whose context assets needs resolving
      reached_steps: An iterable containing candidate (i.e. already compromised)
        steps to populate the detector context.
      attack_step: The attack_step holding the detector.

    Returns:
      A dictionary where keys are the detector context labels and values are the
      model assets that have been selected to populate the context.
    """
    context_assets = {}

    for label, lang_graph_asset in detector.context.items():
      subasset_names = self.subasset_names[lang_graph_asset.name]

      if attack_step.model_asset.type in subasset_names:
        # The model asset that holds (the step that holds) the detector matches
        # the type required by the detector, thus is selected. This is done to
        # support cases where a log needs to reference the resource that
        # triggers the log itself. In the future a solution at the MAL-spec
        # level (e.g. using `__self__` instead of an asset type in the context
        # definition) would make this more robust.
        context_assets[label] = attack_step.model_asset.name
        continue

      try:
        *_, asset = (
            step.model_asset
            for step in reached_steps
            if step.model_asset.type in subasset_names
        )
      except ValueError as e:
        msg = (
            f"Context {detector.context} cannot be satisfied "
            f"for step {attack_step.full_name}. No {lang_graph_asset.name} "
            "was compromised already."
        )
        raise ValueError(msg) from e

      context_assets[label] = asset.name

    return context_assets


class SearcherAgentConfig(TypedDict, total=False):
  """Supported settings by searcher agents (BFS, DFS)."""

  # Probability of an agent waiting (doing nothing) in a turn
  wait_probability: float

  # A maltoolbox.attackgraph.attackgraph.AttackGraph object
  attack_graph: attackgraph.AttackGraph

  # The profile of the agent, e.g. malicious, benign, etc.
  profile: AgentProfile

  # From agents.BreadthFirstAttacker
  randomize: bool
  seed: int | None

  # Repeatable Action Policy
  repeatable_action_policy: policies.RepeatableActionPolicyConfig


class BreadthFirstAttacker(agents.BreadthFirstAttacker):
  """BFS agent with randomized selection of next action at current level.

  Randomization happens among the available next steps in the agents action
  surface (as received by the simulator) that respect the breadth-first
  policy.
  """

  _default_settings: SearcherAgentConfig = {
      "wait_probability": 0.0,
      **agents.BreadthFirstAttacker._default_settings,
  }

  def __init__(self, agent_config: SearcherAgentConfig):
    """Initializes the BreadthFirstAttacker agent.

    Args:
      agent_config: A dictionary containing the agent's configuration. Must
        include the 'attack_graph' key.

    Raises:
      ValueError: If the 'agent_config' dictionary is missing the
          required 'attack_graph' key.
    """
    super().__init__(agent_config)

    self.current_step = -1
    self._wait_probability = self._settings["wait_probability"]
    self._profile: AgentProfile = agent_config.get(
        "profile", AgentProfile.MALICIOUS
    )
    self._repeat_action_probability = agent_config.get(
        "repeat_action_probability", 0.0
    )

    if "attack_graph" not in self._settings:
      # "attack_graph" is the only setting without a default that is required,
      # thus checking for its presence.
      raise ValueError("Agent configuration requires 'attack_graph'.")

    self.attack_graph = self._settings["attack_graph"]

    self.event_logger = EventLogger(
        self.attack_graph, self.__class__.__name__, self._profile.value
    )

    self._repeatable_actions_set: set[attackgraph.AttackGraphNode] = set()
    self._repeatable_actions_list: list[attackgraph.AttackGraphNode] = []
    self._repeatable_action_policy = policies.repeatable_action_policy_factory(
        agent_config.get("repeatable_action_policy", {"type": "no_repeat"})
    )

  def _select_next_target(self) -> None:
    """Selects a next target or None and saves it to self._current_target."""
    if random.random() < self._wait_probability:
      self._current_target = None
    elif (
        self._repeatable_actions_list
        and random.random() < self._repeat_action_probability
    ):
      self._current_target = random.choice(self._repeatable_actions_list)
    else:
      super()._select_next_target()

    current_target = self._current_target

    if current_target is None:
      return

    if current_target.detectors:
      self.event_logger.collect_logs(current_target, self.current_step)

    if (
        current_target not in self._repeatable_actions_set
        and self._repeatable_action_policy.is_repeatable_action(current_target)
    ):
      self._repeatable_actions_set.add(current_target)
      self._repeatable_actions_list.append(current_target)


class DepthFirstAttacker(BreadthFirstAttacker):
  """BFS agent with randomized selection of next action at current level."""

  # Used in agents.DepthFirstAttacker to convert agents.BreadthFirstAttacker to
  # depth-first. Replicating this pattern here for our version of the BFS and
  # DFS agents.
  _extend_method = "extend"


class QuasiRealisticAgentConfig(TypedDict, total=False):
  """Supported settings by quasi realistic agents."""

  # Probability of an agent waiting (doing nothing) in a turn
  wait_probability: float

  # The profile of the agent, e.g. malicious, benign, etc.
  profile: AgentProfile

  # A maltoolbox.attackgraph.attackgraph.AttackGraph object
  attack_graph: attackgraph.AttackGraph

  # The default priority for steps that don't have a corresponding entry in the
  # `priorities` config.
  default_priority: int

  # Dict mapping attack graph node names to priority numbers
  priorities: dict[str, int]

  # Whether to boost target priorities by the current simulation step.
  # If True, `priority + self.current_step` is used; otherwise, just `priority`.
  enable_recency_boost: bool

  # Repeatable Action Policy
  repeatable_action_policy: policies.RepeatableActionPolicyConfig

  # Probability of selecting a repeatable action as next action
  repeat_action_probability: float


class QuasiRealisticAgent(agents.DecisionAgent):
  """Preference-based agent using attack step type priorities.

  A priority is a non-negative integer. A priority of 0 disabled the step. The
  higher the priority the more probable the step will be picked up if it exists
  in the action surface. Conversion of priorities to probabilities happens using
  softmax. Priorities are set via the `priorities` agent_config dictionary
  setting. A key in the priorities dict can be the name of an attack graph node,
  of a model asset, a language graph step or a language graph asset. The
  priority of an attack step is resolved in the same order of precedence.

  See go/qr-agent for more details.
  """

  def __init__(self, agent_config: QuasiRealisticAgentConfig):
    super().__init__()

    self._targets: dict[str, int] = {}
    self._started = False
    self.current_step = -1
    self._profile: AgentProfile = agent_config.get(
        "profile", next(iter(AgentProfile))
    )
    self._wait_probability = agent_config.get("wait_probability", 0.0)
    self._default_priority = agent_config.get("default_priority", 5)
    self._priorities = agent_config.get("priorities", {})
    self._enable_recency_boost = agent_config.get("enable_recency_boost", False)
    self._repeat_action_probability = agent_config.get(
        "repeat_action_probability", 0.0
    )

    if "attack_graph" not in agent_config:
      raise ValueError("Agent configuration requires 'attack_graph'.")

    self.attack_graph = agent_config["attack_graph"]
    self._full_name_to_node = self.attack_graph._full_name_to_node

    self.event_logger = EventLogger(
        self.attack_graph, self.__class__.__name__, self._profile.value
    )

    self._repeatable_actions_set: set[attackgraph.AttackGraphNode] = set()
    self._repeatable_actions_list: list[attackgraph.AttackGraphNode] = []
    self._repeatable_action_policy = policies.repeatable_action_policy_factory(
        agent_config.get("repeatable_action_policy", {"type": "no_repeat"})
    )

  def get_next_action(
      self,
      agent_state: sim.MalSimAgentStateView,
  ) -> attackgraph.AttackGraphNode | None:
    new_nodes = (
        # See https://github.com/mal-lang/mal-simulator/issues/125
        agent_state.step_action_surface_additions
        if self._started
        else agent_state.action_surface
    )

    self._started = True

    priorities = self._priorities
    for node in new_nodes:
      if node.full_name in priorities:
        priority = priorities[node.full_name]
      elif node.model_asset.name in priorities:
        priority = priorities[node.model_asset.name]
      elif node.lg_attack_step.full_name in priorities:
        priority = priorities[node.lg_attack_step.full_name]
      elif node.lg_attack_step.name in priorities:
        priority = priorities[node.lg_attack_step.name]
      elif node.model_asset.lg_asset.name in priorities:
        priority = priorities[node.model_asset.lg_asset.name]
      else:
        priority = self._default_priority

      if not priority:
        continue

      if self._enable_recency_boost:
        self._targets[node.full_name] = priority + self.current_step
      else:
        self._targets[node.full_name] = priority

    # Removals happen when defender agents operate in parallel to an attacker.
    # We don't plan to use a defender.
    # if agent_state.step_action_surface_removals:
    #   pass

    if random.random() < self._wait_probability or not self._targets:
      # In either case there is no selected action to perform, either due to
      # waiting or an empty, non-zero-priority action surface.
      current_target = None
    elif (
        self._repeatable_actions_list
        and random.random() < self._repeat_action_probability
    ):
      current_target = random.choice(self._repeatable_actions_list)
    else:
      current_target_name = random.choices(
          list(self._targets.keys()),
          weights=softmax(list(self._targets.values())),
      )[0]

      del self._targets[current_target_name]

      current_target = self._full_name_to_node[current_target_name]

    if current_target is None:
      return None

    self._current_target = current_target

    if current_target.detectors:
      self.event_logger.collect_logs(current_target, self.current_step)

    if (
        current_target not in self._repeatable_actions_set
        and self._repeatable_action_policy.is_repeatable_action(current_target)
    ):
      self._repeatable_actions_set.add(current_target)
      self._repeatable_actions_list.append(current_target)

    return current_target


LogEmittingAgents: TypeAlias = (
    BreadthFirstAttacker | DepthFirstAttacker | QuasiRealisticAgent
)


class AgentType(enum.Enum):
  """Enumeration of different agent types used in the simulation."""

  BFS = BreadthFirstAttacker
  DFS = DepthFirstAttacker
  QR = QuasiRealisticAgent

  def __str__(self) -> str:
    return self.name


def softmax(priorities: list[int]) -> list[float]:
  """Compute the softmax of a list of integer priorities.

  The softmax function converts a vector of raw scores (priorities) into a
  probability distribution. The output is a list of floats where each value
  represents the probability of the corresponding input priority.

  Args:
      priorities: A list of integer values representing the priorities.

  Returns:
      A list of floating-point numbers representing the softmax probabilities,
      with each value between 0 and 1, and the sum of all values equal to 1.
  """
  np_priorities = np.asarray(priorities)
  exp = np.exp(np_priorities - np.max(np_priorities))
  return (exp / np.sum(exp)).tolist()
