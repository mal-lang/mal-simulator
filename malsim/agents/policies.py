"""Policies to configure agent behavior."""

import abc
import random
from typing import Any, Literal, TypedDict, override

from maltoolbox import attackgraph


class RepeatableActionPolicy(abc.ABC):
  """Policy interface for whether an action should be considered repeatable.

  A repeatable action policy is not involved in the decision policy of an agent
  at all, i.e. not even when the agent needs to decides about repeating
  a repeatable action.
  """

  def __init__(self, config: dict[str, Any]):
    pass

  @abc.abstractmethod
  def is_repeatable_action(self, step: attackgraph.AttackGraphNode) -> bool:
    """Whether the passed in action should be considered repeatable.

    Args:
      step: The attack graph node to examine repeatability for.

    Returns:
      True if repeatable, False if not.
    """


class NoRepeatPolicyConfig(TypedDict):
  type: Literal["no_repeat"]


class NoRepeatPolicy(RepeatableActionPolicy):
  """A policy that considers all steps as non-repeatable."""

  def __init__(self, config: NoRepeatPolicyConfig):
    pass

  @override
  def is_repeatable_action(self, step: attackgraph.AttackGraphNode) -> bool:
    """Return False regardless of passed-in action."""
    return False


class ProbabilisticRepeatPolicyConfig(TypedDict, total=True):
  type: Literal["probabilistic_repeat"]

  params: TypedDict(
      "_ProbabilisticRepeatPolicyParams",
      {
          "threshold": float,
      },
  )


class ProbabilisticRepeatPolicy(RepeatableActionPolicy):
  """Randomly designate an action as repeatable or not."""

  def __init__(self, config: ProbabilisticRepeatPolicyConfig):
    super().__init__(config)
    self.threshold = config["params"]["threshold"]

  @override
  def is_repeatable_action(self, step: attackgraph.AttackGraphNode) -> bool:
    return random.random() <= self.threshold


RepeatableActionPolicyConfig = ProbabilisticRepeatPolicyConfig | NoRepeatPolicy


def repeatable_action_policy_factory(
    config: RepeatableActionPolicyConfig,
) -> RepeatableActionPolicy:
  """Return a policy object conforming to the passed-in config.

  What policy to instantiate is defined by the "type" key of "config". The
  "params" subdict of "config" will be passed in as argument to the policy
  constructor.

  Args:
    config: A dict with configuration for the selected policy

  Returns:
    An instantiated policy object.
  """
  if config["type"] == "no_repeat":
    policy = NoRepeatPolicy(config)
  elif config["type"] == "probabilistic_repeat":
    policy = ProbabilisticRepeatPolicy(config)
  else:
    raise ValueError(f"Unknown repeatable action policy: {config["type"]}")

  return policy
