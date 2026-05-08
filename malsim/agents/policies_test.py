import math
from unittest import mock

import .policies


class TestProbabilisticRepeatPolicy(unittest.TestCase):

  def test_no_repeat_policy(self):
    """Tests for policies.NoRepeatPolicy."""
    self.assertFalse(policies.NoRepeatPolicy({}).is_repeatable_action(object()))

  def test_probabitlistic_is_repeatable_action(self):
    """Tests for policies.ProbabilisticRepeatPolicy."""
    policy = policies.ProbabilisticRepeatPolicy(
        {"type": "probabilistic_repeat", "params": {"threshold": 0.2}}
    )

    with mock.patch("random.random", return_value=policy.threshold):
      result = policy.is_repeatable_action(object())
      self.assertTrue(result)

    with mock.patch(
        "random.random", return_value=math.nextafter(policy.threshold, 1)
    ):
      result = policy.is_repeatable_action(object())
      self.assertFalse(result)

  @mock.patch.object(policies, "NoRepeatPolicy")
  @mock.patch.object(policies, "ProbabilisticRepeatPolicy")
  def test_policy_factory(self, *mock_policies: mock.MagicMock):
    """Tests for policies.repeatable_action_policy_factory."""
    policies.repeatable_action_policy_factory({"type": "no_repeat"})
    mock_policies[-1].assert_called_once()

    policies.repeatable_action_policy_factory({"type": "probabilistic_repeat"})
    mock_policies[-2].assert_called_once()

    with self.assertRaisesRegex(ValueError, "Unknown repeatable action policy"):
      policies.repeatable_action_policy_factory({"type": "unknown"})


if __name__ == "__main__":
  unittest.main()
