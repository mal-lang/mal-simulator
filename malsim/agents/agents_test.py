import unittest
from unittest import mock

from malsim import agents as malsim_agents
from malsim import mal_simulator as malsim
from maltoolbox import attackgraph
from maltoolbox import language
from maltoolbox import model

import agents
import policies


def _create_mock_attack_graph(test_case: unittest.TestCase) -> mock.Mock:
  """Create a mock attack graph.

  The mock attack graph is a caricature of an attack graph (e.g. OR/AND
  steps), nor does it have connected nodes but suffices for this test case.

  Args:
    test_case: A unittest.TestCase object (e.g. `self`).

  Returns:
    A mock attack graph.
  """
  # Asset examples taken from gcplang
  asset_service_account = mock.Mock(spec=language.LanguageGraphAsset)
  asset_service_account.name = 'ServiceAccount'
  asset_service_account.sub_assets = [asset_service_account]

  asset_google_account = mock.Mock(spec=language.LanguageGraphAsset)
  asset_google_account.name = 'GoogleAccount'
  asset_google_account.sub_assets = [asset_google_account]

  asset_principal = mock.Mock(spec=language.LanguageGraphAsset)
  asset_principal.name = 'Principal'
  asset_principal.sub_assets = [
      asset_principal,
      asset_service_account,
      asset_google_account,
  ]

  asset_project = mock.Mock(spec=language.LanguageGraphAsset)
  asset_project.name = 'Project'
  asset_project.sub_assets = [asset_project]

  lang_graph = mock.Mock(
      spec=language.LanguageGraph,
      assets={
          'Principal': asset_principal,
          'ServiceAccount': asset_service_account,
          'GoogleAccount': asset_google_account,
          'Project': asset_project,
      },
  )

  model_project1 = mock.Mock(spec=model.ModelAsset)
  model_project1.name = 'gcp_sec_demo'
  model_project1.type = asset_project.name
  model_project1.lg_asset = asset_project

  model_project2 = mock.Mock(spec=model.ModelAsset)
  model_project2.name = 'joarjox_project'
  model_project2.type = asset_project.name
  model_project2.lg_asset = asset_project

  model_service_account = mock.Mock(spec=model.ModelAsset)
  model_service_account.name = 'ServiceAccount1'
  model_service_account.type = asset_service_account.name
  model_service_account.lg_asset = asset_service_account

  compromised_step11 = mock.Mock(spec=attackgraph.AttackGraphNode)
  compromised_step11.name = 'compromised_step1'
  compromised_step11.full_name = 'gcp_sec_demo:compromised_step1'
  compromised_step11.model_asset = model_project1
  compromised_step11.lg_attack_step = mock.Mock(
      spec=language.LanguageGraphAsset
  )
  compromised_step11.lg_attack_step.name = 'compromised_step1'
  compromised_step11.lg_attack_step.full_name = 'Project:compromised_step1'
  compromised_step11.detectors = {}

  compromised_step12 = mock.Mock(spec=attackgraph.AttackGraphNode)
  compromised_step12.name = 'compromised_step2'
  compromised_step12.full_name = 'gcp_sec_demo:compromised_step2'
  compromised_step12.model_asset = model_project1
  compromised_step12.lg_attack_step = mock.Mock(
      spec=language.LanguageGraphAsset
  )
  compromised_step12.lg_attack_step.name = 'compromised_step2'
  compromised_step12.lg_attack_step.full_name = 'Project:compromised_step2'
  compromised_step12.detectors = {}

  compromised_step21 = mock.Mock(spec=attackgraph.AttackGraphNode)
  compromised_step21.name = 'compromised_step1'
  compromised_step21.full_name = 'joarjox_project:compromised_step1'
  compromised_step21.model_asset = model_project2
  compromised_step21.lg_attack_step = mock.Mock(
      spec=language.LanguageGraphAsset
  )
  compromised_step21.lg_attack_step.name = 'compromised_step1'
  compromised_step21.lg_attack_step.full_name = 'Project:compromised_step1'
  compromised_step21.detectors = {}

  compromised_step22 = mock.Mock(spec=attackgraph.AttackGraphNode)
  compromised_step22.name = 'compromised_step2'
  compromised_step22.full_name = 'joarjox_project:compromised_step2'
  compromised_step22.model_asset = model_project2
  compromised_step22.lg_attack_step = mock.Mock(
      spec=language.LanguageGraphAsset
  )
  compromised_step22.lg_attack_step.name = 'compromised_step2'
  compromised_step22.lg_attack_step.full_name = 'Project:compromised_step2'
  compromised_step22.detectors = {}

  target_step = mock.Mock(spec=attackgraph.AttackGraphNode)
  target_step.name = 'assume'
  target_step.full_name = 'ServiceAccount1:assume'
  target_step.model_asset = model_service_account
  target_step.lg_attack_step = mock.Mock(spec=language.LanguageGraphAsset)
  target_step.lg_attack_step.name = 'assume'
  target_step.lg_attack_step.full_name = 'ServiceAccount:assume'
  target_step.detectors = {}

  future_step1 = mock.Mock(spec=attackgraph.AttackGraphNode)
  future_step1.name = 'delete'
  future_step1.full_name = 'ServiceAccount1:delete'
  future_step1.model_asset = model_service_account
  future_step1.lg_attack_step = mock.Mock(spec=language.LanguageGraphAsset)
  future_step1.lg_attack_step.name = 'delete'
  future_step1.lg_attack_step.full_name = 'ServiceAccount:delete'
  future_step1.detectors = {}

  future_step2 = mock.Mock(spec=attackgraph.AttackGraphNode)
  future_step2.name = 'create'
  future_step2.full_name = 'ServiceAccount1:create'
  future_step2.model_asset = model_service_account
  future_step2.lg_attack_step = mock.Mock(spec=language.LanguageGraphAsset)
  future_step2.lg_attack_step.name = 'create'
  future_step2.lg_attack_step.full_name = 'ServiceAccount:create'

  detector1 = mock.Mock(spec=language.Detector)
  # Method name Create probably does not exist in the iam.googleapis.com
  # service, don't try to look it up.
  detector1.name = 'iam.googleapis.com.Create'
  ctx = {
      'project': asset_project,
      'principalEmail': asset_principal,
  }
  detector1.context = mock.Mock(spec=language.Context, **ctx)
  detector1.context.items = mock.Mock(return_value=ctx.items())

  detector2 = mock.Mock(spec=language.Detector)
  detector2.name = 'some_other_detector'
  ctx = {'some_label': asset_project}
  detector2.context = mock.Mock(spec=language.Context, **ctx)
  detector2.context.items = mock.Mock(return_value=ctx.items())
  future_step2.detectors = {
      detector1.name: detector1,
      detector2.name: detector2,
  }

  MockAttackGraphClass = mock.patch.object(
      attackgraph, 'AttackGraph', autospec=True
  )
  mock_attack_graph_class_context = MockAttackGraphClass.start()
  attack_graph = mock_attack_graph_class_context.return_value
  attack_graph.lang_graph = lang_graph
  attack_graph.nodes = [
      compromised_step11,
      compromised_step12,
      compromised_step21,
      compromised_step22,
      target_step,
      future_step1,
      future_step2,
  ]
  attack_graph._full_name_to_node = {
      node.full_name: node for node in attack_graph.nodes
  }

  attack_graph.attackers = [
      mock.Mock(spec=attackgraph.Attacker, reached_attack_steps=[])
  ]

  test_case.addCleanup(MockAttackGraphClass.stop)

  return attack_graph


class TestBfsAgent(unittest.TestCase):
  """Tests for simulate.BreadthFirstAttacker."""

  def setUp(self) -> None:
    """Instantiate a BreadthFirstAttacker with a mock attack graph.

    Only the instantiated agent is provided to the test methods (via `self`) and
    access to any mock object in the test methods is done via `self.agent`.
    """
    super().setUp()

    self.super_select_next_target = self.enterContext(
        mock.patch.object(
            malsim_agents.BreadthFirstAttacker, '_select_next_target'
        )
    )
    self.enterContext(mock.patch.object(agents, 'EventLogger', autospec=True))
    self.enterContext(
        mock.patch.object(
            policies, 'repeatable_action_policy_factory', autospec=True
        )
    )

    attack_graph = _create_mock_attack_graph(self)

    agent = agents.BreadthFirstAttacker({
        'attack_graph': attack_graph,
        'wait_probability': 0.3,
        'repeat_action_probability': 0.05,
    })

    agent._current_target = attack_graph.nodes[6]
    agent._repeatable_actions_set = {attack_graph.nodes[5]}
    agent._repeatable_actions_list = [attack_graph.nodes[5]]

    self.agent = agent

  def test_failure_when_no_attack_graph_given(self) -> None:
    """Ensure failure if no attack_graph config option is given."""
    with self.assertRaisesRegex(ValueError, "requires 'attack_graph'."):
      agents.BreadthFirstAttacker({})

  def test_initializing_agent_bf_attacker(self) -> None:
    """Ensure agents get initialized properly."""
    self.assertEqual(self.agent.current_step, -1)

  @mock.patch('random.random', return_value=0.2)
  def test_agent_can_choose_to_wait_and_do_nothing(
      self, _: mock.MagicMock
  ) -> None:
    """Ensure the agent can properly wait if it decides so."""
    self.agent._select_next_target()

    self.assertEqual(self.agent._current_target, None)
    self.super_select_next_target.assert_not_called()

  @mock.patch('random.random', return_value=1.0)
  def test_agent_can_choose_to_act_and_select_next_target(
      self, _: mock.MagicMock
  ) -> None:
    """Ensure the agent can properly act if it decides to."""
    self.agent._select_next_target()

    self.super_select_next_target.assert_called_once()
    self.agent.event_logger.collect_logs.assert_called_once()

  @mock.patch('random.random', return_value=0.01)
  def test_agent_repeats_action(self, _: mock.MagicMock) -> None:
    """Ensure the agent can properly act if it decides to."""
    self.agent._wait_probability = 0.0
    self.agent._select_next_target()

    self.super_select_next_target.assert_not_called()
    self.assertEqual(
        self.agent._current_target, self.agent.attack_graph.nodes[5]
    )


class TestQuasiRealisticAgent(unittest.TestCase):
  """Tests for agents.QuasiRealisticAgent."""

  def setUp(self):
    super().setUp()

    mock_attack_graph = _create_mock_attack_graph(self)

    self.enterContext(mock.patch.object(agents, 'softmax', autospec=True))
    self.enterContext(mock.patch.object(agents, 'EventLogger', autospec=True))
    self.enterContext(
        mock.patch.object(
            policies, 'repeatable_action_policy_factory', autospec=True
        )
    )
    self.enterContext(
        mock.patch(
            'random.choices',
            autospec=True,
            return_value=[mock_attack_graph.nodes[-1].full_name],
        )
    )

    agent = agents.QuasiRealisticAgent({
        'attack_graph': mock_attack_graph,
        'wait_probability': 0.0,
        'default_priority': 1,
        'priorities': {
            'Project': 2,
            'Project:compromised_step2': 3,
            'gcp_sec_demo': 4,
            'gcp_sec_demo:compromised_step2': 5,
            'ServiceAccount1:delete': 0,
        },
    })

    self.agent = agent
    self.mock_state = mock.Mock(
        spec=malsim.MalSimAgentStateView, action_surface=mock_attack_graph.nodes
    )

  def test_initializing_qr_agent(self) -> None:
    """Ensure agents get initialized properly."""
    self.assertEqual(self.agent._started, False)
    self.assertEqual(self.agent.current_step, -1)

  def test_priority_resolution(self):
    next_target = self.agent.get_next_action(self.mock_state)

    nodes = self.agent.attack_graph.nodes

    expected_targets = {
        nodes[0].full_name: 4,
        nodes[1].full_name: 5,
        nodes[2].full_name: 2,
        nodes[3].full_name: 3,
        nodes[4].full_name: 1,
        # nodes[5].full_name: 0,  # disabled
        # nodes[6].full_name: 1,  # chosen, see random.choices mock
    }

    self.assertEqual(next_target, self.mock_state.action_surface[6])
    self.assertEqual(self.agent._targets, expected_targets)

  def test_no_targets_possible(self):
    self.agent._default_priority = 0
    self.agent._priorities = {}

    next_target = self.agent.get_next_action(self.mock_state)

    self.assertEqual(next_target, None)
    self.assertEqual(self.agent._targets, {})

  def test_qr_agent_can_wait(self):
    self.agent._wait_probability = 1.0

    next_target = self.agent.get_next_action(self.mock_state)

    nodes = self.agent.attack_graph.nodes

    expected_targets = {
        nodes[0].full_name: 4,
        nodes[1].full_name: 5,
        nodes[2].full_name: 2,
        nodes[3].full_name: 3,
        nodes[4].full_name: 1,
        # nodes[5].full_name: 0,  # disabled
        nodes[6].full_name: 1,  # not chosen, agent chose to wait
    }

    self.assertEqual(next_target, None)
    self.assertEqual(self.agent._targets, expected_targets)

  def test_recency_boost(self):
    self.agent._enable_recency_boost = True
    self.agent._wait_probability = 1.0

    nodes = self.mock_state.action_surface

    self.mock_state.step_action_surface_additions = set(nodes[:3])
    self.agent.current_step = 1
    self.agent.get_next_action(self.mock_state)

    self.mock_state.step_action_surface_additions = set(nodes[3:])
    self.agent.current_step = 2
    self.agent.get_next_action(self.mock_state)

    expected_targets = {
        nodes[0].full_name: 5,
        nodes[1].full_name: 6,
        nodes[2].full_name: 3,
        nodes[3].full_name: 5,
        nodes[4].full_name: 3,
        # nodes[5].full_name: 0,  # disabled
        nodes[6].full_name: 3,  # not chosen, agent chose to wait
    }
    self.assertEqual(self.agent._targets, expected_targets)


class TestSoftmax(unittest.TestCase):
  """Tests for agents.softmax()."""

  def test_single_priority(self):
    priorities = [5]
    expected_output = [1.0]
    actual_output = agents.softmax(priorities)
    self.assertEqual(expected_output, actual_output)
    self.assertAlmostEqual(sum(actual_output), 1.0, places=8)

  def test_mixed_priorities(self):
    priorities = [-1, 0, 1]
    expected_output = [0.09003057, 0.24472847, 0.66524096]
    actual_output = agents.softmax(priorities)
    for i in range(len(expected_output)):
      self.assertAlmostEqual(expected_output[i], actual_output[i], places=8)
    self.assertAlmostEqual(sum(actual_output), 1.0, places=8)

  def test_zero_priorities(self):
    priorities = [0, 0, 0]
    expected_output = [1 / 3, 1 / 3, 1 / 3]
    actual_output = agents.softmax(priorities)
    for i in range(len(expected_output)):
      self.assertAlmostEqual(expected_output[i], actual_output[i], places=8)
    self.assertAlmostEqual(sum(actual_output), 1.0, places=8)

  def test_equivalent_priorities(self):
    priorities1 = [-1, 0, 1]
    priorities2 = [0, 1, 2]
    expected_output = [0.09003057, 0.24472847, 0.66524096]
    actual_output1 = agents.softmax(priorities1)
    actual_output2 = agents.softmax(priorities2)

    for i in range(len(expected_output)):
      self.assertAlmostEqual(expected_output[i], actual_output1[i], places=8)
      self.assertAlmostEqual(expected_output[i], actual_output2[i], places=8)

    self.assertAlmostEqual(sum(actual_output1), 1.0, places=8)
    self.assertAlmostEqual(sum(actual_output2), 1.0, places=8)
    self.assertEqual(
        actual_output1,
        actual_output2,
        'The outputs for [-1, 0, 1] and [0, 1, 2] should be the same.',
    )


class TestEventLogger(unittest.TestCase):
  """Test the EventLogger."""

  def setUp(self):
    super().setUp()

    attack_graph = _create_mock_attack_graph(self)

    self.current_target = attack_graph.nodes[6]
    self.event_logger = agents.EventLogger(
        attack_graph, 'TestAgent', 'MALICIOUS'
    )

  def test_subasset_names_property(self):
    expected_subasset_names = {
        'Principal': {'Principal', 'ServiceAccount', 'GoogleAccount'},
        'ServiceAccount': {'ServiceAccount'},
        'GoogleAccount': {'GoogleAccount'},
        'Project': {'Project'},
    }

    self.assertEqual(self.event_logger.subasset_names, expected_subasset_names)

  def test_logs_are_properly_collected(self) -> None:
    """Ensure logs are collected from the current target's detectors."""
    self.event_logger._get_context_assets = mock.Mock()
    self.event_logger._get_context_assets.side_effect = [
        {'project': 'gcp_sec_demo', 'principalEmail': 'ServiceAccount1'},
        {'ip_address': '142.250.4.113'},
    ]

    self.event_logger.collect_logs(self.current_target, 11)
    expected = [
        {
            'timestep': 11,
            '_detector': 'iam.googleapis.com.Create',
            'asset': 'ServiceAccount1',
            'attack_step': 'create',
            'agent': 'TestAgent',
            'agent_profile': 'MALICIOUS',
            'project': 'gcp_sec_demo',
            'principalEmail': 'ServiceAccount1',
        },
        {
            'timestep': 11,
            '_detector': 'some_other_detector',
            'asset': 'ServiceAccount1',
            'attack_step': 'create',
            'agent': 'TestAgent',
            'agent_profile': 'MALICIOUS',
            'ip_address': '142.250.4.113',
        },
    ]

    self.assertEqual(self.event_logger.logs, expected, seq_type=list)

  def test_get_context_assets(self) -> None:
    """Ensure detector context resolves to assets properly."""
    compromised_step11 = self.event_logger.attack_graph.nodes[0]
    compromised_step21 = self.event_logger.attack_graph.nodes[1]

    labeled_assets = self.event_logger._get_context_assets(
        self.current_target.detectors['iam.googleapis.com.Create'],
        [compromised_step11, compromised_step21],
        self.current_target,
    )

    expected = {'project': 'gcp_sec_demo', 'principalEmail': 'ServiceAccount1'}
    self.assertEqual(labeled_assets, expected)

  def test_fail_if_context_cannot_be_satisfied(self) -> None:
    """Do fail if detector context can't be satisfied in reached attack steps."""

    with self.assertRaisesRegex(ValueError, 'cannot be satisfied'):
      self.event_logger._get_context_assets(
          self.current_target.detectors['iam.googleapis.com.Create'],
          [],
          self.current_target,
      )


if __name__ == '__main__':
  unittest.main()
