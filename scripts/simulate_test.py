import datetime
import enum
import logging
import unittest
from unittest import mock

from absl import flags
from absl.testing import parameterized
from malsim import mal_simulator as malsim
from malsim import scenario
from maltoolbox import attackgraph
from maltoolbox import language
from maltoolbox import model

import agents
import simulate as sim


class TestCreateEnv(unittest.TestCase):
  """Tests for sim.create_env()."""

  def setUp(self) -> None:
    super().setUp()

    self.enterContext(
        mock.patch.object(
            language,
            'LanguageGraph',
            return_value=mock.Mock(spec=language.LanguageGraph),
        )
    )

    self.enterContext(
        mock.patch.object(
            model.Model, '_from_dict', return_value=mock.Mock(spec=model.Model)
        )
    )

    attacker = mock.Mock(spec=attackgraph.Attacker, reached_attack_steps=[])
    attacker.id = 0
    attacker.name = 'Attacker:0'

    mock_attack_graph = mock.Mock(spec=attackgraph.AttackGraph, nodes=[1, 2, 3])
    mock_attack_graph.attackers = {0: attacker}

    self.enterContext(
        mock.patch.object(
            attackgraph, 'AttackGraph', return_value=mock_attack_graph
        )
    )

    mock_mal_simualtor_class = mock.Mock(spec=malsim.MalSimulator)
    mock_env = mock.Mock(
        spec=malsim.MalSimulator, max_iter=None, attack_graph=mock_attack_graph
    )
    mock_mal_simualtor_class.return_value = mock_env
    self.enterContext(
        mock.patch.object(malsim, 'MalSimulator', mock_mal_simualtor_class)
    )

    self.mock_mal_simualtor_class = mock_mal_simualtor_class
    self.mock_env = mock_env

  def test_create_env_with_int_horizon(self):
    """Ensure simulator creation path is there."""
    horizon = 5
    env = sim.create_env(mal_spec={}, kth_model={}, horizon=horizon)

    self.mock_mal_simualtor_class.assert_called_once_with(
        self.mock_env.attack_graph,
        max_iter=horizon,
    )
    self.assertEqual(env, self.mock_env)
    env.register_attacker.assert_called_once_with('attacker', 0)
    env.attack_graph.attach_attackers.assert_called_once()

  def test_create_env_with_float_horizon(self):
    """Ensure simulator creation path is there."""
    horizon = 5.0
    env = sim.create_env(mal_spec={}, kth_model={}, horizon=horizon)

    self.mock_mal_simualtor_class.assert_called_once_with(
        self.mock_env.attack_graph,
        max_iter=horizon * len(self.mock_env.attack_graph.nodes),
    )
    self.assertEqual(env, self.mock_env)
    env.register_attacker.assert_called_once_with('attacker', 0)
    env.attack_graph.attach_attackers.assert_called_once()

  @mock.patch.object(scenario, 'create_scenario_attacker', autospec=True)
  def test_create_env_with_entry_points(
      self, create_scenario_attacker: mock.MagicMock
  ):
    """Check entry points replace model ones."""
    kth_model = {
        'attackers': {
            0: {
                'entry_points': {
                    'ServiceAccount1': {
                        'asset_id': '1',
                        'attack_steps': ['compromise'],
                    },
                    'gcp_sec_demo': {
                        'asset_id': '2',
                        'attack_steps': ['exploit'],
                    },
                },
                'name': 'KTH:Attacker',
            }
        }
    }

    override_entry_points = ['some_asset:step1', 'another_asset:step2']

    new_attacker = mock.Mock(spec=attackgraph.Attacker)
    new_attacker.id = 1
    self.mock_env.attack_graph.attackers[1] = new_attacker

    env = sim.create_env(
        mal_spec={},
        kth_model=kth_model,
        horizon=0,
        entry_points=override_entry_points,
    )

    self.assertEqual(env, self.mock_env)

    create_scenario_attacker.assert_called_once_with(
        self.mock_env.attack_graph, 'KTH:Attacker', override_entry_points
    )

    self.assertEqual(env.attack_graph.attackers, {0: new_attacker})


class TestCalculateStride(parameterized.TestCase):
  """Tests for sim.calculate_stride_size()."""

  def test_value_zero_or_negative(self):
    with self.assertRaisesRegex(
        ValueError, 'total_items must be a positive number'
    ):
      sim.calculate_stride_size(0)
    with self.assertRaisesRegex(
        ValueError, 'total_items must be a positive number'
    ):
      sim.calculate_stride_size(-10)

  @parameterized.parameters(
      (1, 10, 1),
      (50, 2, 50),
      (149, 10, 149),
      # first_digit * (scale_factor ** (order_of_magnitude - 1))
      (150, 10, 1 * (10 ** (2 - 1))),
      (550, 10, 5 * (10 ** (2 - 1))),
      (999, 10, 9 * (10 ** (2 - 1))),
      (1e3, 10, 1 * (10 ** (3 - 1))),
      (1e4, 10, 1 * (10 ** (4 - 1))),
      (1e5, 10, 1 * (10 ** (5 - 1))),
      (1e6, 10, 1 * (10 ** (6 - 1))),
      (1e7, 10, 1 * (10 ** (7 - 1))),
      (7.5e3, 10, 7 * (10 ** (3 - 1))),
      (150, 2, 1 * (2 ** (2 - 1))),
      (3e2, 2, 3 * (2 ** (2 - 1))),
      (876, 2, 8 * (2 ** (2 - 1))),
      (1.6e3, 2, 1 * (2 ** (3 - 1))),
      (2e2, 5, 2 * (5 ** (2 - 1))),
      (6e2, 5, 6 * (5 ** (2 - 1))),
      (1.2e3, 5, 1 * (5 ** (3 - 1))),
  )
  def test_stride_calculation(self, value: int, scale_factor: int, stride: int):
    self.assertEqual(sim.calculate_stride_size(value, scale_factor), stride)


class TestGenerateLogsFunction(unittest.TestCase):
  """Tests for the sim.generate_logs() function."""

  def setUp(self):
    super().setUp()

    self.enterContext(
        mock.patch.object(
            sim,
            'TimerStart',
            autospec=True,
            return_value=mock.Mock(
                GetDuration=mock.Mock(return_value=0.0),
                Stop=mock.Mock(return_value=0.0),
            ),
        )
    )

    self.enterContext(
        mock.patch.object(sim, 'calculate_stride_size', autospec=True)
    )

    mock_logs = ['log-one', 'log-two']
    mock_bfs_agent = mock.Mock(spec=agents.BreadthFirstAttacker)
    mock_dfs_agent = mock.MagicMock(
        spec=agents.DepthFirstAttacker, logs=mock_logs
    )
    mock_dfs_agent.event_logger = mock.Mock(
        spec=agents.EventLogger, logs=mock_logs
    )

    mock_bfs_agent_class = mock.Mock(return_value=mock_bfs_agent)
    mock_dfs_agent_class = mock.Mock(return_value=mock_dfs_agent)

    class MockAgentType(enum.Enum):
      BFS = mock_bfs_agent_class
      DFS = mock_dfs_agent_class

      def __str__(self) -> str:
        return self.name

    self.enterContext(mock.patch.object(agents, 'AgentType', MockAgentType))

    self.mock_agent_profile = mock.Mock()
    self.mock_agent_profile.value = 'MALICIOUS'

    mock_state = mock.Mock(malsim.MalSimAttackerState, return_value=2)
    type(mock_state).terminated = mock.PropertyMock(
        # Terminate the agent on the 6th step
        side_effect=iter(5 * [False] + [False, True] + 99 * [True, True])
    )
    mock_state.truncated = False

    mock_env = mock.Mock(spec=malsim.MalSimulator)
    mock_env.reset.return_value = {'attacker': mock_state}
    mock_env.agent_states = {'attacker': mock_state}
    mock_env.max_iter = 10
    mock_env.attack_graph = mock.Mock(spec=attackgraph.AttackGraph)

    self.mock_logs = mock_logs
    self.mock_bfs_agent = mock_bfs_agent
    self.mock_dfs_agent = mock_dfs_agent
    self.mock_bfs_agent_class = mock_bfs_agent_class
    self.mock_dfs_agent_class = mock_dfs_agent_class
    self.mock_state = mock_state
    self.mock_env = mock_env

  def test_start_step_larger_than_horizon(self):
    with self.assertLogs(level=logging.INFO) as info_logs:
      result = sim.generate_logs(
          env=self.mock_env,
          start_step=15,
          attacker_class=agents.AgentType.DFS,
          wait_probability=0.2,
          seed=None,
          agent_profile=self.mock_agent_profile,
      )

    self.assertEquals(result, [])

    for output in info_logs.output:
      self.assertFalse(output.endswith('Deploying DFS attacker.'))

    self.mock_bfs_agent_class.assert_not_called()

  def test_simulation_with_start_step_and_attacker_given(self):
    with self.assertLogs(level=logging.INFO) as info_logs:
      result = sim.generate_logs(
          env=self.mock_env,
          start_step=5,
          attacker_class=agents.AgentType.DFS,
          wait_probability=0.2,
          seed=None,
          agent_profile=self.mock_agent_profile,
      )

    self.assertEqual(result, self.mock_logs)

    self.mock_dfs_agent.get_next_action.assert_called_once()
    self.assertEqual(self.mock_dfs_agent.current_step, 5)

    expected_info_logs = [
        'INFO:absl:Simulation started.',
        'INFO:absl:Timestep  5/10: Deploying DFS attacker.',
        'INFO:absl: 5/10: Attacker has nothing more to do.',
        'INFO:absl:Simulation completed after: 0.00s',
    ]
    self.assertEqual(info_logs.output, expected_info_logs)

    self.mock_bfs_agent_class.assert_not_called()
    self.mock_dfs_agent_class.assert_called_once()


class TestPostprocesslogsFunction(unittest.TestCase):
  """Tests for sim.post_process_logs()."""

  def test_args_are_added(self):
    logs = [{'a': 1}, {'b': 2, 'c': 3}]
    expected = [{'a': 1, 'd': 4, 'e': 5}, {'b': 2, 'c': 3, 'd': 4, 'e': 5}]

    result = sim.post_process_logs(logs, d=4, e=5)

    self.assertEqual(result, expected)


class TestSaveLogs(unittest.TestCase):
  """Tests for sim.save_logs()."""

  def setUp(self):
    super().setUp()

    mock_strftime = mock.Mock(
        spec=datetime.datetime.strftime, return_value='20250507_144658'
    )
    mock_now = mock.Mock(
        spec=datetime.datetime.now,
        return_value=mock.Mock(spec=datetime.datetime, strftime=mock_strftime),
    )
    self.enterContext(
        mock.patch.object(datetime, 'datetime', autospec=True, now=mock_now)
    )

    self.mock_open = self.enterContext(mock.patch('open', autospec=True))

    self.mock_logs = [
        {
            'timestep': 1,
            'principalEmail': 'email@example.com',
            'resourceName': 'VM1',
            'agent_profile': agents.AgentProfile.USER.value,
            '_detector': 'LIST_OBJECTS',
        },
        {
            'timestep': 3,
            '_detector': 'SET_BUCKET_IAM_POLICY',
        },
    ]


  def test_saving_logs_to_file(self):
    sim.save_logs(
        logs=self.mock_logs,
        output_dir='/test/output/dir',
        seed=123,
        start_step=5,
        model_file='/path/to/test_model.json',
        horizon=10,
        attacker_class='dfs',
        wait_probability=0.5,
    )

    filepath = (
        '/test/output/dir/'
        '20250507_144658_'
        'model_test_model_'
        'seed_123_'
        'horizon_10_'
        'startstep_5_'
        'attacker_dfs_'
        'waitprob_0.5.json'
    )
    self.mock_open.assert_called_once_with(filepath, 'w')


if __name__ == '__main__':
  unittest.main()
