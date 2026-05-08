"""Functions and classes to run an attack simulation and generate logs."""

import collections
import datetime
import json
import math
import os
import random
from typing import Any

from absl import logging
from malsim import mal_simulator as sim
from malsim import scenario
from maltoolbox import attackgraph
from maltoolbox import language
from maltoolbox import model as malmodel

import .agents


class Timer:
  """A simple timer class for measuring elapsed time."""

  def __init__(self):
    self._start_time = None
    self._end_time = None

  def start(self) -> 'Timer':
    """Starts the timer."""
    self._start_time = time.perf_counter()
    self._end_time = None # Reset end time on start
    return self

  def get_duration(self) -> float:
    """Returns the elapsed time since start, or current duration if still running."""
    if self._start_time is None:
      return 0.0
    if self._end_time is not None:
      return self._end_time - self._start_time
    return time.perf_counter() - self._start_time

  def stop(self) -> float:
    """Stops the timer and returns the total elapsed time."""
    if self._start_time is None:
      logging.warning("Timer was stopped without being started.")
      return 0.0
    self._end_time = time.perf_counter()
    return self._end_time - self._start_time

def TimerStart() -> Timer:
  return Timer().start()


def load_files(lang_file, model_file: str) -> tuple[dict, dict]:
  """Load MAL spec and model files."""
  mal_spec = language.compiler.MalCompiler().compile(lang_file)
  logging.info("MAL spec loaded")

  with open(model_file, "r") as f:
    kth_model = json.load(f)
  logging.info("KTH instance model loaded")

  return mal_spec, kth_model


def load_files(lang_file, model_file: str) -> tuple[dict, dict]:
  """Load MAL spec and model files."""
  mal_spec = language.compiler.MalCompiler().compile(lang_file)
  logging.info("MAL spec loaded")

  with open(model_file, "r") as f:
    kth_model = json.load(f)
  logging.info("KTH instance model loaded")

  return mal_spec, kth_model


def create_env(
    mal_spec: dict,
    kth_model: dict,
    horizon: float | int,
    entry_points: list[str] | None = None,
) -> sim.MalSimulator:
  """Create the MAL simulator.

  If horizon is an int, it is interpreted as the maximum number of steps to run
  the simulation for. If it is a float, it is multiplied with the number of
  nodes in the created attack graph to derive the maximum number of steps to run
  for.

  Args:
    mal_spec: Dict containing the compiled MAL language.
    kth_model: Dict with the instance model to build the attack graph of.
    horizon: Max number of simulation steps to run as an integer or as ratio
      (float) of the nodes in the built attack graph.
    entry_points: List of attack step names to start the attacker from.

  Returns:
    A MAL Simulator environment.
  """
  lang_graph = language.LanguageGraph(mal_spec)
  logging.info("Langgraph created")

  model = malmodel.Model._from_dict(kth_model, lang_graph)
  logging.info("Model loaded")

  attack_graph = attackgraph.AttackGraph(lang_graph, model)
  logging.info("Attack graph generated")

  attack_graph.attach_attackers()

  if entry_points is not None:
    scenario.create_scenario_attacker(
        attack_graph=attack_graph,
        attacker_name=kth_model["attackers"][0]["name"]
        if kth_model["attackers"]
        else "Attacker:0",
        entry_point_names=entry_points,
    )

    # After the above create_scenario_attacker() function call, the "attackers"
    # list will contain both the original attacker and the newly created one.
    # The former will contain the entry_points from the model file, the latter
    # will contain the custom entry points the user of "create_env()" has
    # specified. Here we keep only the latter attacker.
    attack_graph.attackers[0] = attack_graph.attackers.pop(1)
    attack_graph.attackers[0].id = 0

  env = sim.MalSimulator(
      attack_graph,
      max_iter=horizon
      if isinstance(horizon, int)
      else round(horizon * len(attack_graph.nodes)),
  )

  env.register_attacker("attacker", 0)

  logging.info("Simulator created")

  return env


def _get_agent_config(
    attacker_class: agents.AgentType,
    agent_profile: agents.AgentProfile,
    env: sim.MalSimulator,
    wait_probability: float,
    seed: int,
) -> agents.SearcherAgentConfig | agents.QuasiRealisticAgentConfig:
  """Return an agent config compatible with the attacker_class."""
  if attacker_class in [agents.AgentType.BFS, agents.AgentType.DFS]:
    return {
        "profile": agent_profile,
        "attack_graph": env.attack_graph,
        "wait_probability": wait_probability,
        "randomize": True,
        "seed": seed,
    }

  if attacker_class in [agents.AgentType.QR_USER, agents.AgentType.QR_ATTACKER]:
    return {
        "profile": agent_profile,
        "attack_graph": env.attack_graph,
        "wait_probability": wait_probability,
        "priorities": {},
    }

  raise ValueError()


def generate_logs(
    env: sim.MalSimulator,
    start_step: int,
    attacker_class: agents.AgentType,
    wait_probability: float,
    seed: int | None,
    agent_profile: agents.AgentProfile,
) -> list[dict]:
  """Run the MAL Simulator."""

  state = env.reset()["attacker"]
  logging.info("Simulation started.")

  stopwatch = TimerStart()

  stride = calculate_stride_size(env.max_iter)

  attacker: agents.LogEmittingAgents = None

  reported = False  # flag for logging agent termination

  for timestep in range(env.max_iter):
    stepstamp = "{:>{}}/{}".format(
        timestep, len(str(env.max_iter)), env.max_iter
    )

    if timestep % stride == 0:
      logging.info(
          "Timestep %s. Time: %2.fs.", stepstamp, stopwatch.GetDuration()
      )

    if timestep == start_step:
      logging.info(
          "Timestep %s: Deploying %s attacker.", stepstamp, attacker_class
      )
      attacker = attacker_class.value(
          _get_agent_config(
              attacker_class, agent_profile, env, wait_probability, seed
          )
      )

    actions = {"attacker": []}  # do nothing by default
    if timestep >= start_step and not state.terminated:
      attacker.current_step = timestep
      if next_action := attacker.get_next_action(state):
        actions = {"attacker": [next_action]}

    env.step(actions)
    state = env.agent_states["attacker"]

    if state.terminated and not reported:
      reported = True
      logging.info("%s: Attacker has nothing more to do.", stepstamp)

  logging.info("Simulation completed after: %.2fs", stopwatch.Stop())

  return attacker.event_logger.logs if attacker else []


def calculate_stride_size(total_items: int, scaling_factor: int = 10) -> int:
  """Calculates stride size using order of magnitude & first digit of total_items.

  Stride is the block size that will be used to split total_items. It can be
  used as the period of a repeating action happening on a `total_items`-number
  of items. For example, if there are 1000 items (i.e. total_items=1000),
  a stride of 150 means that the items should be split in buckets of 150 items
  each or that an action should be performed on every 150th item.

  If total_items is too small (less than 150) sets the stride to total_items,
  i.e. there is no repetition.

  Args:
    total_items: The input integer value.
    scaling_factor: Scales the stride size. Higher values result in larger
      strides.

  Returns:
    The calculated stride size.
  """
  if total_items <= 0:
    msg = "total_items must be a positive number"
    logging.error(msg)
    raise ValueError(msg)
  if total_items < 150:
    stride_size = total_items
  else:
    order_of_magnitude = math.floor(math.log10(total_items))
    first_digit = int(str(total_items)[0])
    base = scaling_factor ** max(0, order_of_magnitude - 1)
    multiplier = max(1, first_digit)
    stride_size = math.floor(base * multiplier)

  return min(stride_size, total_items)


def post_process_logs(logs: list[dict], **kwargs: Any) -> list[dict]:
  """Add extra keys to the passed in synthetic logs.

  Args:
    logs: List of dictionaries containing each logs data.
    **kwargs: Arbitrary arguments to populate each log with.

  Returns:
    The list of updated logs.
  """
  for log in logs:
    log.update(kwargs)

  return logs


def save_logs(
    logs: list[dict],
    output_dir: str,
    seed: int | None,
    start_step: int,
    model_file: str,
    horizon: float | int,
    attacker_class: agents.AgentType,
    wait_probability: float,
) -> None:
  """Save generated logs to a file under output_dir."""
  logging.info("Saving generated logs")

  timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
      "%Y%m%d_%H%M%S"
  )

  log_output_file = (
      f"{output_dir}/"
      f"{timestamp_utc}_"
      f"model_{os.path.basename(model_file).split('.')[0]}_"
      f"seed_{seed}_"
      f"horizon_{horizon}_"
      f"startstep_{start_step}_"
      f"attacker_{attacker_class}_"
      f"waitprob_{wait_probability}"
  )

  log_output_file += ".json"
  logging.info("Dumping logs to %s", log_output_file)
  with open(log_output_file, "w") as f:
    json.dump(logs, f, indent=2)


def run(
    lang_file: str,
    model_file: str,
    horizon: float | int,
    attacker_class: agents.AgentType,
    entry_points: list[str],
    wait_probability: float,
    seed: int | None = None,
    start_step: int | None = None,
    output_dir: str | None = None,
    agent_profile: agents.AgentProfile = agents.AgentProfile.MALICIOUS,
) -> list[dict]:
  """Run a full simulation and generate synthetic logs."""
  if seed is not None:
    random.seed(seed)

  mal_spec, kth_model = load_files(lang_file, model_file)
  env = create_env(mal_spec, kth_model, horizon, entry_points)

  if start_step is None:
    start_step = random.randint(0, env.max_iter // 2)

  logs = generate_logs(
      env, start_step, attacker_class, wait_probability, seed, agent_profile
  )

  if output_dir:
    save_logs(
        logs,
        output_dir=output_dir,
        seed=seed,
        start_step=start_step,
        model_file=model_file,
        horizon=horizon,
        attacker_class=attacker_class,
        wait_probability=wait_probability,
    )

  return logs
