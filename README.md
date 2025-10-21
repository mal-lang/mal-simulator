# MAL Simulator

## Overview

A MAL compliant simulator.

## Installation

```pip install mal-simulator```

To also get ML dependencies (pettingzoo, gymnasium):

```pip install mal-simulator[ml]```

For additional dev tools:

```pip install mal-simulator[dev]```

## Contributing

- Use [Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)
- The CI pipeline runs `mypy` and `ruff` for linting and type checking, and PRs will only be merged if pipeline succeeds.

## MalSimulator

A `mal_simulator.MalSimulator` can be created to be able to run simulations.

### MalSimulatorSettings
The MalSimulator can be given a settings (`mal_simulator.MalSimulatorSettings`)
through the parameter 'sim_settings'. Giving sim_settings is optional, otherwise default settings are used.

```python

settings = MalSimulatorSettings(
    # Default values
    uncompromise_untraversable_steps: bool = False
    ttc_mode: TTCMode = TTCMode.DISABLED
    seed: Optional[int] = None
    attack_surface_skip_compromised: bool = True
    attack_surface_skip_unviable: bool = True
    attack_surface_skip_unnecessary: bool = True
    run_defense_step_bernoullis: bool = True
    attacker_reward_mode: RewardMode = RewardMode.CUMULATIVE
    defender_reward_mode: RewardMode = RewardMode.CUMULATIVE
)
sim = MalSimulator(attack_graph, sim_settings=settings, ...)

```

## Scenarios

To make it easier to define simulation environment you can use scenarios defined in yml-files, or create `malsim.scenario.Scenario` objects.
Scenarios consist of MAL language, model, rewards, agents and some other values.
they are a setup for running a simulation. This is how the format looks like:

```yml
extends: <path to another scenario file> # optional
lang_file: <path to .mar-archive>
model_file: <path to json/yml model>

# Add agents / entry points to simulator / attack graph
# Possible values for AGENT_CLASS:
# PassiveAgent | DecisionAgent | KeyboardAgent | BreadthFirstAttacker |
# DepthFirstAttacker | DefendCompromisedDefender | DefendFutureCompromisedDefender
agents:
  '<agent_name>':
    type: 'attacker'
    agent_class: <AGENT_CLASS>
    entry_points:
    - 'Credentials:6:attemptCredentialsReuse'
    config:                   # optional
      seed: 1
    goals:                    # optional
      - 'Host A:fullAccess'

  '<agent_name>':
    type: 'defender'
    agent_class: <AGENT_CLASS>


# Optionally add rewards to attack graph nodes.
# Applies reward per attack step (default 0)
rewards:
  by_asset_type:
    <asset_type>:
      <step name>: reward (float)
  by_asset_name:
    <asset_name>:
      <step name>: reward (float)

# Example:
#   by_asset_type:
#     Host:
#       access: 10
#       authenticate: 15
#     Data:
#       read: 1

#   by_asset_name:
#     User_3:
#       phishing: 10
#     ...

# Optionally add observability rules that are applied to AttackGrapNodes
# to make only certain steps observable.
# Note: These do not change the behavior of the simulator.
#       Instead, they just create a dict in the scenario object.
#
# If 'observable_steps' are set:
# - Nodes that match any rule will be marked as observable
# - Nodes that don't match any rules will be marked as non-observable
# If 'observable_steps' are not set:
# - All nodes will be marked as observable
observable_steps:
  by_asset_type:
    <asset_type>:
      - <step name>
  by_asset_name:
    <asset_name>:
      - <step name>

# Optionally add actionability rules that are applied to AttackGrapNodes
# to make only certain steps actionable. Works exactly as observability.
actionable_steps:
  by_asset_type:
    <asset_type>:
      - <step name>
  by_asset_name:
    <asset_name>:
      - <step name>

# Example:
#   by_asset_type:
#     Host:
#       - access
#       - authenticate
#     Data:
#       - read

#   by_asset_name:
#     User_3:
#       - phishing
#     ...

# Optionally add false positive or false negative rates
# Note: Also enable settings in MalSimulatorSettings
false_positive_rates:
  by_asset_type:
    <asset_type>:
      - <step name>
  by_asset_name:
    <asset_name>:
      - <step name>

false_negative_rates:
  by_asset_type:
    <asset_type>:
      - <step name>
  by_asset_name:
    <asset_name>:
      - <step name>

# Examples
# false_positive_rates:
#   by_asset_name:
#     Host:0:
#       access: 0.2
#     Host:1:
#       access: 0.3

# false_negative_rates:
#   by_asset_name:
#     Host:0:
#       access: 0.4
#     Host:1:
#       access: 0.5
#     User:3:
#       compromise: 1.0


```

### Extending Scenarios

Instead of copy pasting an entire scenario file it is possible to extend the scenario and only override
specific values. Use the `extends` key pointing to another scenario. All keys present in the extending
scenario will override the settings in the original (extended) scenario when you load the extending scenario.

### Loading a scenario from a python script

#### Load attack graph and config

If you just want to load a scenario from a file, use `malsim.scenarios.load_scenario`.

```python
from malsim import load_scenario

scenario_file = "scenario.yml"
scenario: Scenario = load_scenario(scenario_file)

# Scenario is a class containing:
class Scenario:
    """Scenarios defines everything needed to run a simulation"""
      lang_file: str,
      agents: dict[str, Any],
      model_dict: Optional[dict[str, Any]] = None,
      model_file: Optional[str] = None,
      rewards: Optional[dict[str, Any]] = None,
      false_positive_rates: Optional[dict[str, Any]] = None,
      false_negative_rates: Optional[dict[str, Any]] = None,
      is_observable: Optional[dict[str, Any]] = None,
      is_actionable: Optional[dict[str, Any]] = None,

```

#### Load simulator and config

If you want to create a simulator from a scenario, use `MalSimulator.from_scenario`.

```python
from malsim import load_scenario, MalSimulator

scenario_file = "scenario.yml"
scenario = load_scenario(scenario_file)
mal_simulator = MalSimulator.from_scenario(scenario)

```
The returned MalSimulator contains the attackgraph created from the scenario, as well as registered agents.
At this point, the simulator and the scenario agents can be used for running a simulation

(use or refer to function `malsim.mal_simulator.run_simulation` to create your own simulation loop).
`run_simulation` will return the paths each agent took during the simulation.

If you want deterministic simulations, give `sim_settings` with a seed to the MalSimulator,
and seed to agents in the scenario files.

```python
from malsim import (
  MalSimulator,
  MalSimulatorSettings,
  run_simulation,
  load_scenario
)

SCENARIO_FILE = "tests/testdata/scenarios/traininglang_scenario.yml"
scenario = load_scenario(SCENARIO_FILE)
mal_simulator = MalSimulator.from_scenario(
  scenario, sim_settings=MalSimulatorSettings(seed=10)
)
run_simulation(mal_simulator, scenario.agents)
```

## Agents

Attacker and defender agents are important for running simulations in the MAL Simulator.

### Attacker agent
`type` - 'attacker'

`entry_points` - Where the agent starts off

`goals` - Optional setting telling where the agent wants to end up. If the goal is fulfilled the simulator will terminate the attacker agent.

`config` - A dictionary given to the Agent class on initialization if running simulations with `malsim.mal_simulator.run_simulation` or CLI.

`agent_class` - Name of the class for the agent used when running simulations with `run_simulation` or CLI, can be left empty or set to PassiveAgent if the agent should not act.

### Defender agent

`type` - 'defender'

`config` - A dictionary given to the Agent class on initialization if running simulations with `malsim.mal_simulator.run_simulation` or CLI.

`agent_class` - Name of the class for the agent used when running simulations with `run_simulation` or CLI, can be left empty or set to PassiveAgent if the agent should not act.

## Rewards

Reward functions are important, especially for ML implementations.
Reward values can be set either in a scenario or by giving `node_rewards` to the MalSimulator.

RewardMode can be either CUMULATIVE or ONE_OFF, this is set through the MalSimulatorSettings given to MalSimulator (`sim_settings`). Default is CUMULATIVE.

By default:

- Attacker is rewarded for compromised nodes
- Defender is penalized for compromised nodes and enabled defenses

If a user wants to implement their own custom reward function, the current recommendation is to:

- Create a class that inherits MalSimulator
- Override methods:
  - `_attacker_step_reward`
  - `_defender_step_reward`

These methods are run to generate a reward every step for each agent (`MalSimAgentState`),
and they are defined like this:

```python
    def _attacker_step_reward(
        self,
        attacker_state: MalSimAttackerState,
        reward_mode: RewardMode,
    ) -> float:

    def _defender_step_reward(
        self,
        defender_state: MalSimDefenderState,
        reward_mode: RewardMode
    ) -> float:
```

## CLI

### Running a scenario simulation with the CLI

```
usage: malsim [-h] [-o OUTPUT_ATTACK_GRAPH] [-s SEED] [-t TTC_MODE] [-g] scenario_file

positional arguments:
  scenario_file         Can be found in https://github.com/mal-lang/malsim-scenarios/

options:
  -h, --help            show this help message and exit
  -o OUTPUT_ATTACK_GRAPH, --output-attack-graph OUTPUT_ATTACK_GRAPH
                        If set to a path, attack graph will be dumped there
  -s SEED, --seed SEED  If set to a seed, simulator will use it as setting
  -t TTC_MODE, --ttc-mode TTC_MODE
                        0: EFFORT_BASED_PER_STEP_SAMPLE 1: PER_STEP_SAMPLE 2: PRE_SAMPLE 3: EXPECTED_VALUE 4:
                        DISABLED
  -g, --send-to-gui     If set, simulator will send actions to malsim-gui
  ```

This will create an attack graph using the configuration in the scenarios file, apply the rewards, register the agents and run the simulation.

## Running the simulator without the CLI

To run a more customized simulator or use wrappers/gym envs, you must write your own simulation loop.

To initialize the MalSimulator you either need a scenario file or an attack graph loaded through some other means.

### Initializing simulator programatically with a scenario file

The regular simulator works with attack graph nodes and keeps track on agents state with those.

```python
from malsim import MalSimulator, run_simulation, load_scenario

logging.basicConfig() # Enable logging

SCENARIO_FILE = "tests/testdata/scenarios/traininglang_scenario.yml"
scenario = load_scenario(SCENARIO_FILE)
sim = MalSimulator.from_scenario(scenario) # Can provide settings here
agent_actions = run_simulation(sim, scenario.agents)

```

## Running the VectorizedEnv (serialized observations)

You can run the vectorized without gymnasium to receive serialized observations.

```python

import logging
from typing import Optional

from malsim import load_scenario, MalSimulator
from malsim.envs import MalSimVectorizedObsEnv

logging.basicConfig() # Enable logging

scenario_file = "tests/testdata/scenarios/traininglang_scenario.yml"
scenario = load_scenario(scenario_file)

# The vectorized obs env is a wrapper that creates serialized observations
# for the simulator, similar to how the old simulator used to work, tailored
# for use in gym envs.
vectorized_env = MalSimVectorizedObsEnv(MalSimulator.from_scenario(scenario))

# Run reset after agents are registered
obs, info = vectorized_env.reset()

# You need to write your own simulator loop:
while not vectorized_env.sim.done():
    actions: dict[str, tuple[int, Optional[int]]] = {}

    for agent in scenario.agents:
        vectorized_agent_info = info[agent['name']] # Contains action mask which can be used
        regular_agent_info = vectorized_env.sim.agent_states[agent['name']] # Also contains action mask
        action = next(iter(regular_agent_info.action_surface))

        if action:
            actions[agent['name']] = (1, vectorized_env.node_to_index(action))
        else:
            actions[agent['name']] = (0, None)

    obs, rew, term, trunc, info = vectorized_env.step(actions)


```

## Running the Gym envs

You can run the gym envs.

```python
import logging

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Dict

from malsim.envs.gym_envs import register_envs

# Enable logging to stdout
logging.basicConfig()

env_name = "MALDefenderEnv"
scenario_file = "tests/testdata/scenarios/traininglang_scenario.yml"
register_envs()
env: gym.Env[Dict, MultiDiscrete] = gym.make(
    env_name,
    scenario_file=scenario_file
)

# info contains serialized action mask
obs, info = env.reset()

# Simulation loop
term = False
while not term:
    # Sample an action from action space
    serialized_action = env.action_space.sample(info['action_mask'])
    obs, rew, term, trunc, info = env.step(serialized_action)
```


## TTCs

TTC (Time to compromise) can be enabled with the `sim_settings` ttc_mode option.
TTCs can be defined in the MAL language for attack steps as probability distributions.
- TTCs for attack steps tells the difficulty to compromise a step in the mal-simulator.

Defense steps can also have TTCs, but they are interpreted differently.
- TTCs for defense steps tells the probability that the defense is enabled at the start of a simulation.
- Always Bernoullis. (Enabled = Bernoulli(1), Disabled=Bernoulli(0))

[Read more about TTCs](https://github.com/mal-lang/malcompiler/wiki/Supported-distribution-functions)

In the MalSimulator, TTCs can be used in different ways (set through malsim settings)

1. EFFORT_BASED_PER_STEP_SAMPLE
  - Run a random trial and compare with the success probability of an attack step after n attempts.
  - Let the agent compromise if the trial succeeds

2. PER_STEP_SAMPLE
  - Sample the distribution for an attack step each time an agent tries to compromise a step
  - Let agent compromise a node if the sampled value is <= 1

3. PRE_SAMPLE
  - Sample the distribution for each attack step at the beginning of a simulation and decrement it every step an agent tries to compromise it
  - Let agent compromise a node if the ttc value is < 1

4. EXPECTED VALUE
  - Set the ttc value of a step to the expected value at the beginning of a simulation and decrement it every step an agent tries to compromise it
  - Let agent compromise a node if the ttc value is < 1

5. DISABLED (default)
  - Don't use TTCs, all attack steps are compromised on the agents first attempt (as long as they are allowed to)

#### Bernoullis in attack steps

If an attack step has a Bernoulli in its TTC, it will be sampled at the start of the simulation.
If the Bernoulli does not succeed, the step will not be compromisable.

This is to match the  https://github.com/mal-lang/malcompiler/wiki/Supported-distribution-functions#bernoulli-behaviour

## GUI (slightly experimental)

It is possible to view simulation runs graphically with the [malsim-gui](https://github.com/mal-lang/malsim-gui). Recommended way to run it is through docker.

When you run simulations in the simulator, set `send_to_api=True` in the Malsimulator init or use the `-g` flag when running the simulator from command line.

This will show the model and the performed actions in the web GUI at http://localhost:8888.
