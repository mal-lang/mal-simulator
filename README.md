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
    enable_false_positives: bool = False
    enable_false_negatives: bool = False
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

### Bernoullis in attack steps

If an attack step has a Bernoulli in its TTC, it will be sampled at the start of the simulation.
If the Bernoulli does not succeed, the step will not be compromisable.

This is to match the  https://github.com/mal-lang/malcompiler/wiki/Supported-distribution-functions#bernoulli-behaviour

## Node properties in the MAL Simulator

To implement the MAL logic, a few additional properties have been added to nodes in the MAL simulator.

### Viability

We want to determine which attack steps are possible to compromise in the attack graph with respect to the state of `defense`/`exist`/`notExist` steps. We call that property viability. To get the viability of an attack step, we look at its parents' viability.
Some attack steps have `defense`/`exist`/`notExist` steps as parents. The viability of those steps is determined by whether they are enabled/have their conditions met or not.
In this way the viability propagates from `defense`/`exist`/`notExist` steps to attack steps and indicates whether the attack steps are viable based on current `defense`/`exists`/`notExist` statuses.

[See implementation](https://github.com/mal-lang/mal-simulator/blob/0144efd78d78b606ab25c74a73675e00dafd4887/malsim/graph_processing.py#L147)

Attack steps:
- Viability on attack steps represents whether it is possible to compromise with respect to the state of the attack graphs `defense`/`exist`/`notExist` steps.
  - An `AND`-step is viable if all of its (necessary) parents are viable. Otherwise it is unviable.
  - An `OR`-step is viable if any of its parents are viable. Otherwise it is unviable.
  - An attack step with an uncertain TTC (i.e. it includes a Bernoulli distribution) is unviable if its Bernoulli sampling 'fails' (optional with setting `run_attack_step_bernoullis`).

- Viability on defense steps represents whether they are enabled or not and is used to propagate viability to their child attack steps.
  - A disabled `defense` step is viable (which means it is not making any of its children unviable)
  - An enabled `defense` step is unviable (which means it makes its AND-children unviable)
  - A `defense` step can be pre-enabled based on its TTC (optional with setting `run_defense_step_bernoullis`)

- Viability on `exist`/`notExist` steps is also used to propagate viability to its children.
  - if an `exist` step has any of its requirements met it is viable. It will be unviable if
none of its requirements are present.
  - if a `notExist` step has any of its requirements met it is not viable. It will be viable
if none of its requirements are present.


### Necessity

We want to know whether an attacker needs to compromise an (`OR`/`AND`) attack step to progress (to its children). This concept is called necessity.

[See implementation](https://github.com/mal-lang/mal-simulator/blob/0144efd78d78b606ab25c74a73675e00dafd4887/malsim/graph_processing.py#L52)

Why are not all nodes necessary?

**Answer**: To allow structures in the attack graph where steps are required in some but not all conditions.

**Example**: If you have encrypted data, you must first decrypt them in order to access them. However, if they’re not encrypted you can access the data directly, meaning that the decryption step is unnecessary

Attack steps:
  - An `OR`-step with all of parents necessary is necessary. If not all parents are necessary, the `OR`-node is unnecessary.
  - An `AND`-step with any parent necessary is necessary. If no parents are necessary the `AND`-node is unnecessary.

`defense` steps:
  - Enabled -> Necessary, Disabled -> Unnecessary

`exist` steps:
  - Necessary if it has none of its requirements met, otherwise unnecessary.

`notExist` steps:
  - Necessary if it has any of its requirements met, otherwise unnecessary.

Note: You can decide to ignore these rules effect on attack surfaces with settings:

- `attack_surface_skip_unnecessary`

### Compromised

A node becomes compromised by an attacker if:
1. The node is set as an entrypoint for the attacker.
2. The node is traversable for an attacker and the attacker decides to compromise it.

If TTCs are enabled, the compromise might require several attempts.

### Traversability

[Implementation](https://github.com/mal-lang/mal-simulator/blob/0144efd78d78b606ab25c74a73675e00dafd4887/malsim/mal_simulator.py#L417)

Traversability is a per-attacker node property. Based on an attackers previously compromised nodes, an additional node is traversable iff:
1. The node is viable
2. If node is type OR, at least one of its parents must be reached
3. If node is type AND, all of its necessary parents must be reached

If these are not true, the node is not traversable.

### Actionability / Observability

Actionability/observability are two additional options a node can have, and can be set in a scenario.

They do not affect viability or necessity, but act as a filter.

Observability means that a node is observed by a defender when it is compromised. Being observed in this case means it is added to the `MalSimDefenderState.observed_nodes`.

Actionability currently has no impact in the base simulator, but is used in Vejde MALSIM it controls whether a certain type of attack step can be used as an action by agents or not.


## GUI (slightly experimental)

It is possible to view simulation runs graphically with the [malsim-gui](https://github.com/mal-lang/malsim-gui). Recommended way to run it is through docker.

When you run simulations in the simulator, set `send_to_api=True` in the Malsimulator init or use the `-g` flag when running the simulator from command line.

This will show the model and the performed actions in the web GUI at http://localhost:8888.
