# MAL Simulator

## Overview

A MAL compliant simulator.

## Installation

```pip install mal-simulator```

To also get ML dependencies (numpy, pettingzoo, gymnasium):

```pip install mal-simulator[ml]```

For additional dev tools:

```pip install mal-simulator[dev]```

## MalSimulator

A `mal_simulator.MalSimulator` can be created to be able to run simulations.

### MalSimulatorSettings
The constructor of MalSimulator can be given a settings object (`mal_simulator.MalSimulatorSettings`)
through the parameter 'sim_settings'. Giving sim_settings is optional, otherwise default settings are used.

```python

settings = MalSimulatorSettings(
  uncompromise_untraversable_steps=True, # default is False
  cumulative_defender_obs=False # default is True
)
sim = MalSimulator(lang_graph, model, attack_graph, sim_settings=settings)

```

## Scenarios

To make it easier to define simulation environment you can use scenarios defined in yml-files.
Scenarios consist of MAL language, model, rewards, agent classes and attacker entrypoints,
they are a setup for running a simulation. This is how the format looks like:

```yml
lang_file: <path to .mar-archive>
model_file: <path to json/yml model>

# Add agents / entry points to simulator / attack graph
# Note: When defining attackers and entrypoints in a scenario,
#       these override attackers in the model.
# Possible values for AGENT_CLASS:
# PassiveAgent | DecisionAgent | KeyboardAgent | BreadthFirstAttacker |
# DepthFirstAttacker | DefendCompromisedDefender | DefendFutureCompromisedDefender
agents:
  '<agent_name>':
    type: 'attacker'
    agent_class: <AGENT_CLASS>
    entry_points:
    - 'Credentials:6:attemptCredentialsReuse'

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
#       Instead, they just add the value to each nodes '.extras' field.
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
# to make only certain steps actionable
# Works exactly as observability
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


# Optionally add false positive/negative rates to observations.
#
# False positive/negative `rate` is a number between 0.0 and 1.0.
#  - A false positive rate of x for means that an inactive attack step
#    will be observed as active at a rate of x in each observation
#  - A false negative rate of x for means that an active attack step
#    will be observed as inactive at a rate of x in each observation
# Default false positive/negative rate is 0, which is assumed if none are given.

# Applies false positive rates per attack step (default 0)
false_positive_rates:
  by_asset_type:
    <asset_type>:
      <step name>: rate (float)
  by_asset_name:
    <asset_name>:
      <step name>: rate (float)

# Applies false negative rates per attack step (default 0)
false_negative_rates:
  by_asset_type:
    <asset_type>:
      <step name>: rate (float)
  by_asset_name:
    <asset_name>:
      <step name>: rate (float)

# Example:
#   by_asset_type:
#     Host:
#       access: 0.1
#       authenticate: 0.4
#     Data:
#       read: 0.1

#   by_asset_name:
#     User_3:
#       phishing: 0.3
#     ...

```

### Loading a scenario from a python script

#### Load attack graph and config

If you just want to load a resulting attack graph from a scenario, use `malsim.scenarios.load_scenario`.

```python
from malsim.scenarios import load_scenario

scenario_file = "scenario.yml"
attack_graph, sim_config = load_scenario(scenario_file)

```

#### Load simulator and config

If you instead want to load a simulator, use `malsim.scenarios.create_simulator_from_scenario`.

```python
from malsim.scenarios import create_simulator_from_scenario

scenario_file = "scenario.yml"
mal_simulator, agents = create_simulator_from_scenario(scenario_file)

```
The returned MalSimulator contains the attackgraph created from
the scenario, as well as registered agents. At this point, simulator and sim_config
(which contains the decision agents) can be used for running a simulation
(refer to `malsim.__main__.run_simulation` to see example of this).


## CLI

### Running a scenario simulation with the CLI

```
usage: malsim [-h] [-o OUTPUT_ATTACK_GRAPH] scenario_file

positional arguments:
  scenario_file         Can be found in https://github.com/mal-lang/malsim-scenarios/

options:
  -h, --help            show this help message and exit
  -o OUTPUT_ATTACK_GRAPH, --output-attack-graph OUTPUT_ATTACK_GRAPH
                        If set to a path, attack graph will be dumped there
```

This will create an attack graph using the configuration in the scenarios file, apply the rewards, add the attacker and run the simulation with the attacker.
Currently having more than one attacker in the scenario file will have no effect to how the simulation is run, it will only run the first one as an agent.

## Running the simulator without the CLI

To run a more customized simulator or use wrappers/gym envs, you must write your own simulation loop.

To initialize the MalSimulator you either need a scenario file or an attack graph loaded through some other means.

### Initializing simulator programatically with a scenario file

The regular simulator works with attack graph nodes and keeps track on agents state with those.

```python
import logging

from malsim.scenario import create_simulator_from_scenario
from malsim.envs import MalSimVectorizedObsEnv
from malsim import MalSimulator

logging.basicConfig() # Enable logging

scenario_file = "tests/testdata/scenarios/traininglang_scenario.yml"
sim, agents = create_simulator_from_scenario(scenario_file)

# `sim` is the actual MALSimulator
assert isinstance(sim, MalSimulator)

# `agents` is a list of the scenario agents which are
# automatically registered when you use `create_simulator_from_scenario``
assert isinstance(agents, list)

agent_states = sim.reset()

# `agent_states` is a dict of agent names mapping to agent states
# agent states contain info about the agents current state
assert isinstance(agent_states, dict)

# You can run simulations with the MalSimulator,
# but you need to write a simulation loop:

# Termination condition for our simulation loop
all_agents_term_or_trunc = False

i = 1
while not all_agents_term_or_trunc:
    all_agents_term_or_trunc = True
    actions = {}

    # Select actions for each agent
    for agent_dict in agents:
        agent_name = agent_dict['name']
        # Generate actions - empty list is none action
        # In this case we just pick the first action from the action surface
        action = next(iter(agent_states[agent_name].action_surface))
        actions[agent_dict['name']] = [action] if action else []

    # Perform next step of simulation
    agent_states = sim.step(actions)

    for agent_dict in agents:
        agent_state = agent_states[agent_dict['name']]
        if not agent_state.terminated and not agent_state.truncated:
            all_agents_term_or_trunc = False

    print("---\n")
    i += 1

print("Game Over.")
```

## Running the VectorizedEnv (serialized observations)

You can run the vectorized without gymnasium to receive serialized observations.

```python

import logging
from typing import Optional

from malsim.scenario import load_scenario
from malsim.envs import MalSimVectorizedObsEnv
from malsim.mal_simulator import MalSimulator, AgentType

logging.basicConfig() # Enable logging

scenario_file = "tests/testdata/scenarios/traininglang_scenario.yml"
attack_graph, agents = load_scenario(scenario_file)

# The vectorized obs env is a wrapper that creates serialized observations
# for the simulator, similar to how the old simulator used to work, tailored
# for use in gym envs.
vectorized_env = MalSimVectorizedObsEnv(MalSimulator(attack_graph))

# You need to register the agents manually.
for agent in agents:
    if agent['type'] == AgentType.ATTACKER:
        vectorized_env.register_attacker(agent['name'], agent['attacker_id'])
    elif agent['type'] == AgentType.DEFENDER:
        vectorized_env.register_defender(agent['name'])

# Run reset after agents are registered
obs, info = vectorized_env.reset()

# You need to write your own simulator loop:
done = False
while not done:
    actions: dict[str, tuple[int, Optional[int]]] = {}

    for agent in agents:
        vectorized_agent_info = info[agent['name']] # Contains action mask which can be used
        regular_agent_info = vectorized_env.sim.agent_states[agent['name']] # Also contains action mask
        action = next(iter(regular_agent_info.action_surface))

        if action:
            actions[agent['name']] = (1, vectorized_env.node_to_index(action))
        else:
            actions[agent['name']] = (0, None)

    obs, rew, term, trunc, info = vectorized_env.step(actions)

    for agent in agents:
        done = all(term.values()) or all(trunc.values())

```

## Running the Gym envs

You can run the gym envs.

```python
import logging

from malsim.envs.gym_envs import register_envs
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Dict

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
    # Action selection should not be handled like this is you
    # are training an ML agent naturally
    defender_name = env.unwrapped.defender_agent_name
    agent_info = env.unwrapped.sim.get_agent_state(defender_name)
    action_node = next(iter(agent_info.action_surface))

    # This is to translate a node to an index
    serialized_action = (0, None)
    if action_node:
        serialized_action = (1, env.unwrapped.sim.node_to_index(action_node))

    obs, rew, term, trunc, info = env.step(serialized_action)
```