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
