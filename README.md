# MAL Simulator

## Overview

A MAL compliant simulator.

## Installation
```pip install mal-simulator```

## MalSimulator

A `sims.mal_simulator.MalSimulator` can be created to be able to run simulations.

### MalSimulatorSettings
The constructor of MalSimulator can be given a settings object (`sims.mal_simulator.MalSimulatorSettings`)
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

attacker_agent_class: 'BreadthFirstAttacker' | 'DepthFirstAttacker' | 'KeyboardAgent'

# For defender_agent_class, null and False are treated the same - no defender will be used in the simulation
defender_agent_class: 'BreadthFirstAttacker' | 'DepthFirstAttacker' | 'KeyboardAgent' | null | False


# Optionally add rewards for each attack step
rewards:
  <full name of attack step>: <reward>

  # example:
  # Program 1:notPresent: 3
  # Data A:read: 100
  ...


# Optionally add entry points to AttackGraph with attacker name and attack step full_names.
# NOTE: If attacker entry points defined in both model and scenario,
#       the scenario overrides the ones in the model.
attacker_entry_points:
  <attacker name>:
    - <attack step full name>

  # example:
  # 'Attacker1':
  #   - 'Credentials:6:attemptCredentialsReuse'

# Optionally add observability rules that are applied to AttackGrapNodes
# to make only certain attack steps observable
#
# If 'observable_attack_steps' are set:
# - Nodes that match any rule will be marked as observable
# - Nodes that don't match any rules will be marked as non-observable
# If 'observable_attack_steps' are not set:
# - All nodes will be marked as observable
#
observable_attack_steps:
  by_asset_type:
    <asset_type>:
      - <attack step name>
  by_asset_name:
    <asset_name>:
      - <attack step name>

  # Example:
  #   by_asset_type:
  #     Host:
  #       - access
  #       - authenticate
  #     Data:
  #       - read

  #   by_asset_name:
  #     User:3:
  #       - phishing
  #     ...

```

Note: When defining attackers and entrypoints in a scenario, these override potential attackers in the model.

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
mal_simulator, sim_config = create_simulator_from_scenario(scenario_file)

```
The returned MalSimulator contains the attackgraph created from
the scenario, as well as registered agents. At this point, simulator and sim_config
(which contains the agent classes) can be used for running a simulation
(refer to malsim.cli.run_simulation or wrappers.gym_wrappers to see example of this).


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

This will create an attack using the configuration in the scenarios file, apply the rewards, add the attacker and run the simulation with the attacker.
Currently having more than one attacker in the scenario file will have no effect to how the simulation is run, it will only run the first one as an agent.
