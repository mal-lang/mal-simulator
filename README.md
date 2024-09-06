# Overview

A MAL compliant simulator.

# Installation 
```pip install mal-simulator```

# CLI

Use the malsim CLI to run run simulations on scenarios.

## Scenarios

Scenarios consist of MAL language, model, rewards, agent classes and attacker entrypoints,
they are a setup for running a simulation. They can be written down in a yaml file like this:

```yml
lang_file: <path to .mar-archive>
model_file: <path to json/yml model>

attacker_agent_class: 'BreadthFirstAttacker' | 'DepthFirstAttacker' | 'KeyboardAgent'

# For defender_agent_class, null and False are treated the same - no defender will be used in the simulation
defender_agent_class: 'BreadthFirstAttacker' | 'DepthFirstAttacker' | 'KeyboardAgent' | null | False

# Rewards for each attack step
rewards:
  <full name of attack step>: <reward>
  <full name of attack step>: <reward>

  # example:
  # Program 1:notPresent: 3
  ...

# Add entry points to AttackGraph with attacker name and attack step full_names.
# NOTE: If attacker entry points defined in both model and scenario,
#       the scenario overrides the ones in the model.
attacker_entry_points:
  <attacker name>:
    - <attack step full name>

  # example:
  # 'Attacker1':
  #   - 'Credentials:6:attemptCredentialsReuse'
```

Note: When defining attackers and entrypoints in a scenario, these override potential attackers in the model.

## Loading a scenario from a python script

```python
from malsim.scenarios import load_scenario

scenario_file = "scenario.yml"
attack_graph, sim_config = load_scenario(args.scenario_file)

# At this point, the attack graph and sim_config (which contains the agent classes) can be used
# for running a simulation (refer to malsim.cli.run_simulation to see example of this)

```

## Running a scenario simulation with the CLI

```
usage: malsim [-h] scenario_file

positional arguments:
  scenario_file  Examples can be found in https://github.com/mal-lang/malsim-scenarios/

```

This will create an attack using the configuration in the scenarios file, applying the rewards, adding the attacker and running the simulation with the attacker.
Currently having more than one attacker in the scenario file will have no effect to how the simulation is run, it will only run the first one as an agent.
