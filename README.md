# Overview

A MAL compliant simulator.

# Installation 
```pip install mal-simulator```

# CLI

Use the malsim CLI to run run simulations on scenarios.

## Scenarios

Scenarios consist of MAL language, model, rewards, agent classes and attacker entrypoints,
they are a setup for running a simulation. They can be written down in a yaml file like this:

```
lang_file: <path to .mar-archive>
model_file: <path to json/yml model>

attacker_agent_class: 'BreadthFirstAttacker' | 'DepthFirstAttacker' | 'KeyboardAgent'
defender_agent_class: 'BreadthFirstAttacker' | 'DepthFirstAttacker' | 'KeyboardAgent'

# Rewards for each attack step
rewards:
  <full name of attack step>: <reward>
  <full name of attack step>: <reward>

  # example:
  # Program 1:notPresent: 3
  ...

# Add entry points to AttackGraph with attacker_ids
# and attack step full_names
attacker_entry_points:
  <attacker name>:
    - <attack attack step>

  # example:
  # 'Attacker1':
  #   - 'Credentials:6:attemptCredentialsReuse'
```

## Running a scenario simulation with the CLI

```
usage: malsim [-h] scenario_file

positional arguments:
  scenario_file  Examples can be found in https://github.com/mal-lang/malsim-scenarios/

```

This will create an attack using the configuration in the scenarios file, applying the rewards, adding the attacker and running the simulation with the attacker agents specified.