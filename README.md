# MAL Simulator

## Overview

The **MAL Simulator** is a tool designed for running simulations on systems modeled in the **Modeling Attack Language (MAL)**.

It allows you to:
* Simulate the adversarial and defensive actions of intelligent agents (attackers and defenders).
* Analyze system security and evaluate attack/defense strategies.
* Develop machine learning agents (e.g., using Reinforcement Learning) in a safe, dynamic environment.
* Incorporate stochastic modeling through Time-to-Compromise (TTC) distributions.

For in-depth documentation please refer to the official **[Wiki](https://github.com/mal-lang/mal-simulator/wiki)**.

## Contributing

- Use [Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)
- The CI pipeline runs `mypy` and `ruff` for linting and type checking, and PRs will only be merged if pipeline succeeds.

---

## 🛠️ How to Run a Simulation

### 1. Installation

```pip install mal-simulator```

To also get ML dependencies (pettingzoo, gymnasium):

```pip install mal-simulator[ml]```

For additional dev tools:

```pip install mal-simulator[dev]```

### 2. Running a Scenario via CLI

You can execute a simulation using a YAML scenario file, which defines the MAL model, the agents (attackers and defenders), their policies, and their goals.

The basic command structure is:

```malsim <scenario_file> [options]```

#### Example

Execute a simulation defined in ```my_scenario.yml```:

```malsim my_scenario.yml```

#### Common Options:

| Option | Description |
| :--- | :--- |
| `-s, --seed SEED` | Sets the random seed for deterministic simulations. |
| `-t, --ttc-mode MODE` | Sets the Time-to-Compromise mode (e.g., `DISABLED`, `PER_STEP_SAMPLE`). |
| `-g, --send-to-gui` | Sends simulation data to the `malsim-gui` for visualization (requires the GUI to be running). |

### 3. Programmatic Usage

For advanced and customized simulations (e.g., integrating with a custom ML training loop), you can initialize and run the simulator directly in a Python script:

```
from malsim import MalSimulator, run_simulation, Scenario

# 1. Load the scenario configuration
SCENARIO_FILE = "my_scenario.yml"
scenario = Scenario.load_from_file(SCENARIO_FILE)

# 2. Create the simulator instance
sim = MalSimulator.from_scenario(scenario)

# 3. Run the simulation
agent_actions = run_simulation(sim, scenario.agents)

print("Simulation finished. Agent actions:", agent_actions)
```

You can also run your own simulation loop.

```
from malsim import MalSimulator, Scenario

scenario = Scenario.load_from_file("my_scenario.yml")
sim = MalSimulator.from_scenario(scenario)
agent_states = sim.reset()

while not sim.done():
    # ... calculate actions for each agent
    agent_states = sim.step(actions)
```

Read the [Wiki](https://github.com/mal-lang/mal-simulator/wiki) for more ways to run the simulator.
