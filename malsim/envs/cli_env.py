"""CLI to run simulations in MAL Simulator using scenario files"""

from __future__ import annotations
import logging

from ..scenario import create_simulator_from_scenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

class CLIEnv:
    """An environment with a that runs simulations in the CLI"""

    def __init__(
            self,
            scenario_file: str
        ):
        """Create CLIEnv"""
        self.sim, self.agents = \
            create_simulator_from_scenario(scenario_file)

    def run(self):
        """Run a simulation on an attack graph with given config"""

        self.sim.reset()
        total_rewards = {agent_id: 0 for agent_id in self.agents}
        all_agents_waiting = False

        logger.info("Starting CLI env simulator.")

        while not all_agents_waiting:
            actions = {}

            # Select actions for each agent
            for agent_id, agent in self.agents.items():
                if agent is None:
                    logger.warning(
                        'Agent "%s" has no decision agent class '
                        'specified in scenario. Waiting.', agent_id,
                    )
                    continue

                if agent_action := agent.get_next_action():
                    actions[agent_id] = agent_action

                logger.info(
                    'Agent "%s" chose actions: %s', agent_id,
                    [n.full_name for n in agent_action]
                )

            all_agents_waiting = len(actions) == 0

            # Perform next step of simulation
            self.sim.step(actions)
            for agent in self.agents.values():
                total_rewards[agent.name] += agent.reward
            print("---\n")

        logger.info("Game Over.")

        # Print total rewards
        for agent in self.agents.values():
            print(f'Total reward "{agent.name}"', total_rewards[agent.name])
