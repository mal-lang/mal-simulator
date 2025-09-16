"""CLI to run simulations in MAL Simulator using scenario files"""

from __future__ import annotations
import argparse
import logging

from . import (
    MalSimulator,
    MalSimulatorSettings,
    run_simulation,
    load_scenario
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def main() -> None:
    """Entrypoint function of the MAL Toolbox CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'scenario_file',
        type=str,
        help="Can be found in https://github.com/mal-lang/malsim-scenarios/"
    )
    parser.add_argument(
        '-o', '--output-attack-graph', type=str,
        help="If set to a path, attack graph will be dumped there",
    )
    parser.add_argument(
        '-s', '--seed', type=int,
        help="If set to a seed, simulator will use it as setting",
    )
    args = parser.parse_args()
    scenario = load_scenario(args.scenario_file)
    sim = MalSimulator.from_scenario(
        scenario, MalSimulatorSettings(seed=args.seed)
    )

    if args.output_attack_graph:
        sim.attack_graph.save_to_file(args.output_attack_graph)

    run_simulation(sim, scenario.agents)


if __name__ == '__main__':
    main()
