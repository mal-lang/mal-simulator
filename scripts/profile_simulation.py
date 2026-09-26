"""
CLI to profile simulations in MAL Simulator using scenario files
Run this file with <scenario_file> and it will output a cProfile file.
"""

from __future__ import annotations
import argparse
import logging
import cProfile

from malsim.scenario.scenario import Scenario
from malsim.mal_simulator import MalSimulator, run_simulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def main() -> None:
    """Entrypoint function for profiling simulation with CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'scenario_file',
        type=str,
        help='Can be found in https://github.com/mal-lang/malsim-scenarios/',
    )
    parser.add_argument(
        '--profile_output',
        '-o',
        type=str,
        default='simulation_profile.prof',
        help='File to save profiling results',
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=None,
        help='Maximum number of simulation steps to profile',
    )

    args = parser.parse_args()
    scenario = Scenario.load_from_file(args.scenario_file)
    sim = MalSimulator.from_scenario(scenario)

    # Run the profiler
    profiler = cProfile.Profile()
    profiler.enable()

    run_simulation(sim, max_steps=args.max_steps)

    profiler.disable()

    # Save profiling results
    profiler.dump_stats(args.profile_output)

    print(f'Profiling results saved to {args.profile_output}')


if __name__ == '__main__':
    main()
