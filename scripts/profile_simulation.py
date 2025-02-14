"""
CLI to profile simulations in MAL Simulator using scenario files
Run this file with <scenario_file> and it will output a cProfile file.
"""

from __future__ import annotations
import argparse
import logging
import cProfile
import pstats

from malsim.scenario import create_simulator_from_scenario
from malsim.cli import run_simulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def main():
    """Entrypoint function for profiling simulation with CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'scenario_file',
        type=str,
        help="Can be found in https://github.com/mal-lang/malsim-scenarios/"
    )
    parser.add_argument(
        '--profile_output',
        type=str,
        default='simulation_profile.prof',
        help="File to save profiling results"
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=100,
        help="MAX iterations in simulator"
    )

    args = parser.parse_args()
    sim, agents = create_simulator_from_scenario(
        args.scenario_file, max_iter=args.max_iter
    )

    # Run the profiler
    profiler = cProfile.Profile()
    profiler.enable()

    run_simulation(sim, agents)

    profiler.disable()

    # Save profiling results
    with open(args.profile_output, 'w', encoding='utf-8') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs().sort_stats('cumulative').print_stats()

    print(f"Profiling results saved to {args.profile_output}")


if __name__ == '__main__':
    main()
