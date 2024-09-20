from abc import ABC
import functools
import logging
import sys

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import MultiDiscrete, Box, Dict

logger = logging.getLogger(__name__)

def format_table(
        entry_format: str,
        header_entry: list[str],
        entries: list[list[str]],
        reprint_header: int = 0
    ) -> str:
    """
    Format a table according to the parameters specified.

    Arguments:
    entry_format    - The string format for the table
    reprint_header  - How many rows apart to reprint the header. If 0 the
                      header will not be reprinted.
    header_entry    - The entry representing the header of the table
    entries         - The list of entries to format

    Return:
    The formatted table.
    """

    formatted_str = ''
    header = entry_format.format(*header_entry)
    formatted_str += header
    for entry_nr, entry in zip(range(0, len(entries)), entries):
        formatted_str += entry_format.format(*entry)
        if (reprint_header != 0) and ((entry_nr + 1) % reprint_header == 0):
            formatted_str += header
    return formatted_str


class MALEnvBase(ParallelEnv, ABC):
    """
    A base class with methods that are used in both
    MALSimulator and in evaluation environments
    """

    def __init__(self):
        """Set all instance variables to default values
        to give a clear overview and avoid linter warnings"""

        super().__init__()

        # These values should be set by inheriting classes
        self.attack_graph = None
        self.lang_graph = None
        self.model = None

        self._asset_type_to_index = {}
        # TODO: is step name same as full_name?
        self._step_name_to_index = {}
        self._id_to_index = {}

        self._index_to_full_name = []
        self._index_to_id = []
        self._index_to_asset_type = []
        self._index_to_step_name = []

        self._blank_observation = {}

    def create_blank_observation(self):
        # For now, an `object` is an attack step
        num_steps = len(self.attack_graph.nodes)

        observation = {
            "is_observable": num_steps * [1],
            "observed_state": num_steps * [-1],
            "remaining_ttc": num_steps * [0],
            "asset_type": [self._asset_type_to_index[step.asset.type]
                           for step in self.attack_graph.nodes],
            "asset_id": [step.asset.id
                         for step in self.attack_graph.nodes],
            "step_name": [self._step_name_to_index[
                          str(step.asset.type + ":" + step.name)]
                          for step in self.attack_graph.nodes],
        }

        logger.debug(
            f'Create blank observation with {num_steps} attack steps.')

        # Add attack graph edges to observation
        observation["attack_graph_edges"] = [
            [
                self._id_to_index[attack_step.id],
                self._id_to_index[child.id]
            ]
            for attack_step in self.attack_graph.nodes
            for child in attack_step.children
        ]

        # Add reverse attack graph edges for defense steps (required by some
        # defender agent logic)
        for attack_step in self.attack_graph.nodes:
            if attack_step.type == "defense":
                for child in attack_step.children:
                    observation["attack_graph_edges"].append(
                        [
                            self._id_to_index[child.id],
                            self._id_to_index[attack_step.id]
                        ]
                    )

        np_obs = {
            "is_observable": np.array(observation["is_observable"],
                dtype=np.int8),
            "observed_state": np.array(observation["observed_state"],
                dtype=np.int8),
            "remaining_ttc": np.array(observation["remaining_ttc"],
                dtype=np.int64),
            "asset_type": np.array(observation["asset_type"], dtype=np.int64),
            "asset_id": np.array(observation["asset_id"], dtype=np.int64),
            "step_name": np.array(observation["step_name"], dtype=np.int64),
            "attack_graph_edges": np.array(observation["attack_graph_edges"],
                dtype=np.int64),
        }

        return np_obs

    def print_lookup_tables(self):
        """Print all lookup tables of the MAL env object to logger"""

        # Print Index, Attack Step ID and Full Name table
        str_format = "{:<5} {:<15} {:<}\n"
        table = "\n"
        header_entry = ["Index", "Attack Step Id", "Attack Step Full Name"]
        entries = []
        for entry in self._index_to_id:
            entries.append([
                    self._id_to_index[entry],
                    entry,
                    self._index_to_full_name[self._id_to_index[entry]]
                ])
        table += format_table(
            str_format, header_entry, entries, reprint_header = 30
        )
        logger.debug(table)

        # Print index to asset type table
        str_format = "{:<5} {:<}\n"
        table = "\n"
        header_entry = ["Index", "Asset Type"]
        entries = []
        for entry in self._index_to_asset_type:
            entries.append([self._asset_type_to_index[entry], entry])
        table += format_table(
            str_format, header_entry, entries, reprint_header = 30)
        logger.debug(table)

        # Print index to attack step name table
        str_format = "{:<5} {:<}\n"
        table = "\n"
        header_entry = ["Index", "Attack Step Name"]
        entries = []
        for entry in self._index_to_step_name:
            entries.append([self._step_name_to_index[entry], entry])
        table += format_table(
            str_format, header_entry, entries, reprint_header = 30)
        logger.debug(table)


    def format_full_observation(self, observation):
        """
        Return a formatted string of the entire observation. This includes
        sections that will not change over time, these define the structure of
        the attack graph.
        """
        obs_str = '\n'

        str_format = "{:<5} {:<80} {:<6} {:<5} {:<5} {:<5} {:<5} {:<}\n"
        header_entry = [
            "Entry", "Name", "Is_Obs", "State", "RTTC", "Type", "Id", "Step"
        ]

        entries = []
        for entry in range(0, len(observation["observed_state"])):
            entries.append(
                [
                    entry,
                    self._index_to_full_name[entry],
                    observation["is_observable"][entry],
                    observation["observed_state"][entry],
                    observation["remaining_ttc"][entry],
                    observation["asset_type"][entry],
                    observation["asset_id"][entry],
                    observation["step_name"][entry],
                ]
            )
        obs_str += format_table(
            str_format,
            header_entry,
            entries,
            reprint_header = 30
        )

        obs_str += "\nEdges:\n"
        for edge in observation["attack_graph_edges"]:
            obs_str += str(edge) + "\n"

        return obs_str


    def format_obs_var_sec(self,
        observation,
        included_values = [-1, 0, 1]):
        """
        Return a formatted string of the sections of the observation that can
        vary over time.

        Arguments:
        observation     - the observation to format
        included_values - the values to list, any values not present in the list
                          will be filtered out
        """

        str_format = "{:>5} {:>80} {:<5} {:<5} {:<}\n"
        header_entry = ["Id", "Name", "State", "RTTC", "Entry"]
        entries = []
        for entry in range(0, len(observation["observed_state"])):
            if observation["is_observable"][entry] and \
                    observation["observed_state"][entry] in included_values:
                entries.append(
                    [
                        self._index_to_id[entry],
                        self._index_to_full_name[entry],
                        observation["observed_state"][entry],
                        observation["remaining_ttc"][entry],
                        entry
                    ]
                )

        obs_str = format_table(
            str_format,
            header_entry,
            entries,
            reprint_header = 30
        )

        return obs_str

    def _format_info(self, info):
        can_act = "Yes" if info["action_mask"][0][1] > 0 else "No"
        agent_info_str = f"Can act? {can_act}\n"
        for entry in range(0, len(info["action_mask"][1])):
            if info["action_mask"][1][entry] == 1:
                agent_info_str += f"{self._index_to_id[entry]} " \
                    f"{self._index_to_full_name[entry]}\n"
        return agent_info_str

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None):
        # For now, an `object` is an attack step
        num_steps = len(self.attack_graph.nodes)
        num_lang_asset_types = len(self.lang_graph.assets)
        num_lang_attack_steps = len(self.lang_graph.attack_steps)
        num_attack_graph_edges = len(
            self._blank_observation["attack_graph_edges"])
        # TODO is_observable is never set. It will be filled in once the
        # observability of the attack graph is determined.
        return Dict(
            {
                "is_observable": Box(
                    0, 1, shape=(num_steps,), dtype=np.int8
                ),  #  0 for unobservable,
                #  1 for observable
                "observed_state": Box(
                    -1, 1, shape=(num_steps,), dtype=np.int8
                ),  # -1 for unknown,
                #  0 for disabled/not compromised,
                #  1 for enabled/compromised
                "remaining_ttc": Box(
                    0, sys.maxsize, shape=(num_steps,), dtype=np.int64
                ),  # remaining TTC
                "asset_type": Box(
                    0,
                    num_lang_asset_types,
                    shape=(num_steps,),
                    dtype=np.int64,
                ),  # asset type
                "asset_id": Box(
                    0, sys.maxsize, shape=(num_steps,), dtype=np.int64
                ),  # asset id
                "step_name": Box(
                    0,
                    num_lang_attack_steps,
                    shape=(num_steps,),
                    dtype=np.int64,
                ),  # attack/defense step name
                "attack_graph_edges": Box(
                    0,
                    num_steps,
                    shape=(num_attack_graph_edges, 2),
                    dtype=np.int64,
                ),  # edges between attack graph steps
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        num_actions = 2  # two actions: wait or use
        # For now, an `object` is an attack step
        num_steps = len(self.attack_graph.nodes)
        return MultiDiscrete([num_actions, num_steps], dtype=np.int64)

    def _observe_defender(self, defender_agent, observation):
        # TODO We should probably create a separate blank observation for the
        # defenders and just update that with the defense action taken so that
        # we do not have to go through the list of nodes every time. In case
        # we have multiple defenders
        for node in self.attack_graph.nodes:
            index = self._id_to_index[node.id]
            if node.is_enabled_defense():
                observation["observed_state"][index] = 1
            else:
                if node.is_compromised():
                    observation["observed_state"][index] = 1
                else:
                    observation["observed_state"][index] = 0
