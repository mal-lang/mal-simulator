"""Module with helper functions to format and log in the MalSimulator"""

def format_full_observation(sim, observation):
    """
    Return a formatted string of the entire observation. This includes
    sections that will not change over time, these define the structure of
    the attack graph.
    """
    obs_str = '\nAttack Graph Steps\n'

    str_format = "{:<5} {:<80} {:<6} {:<5} {:<5} {:<30} {:<8} {:<}\n"
    header_entry = [
        "Entry", "Name", "Is_Obs", "State",
        "RTTC", "Asset Type(Index)", "Asset Id", "Step"
    ]
    entries = []
    for entry in range(0, len(observation["observed_state"])):
        asset_type_index = observation["asset_type"][entry]
        asset_type_str = sim._index_to_asset_type[asset_type_index ] + \
            '(' + str(asset_type_index) + ')'
        entries.append(
            [
                entry,
                sim._index_to_full_name[entry],
                observation["is_observable"][entry],
                observation["observed_state"][entry],
                observation["remaining_ttc"][entry],
                asset_type_str,
                observation["asset_id"][entry],
                observation["step_name"][entry],
            ]
        )
    obs_str += format_table(
        str_format, header_entry, entries, reprint_header = 30
    )

    obs_str += "\nAttack Graph Edges:\n"
    for edge in observation["attack_graph_edges"]:
        obs_str += str(edge) + "\n"

    obs_str += "\nInstance Model Assets:\n"
    str_format = "{:<5} {:<5} {:<}\n"
    header_entry = [
        "Entry", "Id", "Type(Index)"]
    entries = []
    for entry in range(0, len(observation["model_asset_id"])):
        asset_type_str = sim._index_to_asset_type[
            observation["model_asset_type"][entry]] + \
                '(' + str(observation["model_asset_type"][entry]) + ')'
        entries.append(
            [
                entry,
                observation["model_asset_id"][entry],
                asset_type_str
            ]
        )
    obs_str += format_table(
        str_format, header_entry, entries, reprint_header = 30
    )

    obs_str += "\nInstance Model Edges:\n"
    str_format = "{:<5} {:<40} {:<40} {:<}\n"
    header_entry = [
        "Entry",
        "Left Asset(Id/Index)",
        "Right Asset(Id/Index)",
        "Type(Index)"
    ]
    entries = []
    for entry in range(0, len(observation["model_edges_ids"])):
        assoc_type_str = sim._index_to_model_assoc_type[
            observation["model_edges_type"][entry]] + \
                '(' + str(observation["model_edges_type"][entry]) + ')'
        left_asset_index = int(observation["model_edges_ids"][entry][0])
        right_asset_index = int(observation["model_edges_ids"][entry][1])
        left_asset_id = sim._index_to_model_asset_id[left_asset_index]
        right_asset_id = sim._index_to_model_asset_id[right_asset_index]
        left_asset_str = \
            sim.model.get_asset_by_id(left_asset_id).name + \
            '(' + str(left_asset_id) + '/' + str(left_asset_index) + ')'
        right_asset_str = \
            sim.model.get_asset_by_id(right_asset_id).name + \
            '(' + str(right_asset_id) + '/' + str(right_asset_index) + ')'
        entries.append(
            [
                entry,
                left_asset_str,
                right_asset_str,
                assoc_type_str
            ]
        )
    obs_str += format_table(
        str_format, header_entry, entries, reprint_header = 30
    )

    return obs_str

def format_obs_var_sec(
        sim,
        observation,
        included_values = [-1, 0, 1]
    ):
    """
    Return a formatted string of the sections of the observation that can
    vary over time.

    Arguments:
    observation     - the observation to format
    included_values - the values to list, any values not present in the
                        list will be filtered out
    """

    str_format = "{:>5} {:>80} {:<5} {:<5} {:<}\n"
    header_entry = ["Id", "Name", "State", "RTTC", "Entry"]
    entries = []
    for entry in range(0, len(observation["observed_state"])):
        if observation["is_observable"][entry] and \
            observation["observed_state"][entry] in included_values:
            entries.append(
                [
                    sim._index_to_id[entry],
                    sim._index_to_full_name[entry],
                    observation["observed_state"][entry],
                    observation["remaining_ttc"][entry],
                    entry
                ]
            )

    obs_str = format_table(
        str_format, header_entry, entries, reprint_header = 30
    )

    return obs_str

def format_info(sim, info):
    can_act = "Yes" if info["action_mask"][0][1] > 0 else "No"
    agent_info_str = f"Can act? {can_act}\n"
    for entry in range(0, len(info["action_mask"][1])):
        if info["action_mask"][1][entry] == 1:
            agent_info_str += f"{sim._index_to_id[entry]} " \
                f"{sim._index_to_full_name[entry]}\n"
    return agent_info_str


def log_mapping_tables(logger, sim):
    """Log all mapping tables in MalSimulator"""

    str_format = "{:<5} {:<15} {:<}\n"
    table = "\n"
    header_entry = ["Index", "Attack Step Id", "Attack Step Full Name"]
    entries = []
    for entry in sim._index_to_id:
        entries.append(
            [
                sim._id_to_index[entry],
                entry,
                sim._index_to_full_name[sim._id_to_index[entry]]
            ]
        )
    table += format_table(
        str_format,
        header_entry,
        entries,
        reprint_header = 30
    )
    logger.debug(table)

    str_format = "{:<5} {:<}\n"
    table = "\n"
    header_entry = ["Index", "Asset Id"]
    entries = []
    for entry in sim._model_asset_id_to_index:
        entries.append(
            [
                sim._model_asset_id_to_index[entry],
                entry
            ]
        )
    table += format_table(
        str_format,
        header_entry,
        entries,
        reprint_header = 30
    )
    logger.debug(table)

    str_format = "{:<5} {:<}\n"
    table = "\n"
    header_entry = ["Index", "Asset Type"]
    entries = []
    for entry in sim._asset_type_to_index:
        entries.append(
            [
                sim._asset_type_to_index[entry],
                entry
            ]
        )
    table += format_table(
        str_format,
        header_entry,
        entries,
        reprint_header = 30
    )
    logger.debug(table)

    str_format = "{:<5} {:<}\n"
    table = "\n"
    header_entry = ["Index", "Attack Step Name"]
    entries = []
    for entry in sim._index_to_step_name:
        entries.append([sim._step_name_to_index[entry], entry])
    table += format_table(
        str_format,
        header_entry,
        entries,
        reprint_header = 30
    )
    logger.debug(table)

    str_format = "{:<5} {:<}\n"
    table = "\n"
    header_entry = ["Index", "Association Type"]
    entries = []
    for entry in sim._index_to_model_assoc_type:
        entries.append([sim._model_assoc_type_to_index[entry], entry])
    table += format_table(
        str_format,
        header_entry,
        entries,
        reprint_header = 30
    )
    logger.debug(table)


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


def log_agent_state(
        logger, sim, agent, terminations, truncations, infos
    ):
    """Debug log all an agents current state"""

    agent_obs_str = format_obs_var_sec(
        sim, agent.observation, included_values = [0, 1]
    )

    logger.debug(
        'Observation for agent "%s":\n%s', agent.name, agent_obs_str)
    logger.debug(
        'Rewards for agent "%s": %d', agent.name, agent.reward)
    logger.debug(
        'Termination for agent "%s": %s',
        agent.name, terminations[agent.name])
    logger.debug(
        'Truncation for agent "%s": %s',
        agent.name, str(truncations[agent.name]))
    agent_info_str = format_info(sim, infos[agent.name])
    logger.debug(
        'Info for agent "%s":\n%s', agent.name, agent_info_str)
