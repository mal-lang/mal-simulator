import logging
import sys

from maltoolbox.attackgraph import AttackGraphNode

from malsim.agents.agent_base import AgentType
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box, Dict

from .agent_base import MalSimAgent, MalSimAttacker, MalSimDefender
from ..sims.mal_simulator import MalSimulator
from ..sims.mal_sim_logging_utils import format_full_observation, log_mapping_tables

logger = logging.getLogger(__name__)
null_action = []

class SerializedObsAgent(MalSimAgent):
    """An agent that creates obs for a custom observation interface"""

    def __init__(
            self,
            name: str,
            agent_type: AgentType,
            **kwargs
        ):

        super().__init__(name, agent_type, **kwargs)
        self.simulator: MalSimulator = kwargs.get('simulator')

        # List mapping from node/asset index to id/name/type
        self._index_to_id = [n.id for n in self.simulator.attack_graph.nodes]
        self._index_to_full_name = [
            n.full_name for n in self.simulator.attack_graph.nodes]
        self._index_to_asset_type = [
            n.name for n in self.simulator.attack_graph.lang_graph.assets]
        self._index_to_step_name = [
            n.asset.name + ":" + n.name for n
            in self.simulator.attack_graph.lang_graph.attack_steps]
        self._index_to_model_asset_id = [int(asset.id) for asset in \
            self.simulator.attack_graph.model.assets]
        self._index_to_model_assoc_type = [
            assoc.name + '_' + \
            assoc.left_field.asset.name + '_' + \
            assoc.right_field.asset.name \
            for assoc in self.simulator.attack_graph.lang_graph.associations]

        # Lookup dicts attribute to index
        self._id_to_index = {
            n: i for i, n in enumerate(self._index_to_id)}
        self._asset_type_to_index = {
            n: i for i, n in enumerate(self._index_to_asset_type)}
        self._step_name_to_index = {
            n: i for i, n in enumerate(self._index_to_step_name)
        }
        self._model_asset_id_to_index = {
            asset: i for i, asset in enumerate(self._index_to_model_asset_id)
        }
        self._model_assoc_type_to_index = {
            assoc_type: i for i, assoc_type in \
                enumerate(self._index_to_model_assoc_type)
        }

        if logger.isEnabledFor(logging.DEBUG):
            log_mapping_tables(logger, self)

        self.observation = self._create_blank_observation()

    def _create_blank_observation(self, default_obs_state=-1):
        """Create the initial observation"""
        # For now, an `object` is an attack step
        num_steps = len(self.simulator.attack_graph.nodes)

        observation = {
            # If no observability set for node, assume observable.
            "is_observable": [step.extras.get('observable', 1)
                              for step in self.simulator.attack_graph.nodes],
            # Same goes for actionable.
            "is_actionable": [step.extras.get('actionable', 1)
                              for step in self.simulator.attack_graph.nodes],
            "observed_state": num_steps * [default_obs_state],
            "remaining_ttc": num_steps * [0],
            "asset_type": [self._asset_type_to_index[step.asset.type]
                           for step in self.simulator.attack_graph.nodes],
            "asset_id": [step.asset.id
                         for step in self.simulator.attack_graph.nodes],
            "step_name": [self._step_name_to_index[
                          str(step.asset.type + ":" + step.name)]
                          for step in self.simulator.attack_graph.nodes],
        }

        logger.debug(
            'Create blank observation with %d attack steps.', num_steps)

        # Add attack graph edges to observation
        observation["attack_graph_edges"] = [
            [self._id_to_index[attack_step.id], self._id_to_index[child.id]]
                for attack_step in self.simulator.attack_graph.nodes
                    for child in attack_step.children
        ]

        # Add reverse attack graph edges for defense steps (required by some
        # defender agent logic)
        for attack_step in self.simulator.attack_graph.nodes:
            if attack_step.type == "defense":
                for child in attack_step.children:
                    observation["attack_graph_edges"].append(
                        [self._id_to_index[child.id],
                            self._id_to_index[attack_step.id]]
                    )

        # Add instance model assets
        observation["model_asset_id"] = []
        observation["model_asset_type"] = []
        observation["model_edges_ids"] = []
        observation["model_edges_type"] = []

        for asset in self.simulator.attack_graph.model.assets:
            observation["model_asset_id"].append(asset.id)
            observation["model_asset_type"].append(
                self._asset_type_to_index[asset.type])

        for assoc in self.simulator.attack_graph.model.associations:
            left_field_name, right_field_name = \
                self.simulator.attack_graph.model.get_association_field_names(assoc)
            left_field = getattr(assoc, left_field_name)
            right_field = getattr(assoc, right_field_name)
            for left_asset in left_field:
                for right_asset in right_field:
                    observation["model_edges_ids"].append(
                        [
                            self._model_asset_id_to_index[left_asset.id],
                            self._model_asset_id_to_index[right_asset.id]
                        ]
                    )
                    observation["model_edges_type"].append(
                        self._model_assoc_type_to_index[
                            self._get_association_full_name(assoc)])

        np_obs = {
            "is_observable": np.array(observation["is_observable"],
                             dtype=np.int8),
            "is_actionable": np.array(observation["is_actionable"],
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
            "model_asset_id": np.array(observation["model_asset_id"],
                                dtype=np.int64),
            "model_asset_type": np.array(observation["model_asset_type"],
                                dtype=np.int64),
            "model_edges_ids": np.array(observation["model_edges_ids"],
                                dtype=np.int64),
            "model_edges_type": np.array(observation["model_edges_type"],
                                dtype=np.int64)
        }

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                format_full_observation(self, np_obs)
            )

        return np_obs

    def get_attack_graph_node_by_index(self, index: int) -> AttackGraphNode:
        """Get a node from the attack graph by index

        Index is the position of the node in the lookup list.

        First convert index to id and then fetch the node from the
        AttackGraph.

        Raise LookupError if node with given index does not map to a node in
        the attack graph and IndexError if the index is out of range for the
        lookup list.

        Returns:
        Attack graph node matching the id of the index in the lookup list
        """

        if index >= len(self._index_to_id):
            raise IndexError(
                f'Index {index}, is out of range of the '
                f'lookup list which is of length {len(self._index_to_id)}'
            )

        node_id = self._index_to_id[index]
        node = self.simulator.attack_graph.get_node_by_id(node_id)
        if not node:
            raise LookupError(
                f'Index {index} (id: {node_id}), does not map to a node'
            )
        return node

    def _get_association_full_name(self, association) -> str:
        """Get association full name

        TODO: Remove this method once the language graph integration is
        complete in the mal-toolbox because the language graph associations
        will use their full names for the name property

        Arguments:
        association     - the association whose full name will be returned

        Return:
        A string containing the association name and the name of each of the
        two asset types for the left and right fields separated by
        underscores.
        """

        assoc_name = association.__class__.__name__
        if '_' in assoc_name:
            # TODO: Not actually a to-do, but just an extra clarification that
            # this is an ugly hack that will work for now until we get the
            # unique association names. Right now some associations already
            # use the asset types as part of their name if there are multiple
            # associations with the same name.
            return assoc_name

        left_field_name, right_field_name = \
            self.simulator.attack_graph.model.get_association_field_names(association)
        left_field = getattr(association, left_field_name)
        right_field = getattr(association, right_field_name)
        lang_assoc = self.simulator.attack_graph.lang_graph.get_association_by_fields_and_assets(
            left_field_name,
            right_field_name,
            left_field[0].type,
            right_field[0].type
        )
        if lang_assoc is None:
            raise LookupError('Failed to find association for fields '
                '"%s" "%s" and asset types "%s" "%s"!' % (
                    left_field_name,
                    right_field_name,
                    left_field[0].type,
                    right_field[0].type
                )
            )
        assoc_full_name = lang_assoc.name + '_' + \
            lang_assoc.left_field.asset.name + '_' + \
            lang_assoc.right_field.asset.name
        return assoc_full_name

    def _observe_attacker(
            self,
            attacker_agent,
            performed_actions: dict[str, list[int]]
        ) -> None:
        """
        Update the attacker observation based on the actions performed
        in current step.

        Arguments:
        attacker_agent  - the attacker agent to fill in the observation for
        observation     - the blank observation to fill in
        """
        obs_state = attacker_agent.observation.get("observed_state")

        # Set obs state of reached attack steps to 1 (enabled)
        for _, actions in performed_actions.items():
            for step_index in actions:

                if step_index is None:
                    # Waiting does not affect obs
                    continue

                node_id = self._index_to_id[step_index]
                node = self.simulator.attack_graph.get_node_by_id(node_id)
                if node.type in ('or', 'and'):
                    # Attack step activated, set to 1 (enabled)
                    obs_state[step_index] = 1

                    for child in node.children:
                        # Set its children to 0 (disabled)
                        child_index = self._id_to_index[child.id]
                        if obs_state[child_index] == -1:
                            obs_state[child_index] = 0

    def _observe_defender(
            self,
            defender_agent,
            performed_actions: dict[str, list[int]]
        ):

        obs_state = defender_agent.observation["observed_state"]

        if not self.simulator.sim_settings.cumulative_defender_obs:
            # Clear the state if we do not it to accumulate observations over
            # time.
            obs_state.fill(0)

        # Enable the latest steps taken
        for _, actions in performed_actions.items():
            for action in actions:
                obs_state[action] = 1

    def action_space(self, agent=None):
        num_actions = 2  # two actions: wait or use
        # For now, an `object` is an attack step
        num_steps = len(self.simulator.attack_graph.nodes)
        return MultiDiscrete([num_actions, num_steps], dtype=np.int64)

    def observation_space(self, agent=None):
        # For now, an `object` is an attack step
        num_assets = len(self.simulator.attack_graph.model.assets)
        num_steps = len(self.simulator.attack_graph.nodes)
        num_lang_asset_types = len(self.simulator.attack_graph.lang_graph.assets)
        num_lang_attack_steps = len(self.simulator.attack_graph.lang_graph.attack_steps)
        num_lang_association_types = len(self.simulator.attack_graph.lang_graph.associations)
        num_attack_graph_edges = len(self.observation["attack_graph_edges"])
        num_model_edges = len(self.observation["model_edges_ids"])
        return Dict(
            {
                "is_observable": Box(
                    0, 1, shape=(num_steps,), dtype=np.int8
                ),  #  0 for unobservable, 1 for observable
                "is_actionable": Box(
                    0, 1, shape=(num_steps,), dtype=np.int8
                ),  #  0 for non-actionable, 1 for actionable
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
                "model_asset_id": Box(
                    0,
                    num_assets,
                    shape=(num_assets,),
                    dtype=np.int64,
                ),  # instance model asset ids
                "model_asset_type": Box(
                    0,
                    num_lang_asset_types,
                    shape=(num_assets,),
                    dtype=np.int64,
                ),  # instance model asset types
                "model_edges_ids": Box(
                    0,
                    num_assets,
                    shape=(num_model_edges, 2),
                    dtype=np.int64,
                ),  # instance model edge ids
                "model_edges_type": Box(
                    0,
                    num_lang_association_types,
                    shape=(num_model_edges, ),
                    dtype=np.int64,
                ),  # instance model edge types
            }
        )


class SerializedObsAttacker(MalSimAttacker, SerializedObsAgent):
    """An agent that creates obs for a custom observation interface"""

    def __init__(
            self,
            name: str,
            attacker_id: int,
            simulator: MalSimulator,
            **kwargs
        ):
        super().__init__(name, attacker_id, simulator=simulator, **kwargs)

        self.attacker_id = attacker_id
        attacker = simulator.attack_graph.get_attacker_by_id(attacker_id)
        if self.attacker_id is not None and attacker is None:
            raise LookupError(f"Attacker with id: {attacker_id} not found")
        if attacker:
            self.update_obs(attacker.entry_points)

    def __repr__(self):
        return f"{self.__class__.__name__}(simulator={self.simulator}, " \
               f"attacker_id={self.attacker_id})"

    def _observe_enabled_node(self, node: AttackGraphNode):
        """Set enabled node obs state to enabled and
        its children to disabled"""

        # Mark enabled node obs state with 1 (enabled)
        node_index = self._id_to_index[node.id]
        self.observation['observed_state'][node_index] = 1

        # Mark unknown (-1) children node obs states with 0 (disabled)
        for child_node in node.children:
            child_index = self._id_to_index[child_node.id]
            child_obs = self.observation['observed_state'][child_index]
            if child_obs == -1:
                self.observation['observed_state'][child_index] = 0

    def update_obs(self, performed_steps):
        attacker = self.simulator\
            .attack_graph.get_attacker_by_id(self.attacker_id)

        for node in performed_steps:
            if node.is_compromised_by(attacker):
                self._observe_enabled_node(node)


class SerializedObsDefender(MalSimDefender, SerializedObsAgent):
    """An agent that creates obs for a custom observation interface"""

    def __init__(self, name: str, simulator: MalSimulator, **kwargs):
        super().__init__(name, simulator=simulator, **kwargs)
        self.update_obs(
            [n for n in self.simulator.attack_graph.nodes
             if n.is_enabled_defense()]
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(simulator={self.simulator})"

    def update_obs(self, performed_steps):
        for node in performed_steps:
            index = self._id_to_index[node.id]
            self.observation['observed_state'][index] = 1
