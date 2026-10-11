from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any
from collections.abc import Callable, Iterable, Mapping, Set
import numpy as np
from numpy.random import default_rng

from maltoolbox.attackgraph import AttackGraph, AttackGraphNode

from malsim import _native
from malsim.config.agent_settings import defender_settings
from malsim.config.agent_settings import attacker_settings
from malsim.mal_simulator.agent_states import (
    AgentStates,
    attacker_states,
    defender_states,
)
from malsim.mal_simulator.attacker_state import AttackerState
from malsim.mal_simulator.attacker_state_factories import (
    create_attacker_state_from_native,
    get_entry_points,
)
from malsim.mal_simulator.defender_state import DefenderState
from malsim.mal_simulator.defender_state_factories import (
    create_defender_state_from_native,
)
from malsim.mal_simulator.native_settings import (
    flatten_attacker_settings,
    flatten_defender_settings,
    flatten_sim_settings,
)
from malsim.mal_simulator.node_getters import (
    full_names_or_nodes_to_nodes,
)

from malsim.mal_simulator.rewards import (
    attacker_step_reward_fn,
    defender_step_reward_fn,
)
from malsim.config.agent_settings import (
    AgentSettings,
    AttackerSettings,
    DefenderSettings,
)
from malsim.mal_simulator.simulator import (
    MALSimulatorStaticData,
    _graph_state_from_native,
    _ordered_new_nodes,
    _pre_step_check,
    alive_agents,
)
from malsim.types import (
    Recording,
)
from malsim.scenario.scenario import Scenario
from malsim.dyna_mal_simulator.simulator_state import (
    AssetOp,
    AssocOp,
    DynaMalSimulatorState,
    create_simulator_state,
    modification_record_from_native,
    update_simulator_state,
)
from malsim.config.sim_settings import MalSimulatorSettings, RewardMode
from malsim.visualization.malsim_gui_client import MalSimGUIClient
from malsim import MalSimulator

logger = logging.getLogger(__name__)

PERFORMED_ATTACKS_FUNCS: Mapping[
    RewardMode,
    Callable[[AttackerState], Set[AttackGraphNode]],
] = {
    RewardMode.CUMULATIVE: lambda ds: ds.performed_nodes,
    RewardMode.ONE_OFF: lambda ds: ds.step_performed_nodes,
    RewardMode.EXPECTED_TTC: lambda ds: ds.step_performed_nodes,
    RewardMode.SAMPLE_TTC: lambda ds: ds.step_performed_nodes,
}

ENABLED_DEFENSES_FUNCS: Mapping[
    RewardMode, Callable[[DefenderState], Set[AttackGraphNode]]
] = {
    # all enabled defenses
    RewardMode.CUMULATIVE: lambda ds: ds.performed_nodes,
    # only newly enabled defenses
    # (this means that the reward actually be defined
    # as a function of the state+action
    #  but whatever)
    RewardMode.ONE_OFF: lambda ds: ds.step_performed_nodes,
}

ENABLED_ATTACKS_FUNCS: Mapping[
    RewardMode, Callable[[DefenderState], Set[AttackGraphNode]]
] = {
    # all performed attacks
    RewardMode.CUMULATIVE: lambda ds: ds.compromised_nodes,
    # only newly performed attacks
    RewardMode.ONE_OFF: lambda ds: ds.step_compromised_nodes,
}

BASE_SETTINGS = MalSimulatorSettings()


class DynaMalSimulator(MalSimulator):
    """A MAL Simulator that works on the AttackGraph

    Allows user to register agents (defender and attacker)
    and lets the agents perform actions step by step and updates
    the state of the attack graph based on the steps chosen.
    """

    def __init__(
        self,
        attack_graph: AttackGraph,
        agents: Iterable[AttackerSettings[AttackGraphNode | str] | DefenderSettings],
        sim_settings: MalSimulatorSettings = BASE_SETTINGS,
        send_to_api: bool = False,
    ):
        """
        Args:
            attack_graph           - The attack graph to use
            sim_settings           - Settings for simulator
            agent_settings         - The agents to pre-register
            rewards                - Global rewards per node
            false_positive_rates   - global fpr per node
            false_negative_rates   - global fnr per node
            node_actionabilities   - global actionabilities per node
            node_observabilities   - global obserabilities per node
            send_to_api            - Enable to send data to malsim-gui rest api
        """
        rng = default_rng(sim_settings.seed)
        rest_api_client = MalSimGUIClient() if send_to_api else None

        if attack_graph.model is None:
            raise ValueError(
                'AttackGraph model is not set. DynaMalSimulator requires a model '
                'to be set.'
            )

        native_sim = _native.DynaSimulator(attack_graph, attack_graph.model)

        static_sim_data = MALSimulatorStaticData(
            attack_graph,
            sim_settings,
        )

        _attacker_settings = [a for a in agents if isinstance(a, AttackerSettings)]
        _defender_settings = [a for a in agents if isinstance(a, DefenderSettings)]

        # Entry points/goals by full name, captured while the graph is still
        # pristine: a node removed by a model effect can't be read any more,
        # and `dyna_reset` re-resolves these against the restored graph.
        attacker_settings_by_full_name = {
            a.name: _attacker_settings_with_full_names(a) for a in _attacker_settings
        }
        attacker_settings_with_nodes = [
            a.convert_to_attack_graph_nodes(attack_graph)
            for a in attacker_settings_by_full_name.values()
        ]

        _agent_settings: AgentSettings = {
            a.name: a for a in (_defender_settings + attacker_settings_with_nodes)
        } or {}

        agent_states, sim_state, recording, _agent_settings = dyna_reset(
            static_sim_data,
            _agent_settings,
            attacker_settings_by_full_name,
            rng,
            rest_api_client,
            native_sim,
        )

        defender_reward_fns = {
            agent_id: defender_step_reward_fn(
                ENABLED_DEFENSES_FUNCS[agent_settings.reward_mode],
                ENABLED_ATTACKS_FUNCS[agent_settings.reward_mode],
                agent_settings,
            )
            for agent_id, agent_settings in defender_settings(_agent_settings).items()
        }
        attacker_reward_fns = {
            agent_id: attacker_step_reward_fn(
                PERFORMED_ATTACKS_FUNCS[agent_setting.reward_mode],
                sim_settings.ttc_mode,
                agent_setting,
                rng,
            )
            for agent_id, agent_setting in attacker_settings(_agent_settings).items()
        }

        # Set all instance variables
        self.rng = rng
        self._agent_states = agent_states
        self.sim_state: DynaMalSimulatorState = sim_state
        self.recording = recording
        self.sim_settings = sim_settings
        self.agent_settings = _agent_settings
        self._attacker_settings_by_full_name = attacker_settings_by_full_name
        self.rest_api_client = rest_api_client
        self._attack_graph = attack_graph
        self._static_data = static_sim_data
        self._dyna_native_sim = native_sim
        self._defender_reward_fns = defender_reward_fns
        self._attacker_reward_fns = attacker_reward_fns

    def __getstate__(self) -> dict[str, Any]:
        """Same as `MalSimulator.__getstate__`, but also leaves out the
        native `malsim._native.DynaSimulator` handle - kept under its own
        name because it's a different native type than `_native_sim`
        (PORTING_NOTES.md §11)."""
        state = super().__getstate__()
        state.pop('_dyna_native_sim', None)
        return state

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario | str,
        send_to_api: bool = False,
        sim_settings: MalSimulatorSettings | None = None,
    ) -> DynaMalSimulator:
        """Create a DynaMalSimulator object from a Scenario object or file

        Args:
            scenario - a Scenario object or a path to a scenario file
            send_to_api - whether to send data to GUI REST API or not
        """
        return dyna_create_simulator_from_scenario(scenario, send_to_api, sim_settings)

    def reset(
        self, seed: int | None = None
    ) -> dict[str, AttackerState | DefenderState]:
        """
        Reset the simulator to the initial state.
        Optionally, a seed can be provided to re-seed the random number generator.
        """
        (
            self._agent_states,
            self.sim_state,
            self.recording,
            self.agent_settings,
        ) = dyna_reset(
            self._static_data,
            self.agent_settings,
            self._attacker_settings_by_full_name,
            self.rng,
            self.rest_api_client,
            self._dyna_native_sim,
        )

        if seed is not None:
            self.rng = default_rng(seed)

        return self._agent_states

    def step(
        self, actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]]
    ) -> dict[str, AttackerState | DefenderState]:
        agent_states, recording, sim_state = dyna_step(
            self.recording,
            self.sim_state,
            self._agent_states,
            actions,
            self._dyna_native_sim,
            self.rest_api_client,
        )
        self._agent_states = agent_states
        self.recording = recording
        self.sim_state = sim_state

        return self._agent_states


def dyna_create_simulator_from_scenario(
    scenario: str | Scenario,
    send_to_api: bool = False,
    sim_settings: MalSimulatorSettings | None = None,
) -> DynaMalSimulator:
    if isinstance(scenario, str):
        # Load scenario if file was given
        scenario = Scenario.load_from_file(scenario)

    return DynaMalSimulator(
        scenario.attack_graph,
        sim_settings=sim_settings or scenario.sim_settings,
        send_to_api=send_to_api,
        agents=scenario.agent_settings,
    )


def _attacker_settings_with_full_names(
    settings: AttackerSettings[AttackGraphNode | str],
) -> AttackerSettings[str]:
    """`settings` with entry points and goals as full names, so they can be
    resolved again after a reset regenerates the nodes they refer to."""

    def full_name(node: AttackGraphNode | str) -> str:
        return node if isinstance(node, str) else node.full_name

    entry_points: tuple[Set[str], ...] | Set[str]
    if isinstance(settings.entry_points, Set):
        entry_points = frozenset(full_name(n) for n in settings.entry_points)
    else:
        entry_points = tuple(
            frozenset(full_name(n) for n in eps) for eps in settings.entry_points
        )
    return AttackerSettings(
        name=settings.name,
        entry_points=entry_points,
        goals=frozenset(full_name(n) for n in settings.goals),
        policy=settings.policy,
        actionable_steps=settings.actionable_steps,
        rewards=settings.rewards,
        config=settings.config,
        type=settings.type,
        reward_mode=settings.reward_mode,
        ttc_dists=settings.ttc_dists,
    )


def dyna_reset(
    static_data: MALSimulatorStaticData,
    agent_settings: AgentSettings,
    attacker_settings_by_full_name: Mapping[str, AttackerSettings[str]],
    rng: np.random.Generator,
    rest_api_client: MalSimGUIClient | None,
    native_sim: _native.DynaSimulator,
) -> tuple[
    AgentStates,
    DynaMalSimulatorState,
    Recording,
    AgentSettings,
]:
    """Reset attack graph and reinitialize agents.

    Delegates to `malsim._native.DynaSimulator.reset_native`
    (PORTING_NOTES.md §6 Phase B5, A9-equivalent) - the live `Model`
    (restored to the pristine snapshot native captured when the
    `DynaSimulator` was constructed in `DynaMalSimulator.__init__` - see
    `malsim_core::dyna_simulator::DynaSimulator`, PORTING_NOTES.md §11) and
    `AttackGraph` are both mutated in place through the shared handles,
    same `AttackGraph::partially_regenerate_graph` bookkeeping B1/B2
    already proved. "Multiple entry point sets, sampled at reset" is
    resolved here first, same reasoning as `mal_simulator.simulator.reset`.

    The model is restored (`restore_model_native`) *before* agent
    settings are flattened: nodes the previous episode's model effects
    removed only come back, with regenerated ids, on restore, so rules
    must be resolved against the restored graph (PORTING_NOTES.md §12).
    For the same reason attacker entry points and goals are re-resolved
    from `attacker_settings_by_full_name` after the restore, and the
    re-resolved `agent_settings` are returned alongside the states.
    """
    logger.info('Resetting Dyna MAL Simulator.')
    attack_graph = static_data.attack_graph
    settings = static_data.sim_settings
    assert attack_graph.model is not None, (
        'DynaMalSimulator requires attack_graph.model to be set.'
    )

    native_sim.restore_model_native()

    agent_settings = {
        **agent_settings,
        **{
            name: named.convert_to_attack_graph_nodes(attack_graph)
            for name, named in attacker_settings_by_full_name.items()
        },
    }
    _attacker_settings = attacker_settings(agent_settings)
    _defender_settings = defender_settings(agent_settings)

    entry_points_by_attacker = {
        name: get_entry_points(attack_graph, a_settings, rng)
        for name, a_settings in _attacker_settings.items()
    }

    native_settings = flatten_sim_settings(settings)
    native_agents: dict[str, Any] = {
        name: flatten_attacker_settings(
            a_settings, attack_graph, entry_points_by_attacker[name]
        )
        for name, a_settings in _attacker_settings.items()
    }
    native_agents.update(
        {
            name: flatten_defender_settings(d_settings, attack_graph)
            for name, d_settings in _defender_settings.items()
        }
    )

    native_seed = int(rng.integers(0, 2**63 - 1))
    native_out = native_sim.reset_native(native_settings, native_agents, native_seed)

    graph_state = _graph_state_from_native(attack_graph, native_out['sim_state'])
    sim_state = create_simulator_state(attack_graph, graph_state, settings)

    agent_states: AgentStates = {}
    for name, a_settings in _attacker_settings.items():
        agent_states[name] = create_attacker_state_from_native(
            sim_state,
            name,
            a_settings,
            entry_points_by_attacker[name],
            native_out['agents'][name],
            previous_state=None,
        )
    for name, d_settings in _defender_settings.items():
        agent_states[name] = create_defender_state_from_native(
            sim_state,
            name,
            d_settings,
            native_out['agents'][name],
            previous_state=None,
        )

    # Upload initial state to the REST API
    if rest_api_client:
        rest_api_client.upload_initial_state(attack_graph)

    return agent_states, sim_state, defaultdict(dict), agent_settings


def dyna_step(
    recording: Recording,
    sim_state: DynaMalSimulatorState,
    agent_states: AgentStates,
    actions: dict[str, list[AttackGraphNode]] | dict[str, list[str]],
    native_sim: _native.DynaSimulator,
    rest_api_client: MalSimGUIClient | None = None,
) -> tuple[AgentStates, Recording, DynaMalSimulatorState]:
    """Take a step in the simulation

    Args:
    actions - a dict mapping agent name to agent actions which is a list
                of AttackGraphNode or full names of the nodes to perform

    Returns:
    - A dictionary containing the agent state views keyed by agent names

    Delegates to `malsim._native.DynaSimulator.step_native`
    (PORTING_NOTES.md §6 Phase B5, A9-equivalent), which runs defenders
    before attackers internally and folds any model-effect-created nodes
    into its own `graph_state`/`enabled_defenses` as it goes (B1/B2),
    same ordering `dyna_step`'s old pure-Python version used.
    """

    _pre_step_check(agent_states, alive_agents(agent_states), actions)

    attack_graph = sim_state.attack_graph
    assert attack_graph.model is not None, (
        'DynaMalSimulator requires attack_graph.model to be set.'
    )
    native_actions = {
        name: [node.id for node in full_names_or_nodes_to_nodes(attack_graph, nodes)]
        for name, nodes in actions.items()
    }
    native_out = native_sim.step_native(native_actions)

    new_modification_record: list[AssetOp | AssocOp] = modification_record_from_native(
        attack_graph.model, native_out['sim_state']['step_modification_record']
    )
    # `ttc_values`/etc. only appear in this step's output when a model
    # effect actually grew the graph (PORTING_NOTES.md §6 Phase B5's
    # TTC-gap fix) - `build_step_output`'s `insert_graph_state_fields` call
    # is gated on `step_modification_record` being non-empty.
    new_graph_state = (
        _graph_state_from_native(attack_graph, native_out['sim_state'])
        if 'ttc_values' in native_out['sim_state']
        else None
    )
    sim_state = update_simulator_state(
        sim_state,
        frozenset(
            attack_graph.nodes[node_id]
            for node_id in native_out['sim_state']['step_enabled_defenses']
        ),
        new_modification_record,
        new_graph_state,
    )

    # Populate these from the results for all agents' actions.
    step_compromised_nodes: list[AttackGraphNode] = []
    step_enabled_defenses: list[AttackGraphNode] = []
    current_iteration = 0

    # Perform defender actions first
    for defender_state in defender_states(agent_states).values():
        new_defender_state = create_defender_state_from_native(
            sim_state,
            defender_state.name,
            defender_state.settings,
            native_out['agents'][defender_state.name],
            previous_state=defender_state,
        )
        current_iteration = defender_state.iteration

        requested = list(
            full_names_or_nodes_to_nodes(
                attack_graph, actions.get(defender_state.name, [])
            )
        )
        enabled = _ordered_new_nodes(requested, new_defender_state.step_performed_nodes)
        recording[current_iteration][defender_state.name] = enabled
        step_enabled_defenses += enabled
        agent_states[defender_state.name] = new_defender_state

    # Perform attacker actions afterwards
    for attacker_state in attacker_states(agent_states).values():
        new_attacker_state = create_attacker_state_from_native(
            sim_state,
            attacker_state.name,
            attacker_state.settings,
            attacker_state.entry_points,
            native_out['agents'][attacker_state.name],
            previous_state=attacker_state,
        )
        current_iteration = attacker_state.iteration

        requested = list(
            full_names_or_nodes_to_nodes(
                attack_graph, actions.get(attacker_state.name, [])
            )
        )
        compromised = _ordered_new_nodes(
            requested, new_attacker_state.step_performed_nodes
        )
        step_compromised_nodes += compromised
        recording[current_iteration][attacker_state.name] = compromised
        agent_states[attacker_state.name] = new_attacker_state

    # the way current_iteration is used here is flawed.
    if rest_api_client:
        rest_api_client.upload_performed_nodes(
            step_compromised_nodes + step_enabled_defenses,
            current_iteration,
        )

    return agent_states, recording, sim_state
