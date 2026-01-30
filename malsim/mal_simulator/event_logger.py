"""EventLog functionality which uses detectors to generate logs of compromised nodes"""

from collections.abc import Iterable
from dataclasses import dataclass
import functools
import logging

from maltoolbox.attackgraph import AttackGraphNode
from maltoolbox.language.detector import Detector

from malsim.mal_simulator.attacker_state import MalSimAttackerState
from malsim.mal_simulator.simulator_static_data import MALSimulatorStaticData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogEntry:
    """A single log entry produced by a detector."""

    timestep: int
    detector_name: str
    asset_name: str
    attack_step_name: str
    context_assets: dict[str, str]


class EventLogger:
    """Collects logs from agents during attack simulations."""

    def __init__(self):
        """Initialize an event logger to produce synthetic event logs.

        Args:
          attack_graph: An attack graph to lookup assets in.
          agent_class_name: The name of the agent whose actions will be logged.
          agent_profile_name: The profile of the agent whose actions will be logged.
        """

        self.logs: list[LogEntry] = []

    def __repr__(self) -> str:
        return f'EventLogger(num_logs={len(self.logs)})'

    def collect_logs(
        self,
        attacker_state: MalSimAttackerState,
    ) -> None:
        """Generates logs from an agents latest step and saves them to self.logs.

        Args:
          attack_state: The attacker state from which to collect logs.
        """

        for attack_step in attacker_state.step_performed_nodes:

          for _, detector in attack_step.detectors.items():

              labeled_assets = self._get_context_assets(
                  detector,
                  attacker_state.performed_nodes,
                  attack_step,
              )

              self.logs.append(
                  LogEntry(
                      timestep=attacker_state.iteration,
                      detector_name=detector.name,
                      asset_name=attack_step.model_asset.name,
                      attack_step_name=attack_step.name,
                      context_assets=labeled_assets,
                  )
              )

              logging.debug(
                  'Detector %s triggered on %s', detector.name, attack_step.full_name
              )

    def _get_context_assets(
        self,
        detector: Detector,
        reached_steps: Iterable[AttackGraphNode],
        attack_step: AttackGraphNode,
    ) -> dict[str, str]:
        """Finds model assets that satisfy the detector's context.

        Only model assets that have at least one attack step already compromised are
        considered as possible candidates for the detector context as well as the
        current asset that holds the attack step whose detectors are being resolved.
        If multiple model assets of the type defined in the detector context, the
        last one compromised is selected.

        Args:
          detector: The detector whose context assets needs resolving
          reached_steps: An iterable containing candidate (i.e. already compromised)
            steps to populate the detector context.
          attack_step: The attack_step holding the detector.

        Returns:
          A dictionary where keys are the detector context labels and values are the
          model assets that have been selected to populate the context.
        """
        context_assets = {}

        for label, lang_graph_asset in detector.context.items():

            if attack_step.model_asset.type == lang_graph_asset.name:
                # The model asset that holds (the step that holds) the detector matches
                # the type required by the detector, thus is selected. This is done to
                # support cases where a log needs to reference the resource that
                # triggers the log itself. In the future a solution at the MAL-spec
                # level (e.g. using `__self__` instead of an asset type in the context
                # definition) would make this more robust.
                context_assets[label] = attack_step.model_asset.name
                continue

            try:
                *_, asset = (
                    step.model_asset
                    for step in reached_steps
                    if step.model_asset.type == lang_graph_asset.name
                )
            except ValueError as e:
                msg = (
                    f'Context {detector.context} cannot be satisfied '
                    f'for step {attack_step.full_name}. No {lang_graph_asset.name} '
                    'was compromised already.'
                )
                raise ValueError(msg) from e

            context_assets[label] = asset.name

        return context_assets
