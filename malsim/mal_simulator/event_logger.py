"""EventLog functionality which uses detectors to generate logs of compromised nodes"""

from collections.abc import Iterable
from dataclasses import dataclass
import logging

from maltoolbox.attackgraph import AttackGraphNode
from maltoolbox.language.detector import Detector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogEntry:
    """A single log entry produced by a detector."""

    timestep: int
    detector_name: str
    asset_name: str
    attack_step_name: str
    context_assets: dict[str, str]


def collect_logs(
    iteration: int,
    step_compromised_nodes: Iterable[AttackGraphNode],
    previous_compromised_nodes: Iterable[AttackGraphNode],
) -> tuple[LogEntry, ...]:
    """Generates logs from an agents latest step.

    Args:
        attack_state: The attacker state from which to collect logs.
    """
    logs = []
    for attack_step in step_compromised_nodes:
        for detector in attack_step.detectors.values():
            labeled_assets = get_context_assets(
                detector,
                previous_compromised_nodes,
                attack_step,
            )
            assert attack_step.model_asset is not None, 'Attack step has no model asset'
            logs.append(
                LogEntry(
                    timestep=iteration,
                    detector_name=detector.name,
                    asset_name=attack_step.model_asset.name,
                    attack_step_name=attack_step.name,
                    context_assets=labeled_assets,
                )
            )
            logging.debug(
                'Detector %s triggered on %s', detector.name, attack_step.full_name
            )

    return tuple(logs)


def get_context_assets(
    detector: Detector,
    previous_compromised_nodes: Iterable[AttackGraphNode],
    attack_step: AttackGraphNode,
) -> dict[str, str]:
    """Finds model assets that satisfy the detector's context.

    Args:
        detector: The detector whose context assets needs resolving
        previous_compromised_nodes: An iterable containing candidate
        (i.e. already compromised) steps to populate the detector context.
        attack_step: The attack_step holding the detector.

    Returns:
        A dictionary where keys are the detector context labels and values are the
        model assets that have been selected to populate the context.
    """
    context_assets = {}

    for label, lang_graph_asset in detector.context.items():
        assert attack_step.model_asset, 'Attack step has no model asset'
        if attack_step.model_asset.type == lang_graph_asset.name:
            # The current asset satisfies the context requirement
            context_assets[label] = attack_step.model_asset.name
            continue

        # Search among already compromised steps for a matching asset
        asset = next(
            (
                step.model_asset
                for step in reversed(list(previous_compromised_nodes))
                if step.model_asset is not None
                and step.model_asset.type == lang_graph_asset.name
            ),
            None,
        )

        # No matching asset found among already compromised steps
        if asset is None:
            msg = (
                f'Context {detector.context} cannot be satisfied '
                f'for step {attack_step.full_name}. No {lang_graph_asset.name} '
                'was compromised already.'
            )
            raise ValueError(msg)

        context_assets[label] = asset.name
    return context_assets
