"""EventLog functionality which uses detectors to generate logs of compromised nodes"""

from collections.abc import Iterable
from dataclasses import dataclass
import logging
import numpy as np

from maltoolbox.attackgraph import AttackGraphNode, Detector

from .ttc_utils import TTCDist

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogEntry:
    """A single log entry produced by a detector."""

    timestep: int
    detector_name: str
    asset_name: str
    attack_step_name: str
    context_nodes: dict[str, AttackGraphNode]


# def false_positive_logs(
#     iteration: int,
#     detectors: Iterable[Detector],
#     rng: np.random.Generator,
# ):
#     """
#     Generates false positive logs for a defender based on the detectors
#     in the attack graph and their false positive rates.
#     """
#     fp_logs = []
#     for detector in detectors:
#         if not detector.fprate:
#             continue

#         ttc_dist = TTCDist.from_dict(detector.fprate)
#         if ttc_dist.attempt_bernoulli(rng):
#             fp_logs.append(
#                 LogEntry(
#                 timestep=iteration,
#                 detector_name=detector.name,
#                 asset_name=detector.node.model_asset.name if detector.node.model_asset else '',
#                     attack_step_name=detector.node.name,
#                     context_nodes={},
#                 )
#             )

#     return tuple(fp_logs)

def get_tprate(detector: Detector) -> float:
    """Get the true positive rate for a detector, if it exists."""
    if not detector.tprate:
        return 1.0

    if detector.tprate['type'] != 'number':
        logger.warning(
            'Unsupported tprate type %s for detector %s. Defaulting to 1.0.',
            detector.tprate.get('type'),
            detector.name,
        )
        return 1.0

    return detector.tprate.get('value', 1.0)

def collect_logs(
    iteration: int,
    step_compromised_nodes: Iterable[AttackGraphNode],
    previous_compromised_nodes: Iterable[AttackGraphNode],
    rng: np.random.Generator,
) -> tuple[LogEntry, ...]:
    """Generates logs from an agents latest step.

    Args:
        attack_state: The attacker state from which to collect logs.
    """
    logs = []
    for attack_step in step_compromised_nodes:
        for detector in attack_step.detectors.values():
            labeled_steps = get_context_steps(
                detector, previous_compromised_nodes,
            )
            assert attack_step.model_asset is not None, 'Attack step has no model asset'
            tprate = get_tprate(detector)
            if tprate >= rng.random():
                logs.append(
                    LogEntry(
                        timestep=iteration,
                        detector_name=detector.name,
                        asset_name=attack_step.model_asset.name,
                        attack_step_name=attack_step.name,
                        context_nodes=labeled_steps,
                    )
                )
                logging.debug(
                    'Detector %s true positive on %s', detector.name, attack_step.full_name
                )
            else:
                logging.debug(
                    'Detector %s false negative on %s', detector.name, attack_step.full_name
                )

    return tuple(logs)


def get_context_steps(
    detector: Detector,
    previous_compromised_nodes: Iterable[AttackGraphNode],
) -> dict[str, AttackGraphNode]:
    """Finds node that satisfies the context of a detector
    among the previously compromised nodes.

    Args:
        detector: The detector whose context nodes we want to find.
        previous_compromised_nodes: nodes that have been previously compromised.

    Returns:
        A dictionary where keys are the detector context labels and values are the
        nodes that have been selected to populate the context.
    """
    context_assets = {}

    for label, potential_context_nodes in detector.potential_context.items():
        node = next(
            node for node in potential_context_nodes if node in previous_compromised_nodes
        )

        if node is not None:
            context_assets[label] = node

    return context_assets
