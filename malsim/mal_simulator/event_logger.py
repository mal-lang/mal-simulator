"""EventLog functionality which uses detectors to generate logs of compromised nodes"""

from collections.abc import Iterable
from dataclasses import dataclass
import logging
import numpy as np

from maltoolbox.attackgraph import AttackGraphNode, Detector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogEntry:
    """A single log entry produced by a detector."""

    timestep: int
    detector_name: str
    trigger: AttackGraphNode
    context: dict[str, AttackGraphNode]


def collect_false_positives(
    iteration: int,
    detectors: Iterable[Detector],
    rng: np.random.Generator,
) -> tuple[LogEntry, ...]:
    """Generates logs for false positives from given detectors."""
    logs = []
    for detector in detectors:
        if detector.fprate >= rng.random():
            logs.append(
                LogEntry(
                    timestep=iteration,
                    detector_name=str(detector.name),
                    trigger=detector.node,
                    context=get_random_context(detector),
                )
            )
            logging.debug(
                'Detector %s false positive on %s',
                detector.name,
                detector.node.full_name,
            )
        else:
            logging.debug(
                'Detector %s true negative on %s',
                detector.name,
                detector.node.full_name,
            )

    return tuple(logs)


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
            labeled_steps = get_context(
                detector,
                previous_compromised_nodes,
            )
            assert attack_step.model_asset is not None, 'Attack step has no model asset'
            if detector.tprate >= rng.random():
                logs.append(
                    LogEntry(
                        timestep=iteration,
                        detector_name=str(detector.name),
                        trigger=attack_step,
                        context=labeled_steps,
                    )
                )
                logging.debug(
                    'Detector %s true positive on %s',
                    detector.name,
                    attack_step.full_name,
                )
            else:
                logging.debug(
                    'Detector %s false negative on %s',
                    detector.name,
                    attack_step.full_name,
                )

    return tuple(logs)


def get_context(
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
    context_nodes = {}

    for label, potential_context_nodes in detector.potential_context.items():
        node = next(
            node
            for node in potential_context_nodes
            if node in previous_compromised_nodes
        )

        if node is not None:
            context_nodes[label] = node

    return context_nodes


def get_random_context(detector: Detector) -> dict[str, AttackGraphNode]:
    """Finds random nodes that satisfies the context of a detector"""
    context_nodes = {}

    for label, potential_context_nodes in detector.potential_context.items():
        node = next(node for node in potential_context_nodes)
        if node is not None:
            context_nodes[label] = node
    return context_nodes
