"""Log entries produced by detectors.

Detector rolls and log generation run natively (`malsim._native`); this
module only holds the mirrored `LogEntry` dataclass - see PORTING_NOTES.md.
"""

from dataclasses import dataclass

from maltoolbox.attackgraph import AttackGraphNode, Detector


@dataclass(frozen=True)
class LogEntry:
    """A single log entry produced by a detector."""

    timestep: int
    detector_name: str
    detector: Detector
    trigger: AttackGraphNode
    context: dict[str, AttackGraphNode]
    false_positive: bool = False
