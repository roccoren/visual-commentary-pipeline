"""Basic segment boundary policy decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .state import Decision, SegmentState


@dataclass
class SegmentPolicyResult:
    decision: Decision
    reason: str


def decide_segment_action(segment: SegmentState, *, previous_segment: SegmentState | None = None) -> SegmentPolicyResult:
    if segment.duration < 0.8:
        return SegmentPolicyResult(
            decision=Decision.SKIP_SEGMENT,
            reason="segment too short to narrate robustly",
        )

    if previous_segment and segment.duration < 1.6:
        return SegmentPolicyResult(
            decision=Decision.MERGE_WITH_PREVIOUS,
            reason="segment is very short; merge with previous for more natural narration",
        )

    return SegmentPolicyResult(
        decision=Decision.ACCEPT,
        reason="segment boundary accepted",
    )
