"""Duration-fit decision policy for segment narration."""

from __future__ import annotations

from dataclasses import dataclass

from .state import Decision


@dataclass
class DurationPolicyResult:
    decision: Decision
    reason: str
    gap_ms: int


def decide_duration_action(*, budget_seconds: float, fitted_duration_seconds: float) -> DurationPolicyResult:
    gap_ms = int(round((fitted_duration_seconds - budget_seconds) * 1000))

    if gap_ms <= 300:
        return DurationPolicyResult(
            decision=Decision.ACCEPT,
            reason="fitted audio is within acceptable budget tolerance",
            gap_ms=gap_ms,
        )
    if gap_ms <= 1200:
        return DurationPolicyResult(
            decision=Decision.RETRY_TTS,
            reason="audio slightly exceeds budget; retry TTS or timing settings before rewriting",
            gap_ms=gap_ms,
        )
    return DurationPolicyResult(
        decision=Decision.RETRY_NARRATION,
        reason="audio significantly exceeds budget; rewrite shorter narration instead of over-compressing audio",
        gap_ms=gap_ms,
    )
