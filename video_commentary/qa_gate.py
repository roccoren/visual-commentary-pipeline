"""Minimal QA gate for narration quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .state import Decision


@dataclass
class QAGateResult:
    passed: bool
    decision: Decision
    reason: str
    feedback: list[str] = field(default_factory=list)
    details: dict[str, float | int | str] = field(default_factory=dict)


def _normalize_for_compare(text: str) -> str:
    return "".join(ch.lower() for ch in text.strip() if not ch.isspace())


def evaluate_narration_quality(
    *,
    narration: str,
    previous_narration: str,
    duration_seconds: float,
) -> QAGateResult:
    cleaned = narration.strip()
    normalized = _normalize_for_compare(cleaned)
    previous_normalized = _normalize_for_compare(previous_narration)
    char_count = len(cleaned)
    chars_per_second = char_count / max(duration_seconds, 0.1)
    max_chars = max(18, min(90, int(duration_seconds * 8)))

    if not normalized:
        return QAGateResult(
            passed=False,
            decision=Decision.RETRY_NARRATION,
            reason="narration is empty after normalization",
            feedback=["empty narration"],
            details={"char_count": char_count, "duration_seconds": round(duration_seconds, 3)},
        )

    if previous_normalized and normalized == previous_normalized:
        return QAGateResult(
            passed=False,
            decision=Decision.RETRY_NARRATION,
            reason="narration duplicates the previous accepted segment",
            feedback=["repetitive narration"],
            details={"char_count": char_count, "duration_seconds": round(duration_seconds, 3)},
        )

    if char_count > max_chars * 1.5 or chars_per_second > 12.0:
        return QAGateResult(
            passed=False,
            decision=Decision.NEEDS_HUMAN_REVIEW,
            reason="narration is far too dense for the segment duration",
            feedback=["too long or too dense narration"],
            details={
                "char_count": char_count,
                "duration_seconds": round(duration_seconds, 3),
                "chars_per_second": round(chars_per_second, 3),
                "max_chars": max_chars,
            },
        )

    if char_count > max_chars or chars_per_second > 8.5:
        return QAGateResult(
            passed=False,
            decision=Decision.RETRY_NARRATION,
            reason="narration is somewhat too dense for the segment duration",
            feedback=["too long or too dense narration"],
            details={
                "char_count": char_count,
                "duration_seconds": round(duration_seconds, 3),
                "chars_per_second": round(chars_per_second, 3),
                "max_chars": max_chars,
            },
        )

    return QAGateResult(
        passed=True,
        decision=Decision.ACCEPT,
        reason="qa gate passed",
        feedback=[],
        details={
            "char_count": char_count,
            "duration_seconds": round(duration_seconds, 3),
            "chars_per_second": round(chars_per_second, 3),
            "max_chars": max_chars,
        },
    )
