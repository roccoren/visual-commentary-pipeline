"""Minimal QA gate for narration quality checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EXPLICIT_TRANSITION_BANNED_TOKENS = (
    "上一页",
    "下一页",
    "接下来",
    "基于前面",
    "如前所述",
)

from .state import Decision


@dataclass
class QAConfig:
    """Tunable QA thresholds — surfaced as CLI flags."""

    max_cps_soft: float = 8.5
    """chars/second soft limit → RETRY_NARRATION."""
    max_cps_hard: float = 12.0
    """chars/second hard limit → NEEDS_HUMAN_REVIEW."""
    density_factor: int = 8
    """Factor in ``max_chars = max(18, min(90, int(duration * factor)))``."""
    max_narration_retries: int | None = None
    """Max LLM rewrite attempts per segment (None = auto: 2 with critic, 1 without)."""
    critic_lenient: bool = False
    """When True, high-severity critic issues trigger RETRY instead of NEEDS_HUMAN_REVIEW."""

    def max_chars(self, duration_seconds: float) -> int:
        """Compute the maximum character count for a segment."""
        return max(18, min(90, int(duration_seconds * self.density_factor)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QAConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_DEFAULT_QA_CONFIG = QAConfig()


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
    qa_config: QAConfig | None = None,
) -> QAGateResult:
    cfg = qa_config or _DEFAULT_QA_CONFIG
    cleaned = narration.strip()
    normalized = _normalize_for_compare(cleaned)
    previous_normalized = _normalize_for_compare(previous_narration)
    char_count = len(cleaned)
    chars_per_second = char_count / max(duration_seconds, 0.1)
    max_chars = cfg.max_chars(duration_seconds)

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

    banned_hits = [token for token in EXPLICIT_TRANSITION_BANNED_TOKENS if token in cleaned]
    if banned_hits:
        return QAGateResult(
            passed=False,
            decision=Decision.RETRY_NARRATION,
            reason="narration uses explicit slide-to-slide transition phrasing",
            feedback=["explicit transition phrasing"],
            details={
                "char_count": char_count,
                "duration_seconds": round(duration_seconds, 3),
                "banned_hits": ",".join(banned_hits),
            },
        )

    if char_count > max_chars * 1.5 or chars_per_second > cfg.max_cps_hard:
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

    if char_count > max_chars or chars_per_second > cfg.max_cps_soft:
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
