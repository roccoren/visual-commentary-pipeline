"""Standalone visual-first video commentary pipeline helpers."""

from .core import (
    DEFAULT_TERM_MAP,
    Segment,
    SegmentNarration,
    atempo_chain,
    build_azure_tts_ssml,
    build_segments,
    build_srt_text,
    format_srt_timestamp,
    normalize_terms,
    sample_times,
    serialize_manifest,
    write_srt_file,
)
from .qa_gate import QAGateResult, evaluate_narration_quality
from .state import Decision, Manifest, SegmentState, SegmentStatus

__all__ = [
    "DEFAULT_TERM_MAP",
    "Decision",
    "Manifest",
    "QAGateResult",
    "Segment",
    "SegmentNarration",
    "SegmentState",
    "SegmentStatus",
    "atempo_chain",
    "build_azure_tts_ssml",
    "build_segments",
    "build_srt_text",
    "evaluate_narration_quality",
    "format_srt_timestamp",
    "normalize_terms",
    "sample_times",
    "serialize_manifest",
    "write_srt_file",
]

