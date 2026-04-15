"""LangGraph state definition for the visual commentary pipeline.

This module defines the TypedDict-based state used by the LangGraph StateGraph
orchestration layer.  The state wraps the existing Manifest / SegmentState models
so that graph nodes can read and mutate pipeline state through a single dict.
"""

from __future__ import annotations

from typing import Any, TypedDict


class SegmentProcessingState(TypedDict, total=False):
    """State carried through the per-segment subgraph."""

    # Identity
    segment_id: int

    # Pipeline context passed in at the start of each segment loop
    manifest_dict: dict[str, Any]
    input_video: str
    workdir: str
    frames_dir: str
    raw_audio_dir: str
    fit_audio_dir: str
    ffmpeg_bin: str
    ffprobe_bin: str
    segment_buffer: float
    base_rate: str
    azure_style: str

    # Manifest persistence
    manifest_path: str

    # Content Understanding data (Phase 2)
    cu_segment_data: dict[str, Any]

    # Enhanced pipeline feature flags
    use_content_understanding: bool
    use_llm_critic: bool
    use_doc_intel: bool

    # QA tuning (serialised QAConfig dict)
    qa_config: dict[str, Any]

    # Control flow signals set by gate nodes
    boundary_ok: bool
    qa_passed: bool
    duration_decision: str  # "accept" | "retry_tts" | "retry_narration"
    tts_retry_count: int
    narration_retry_count: int
    max_tts_retries: int
    max_narration_retries: int

    # Outcome
    segment_status: str  # mirrors SegmentStatus.value
    error: str


class PipelineState(TypedDict, total=False):
    """Top-level state for the pipeline-level graph."""

    # Inputs
    input_video: str
    output_video: str
    workdir: str

    # CLI arguments forwarded verbatim
    args_dict: dict[str, Any]

    # Tool binaries discovered once
    ffmpeg_bin: str
    ffprobe_bin: str

    # Manifest as a serialisable dict (round-tripped through Manifest.to_dict/from_dict)
    manifest_dict: dict[str, Any]
    manifest_path: str

    # Content Understanding raw result (Phase 2)
    cu_result: dict[str, Any]

    # Enhanced pipeline feature flags
    use_content_understanding: bool
    use_llm_critic: bool
    use_doc_intel: bool
    use_llm_profiler: bool

    # QA tuning (serialised QAConfig dict)
    qa_config: dict[str, Any]

    # Segment processing tracking
    current_segment_index: int
    segments_total: int
    target_segment_ids: list[int] | None
    redo: str | None

    # Final output artefacts
    result: dict[str, Any]
    error: str
