"""LangGraph StateGraph orchestration for the visual commentary pipeline.

This module re-expresses the imperative loop in ``pipeline.narrate_video`` as a
LangGraph ``StateGraph``.  Every node delegates to the **existing** step
functions in ``pipeline.py`` so that all business logic stays in one place.

Two graphs are built:
* ``build_segment_graph()`` — processes a single segment through boundary →
  frames → vision → narration → QA → TTS → duration-gate, with conditional
  retry edges.
* ``build_pipeline_graph()`` — top-level graph that initialises the manifest,
  iterates over segments (invoking the segment subgraph), and finalises outputs.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from .graph_state import PipelineState, SegmentProcessingState
from .state import Decision, Manifest, SegmentState, SegmentStatus


# ---------------------------------------------------------------------------
# Helpers to move between Manifest objects and serialisable dicts
# ---------------------------------------------------------------------------

def _manifest_from_state(state: dict[str, Any]) -> Manifest:
    return Manifest.from_dict(state["manifest_dict"])


def _save_manifest(state: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    manifest_path = Path(state["manifest_path"])
    manifest.save(manifest_path)
    return {"manifest_dict": manifest.to_dict()}


def _segment_from_manifest(manifest: Manifest, segment_id: int) -> SegmentState:
    return manifest.get_segment(segment_id)


# ---------------------------------------------------------------------------
# Segment-level graph nodes
# ---------------------------------------------------------------------------

def boundary_step(state: SegmentProcessingState) -> dict[str, Any]:
    """Check segment viability (skip / merge / accept)."""
    from .pipeline import get_previous_segment, note_segment_decision
    from .segment_policy import decide_segment_action

    manifest = _manifest_from_state(state)
    segment = _segment_from_manifest(manifest, state["segment_id"])

    boundary = decide_segment_action(
        segment,
        previous_segment=get_previous_segment(manifest, segment["segment_id"])
        if isinstance(segment, dict)
        else get_previous_segment(manifest, state["segment_id"]),
    )

    if boundary.decision == Decision.SKIP_SEGMENT:
        segment.status = SegmentStatus.SKIPPED
        note_segment_decision(segment, decision=boundary.decision, reason=boundary.reason)
        return {**_save_manifest(state, manifest), "boundary_ok": False, "segment_status": segment.status.value}

    if boundary.decision == Decision.MERGE_WITH_PREVIOUS:
        segment.status = SegmentStatus.NEEDS_HUMAN_REVIEW
        segment.human_review_status = "pending-merge-review"
        note_segment_decision(segment, decision=boundary.decision, reason=boundary.reason)
        return {**_save_manifest(state, manifest), "boundary_ok": False, "segment_status": segment.status.value}

    return {"boundary_ok": True}


def frame_extraction_step(state: SegmentProcessingState) -> dict[str, Any]:
    """Extract keyframes for the segment."""
    from .pipeline import run_frame_extraction_step

    manifest = _manifest_from_state(state)
    segment = _segment_from_manifest(manifest, state["segment_id"])

    if segment.status != SegmentStatus.PENDING:
        return {}

    run_frame_extraction_step(
        segment,
        input_video=Path(state["input_video"]),
        frames_dir=Path(state["frames_dir"]),
        ffmpeg_bin=state["ffmpeg_bin"],
    )
    return _save_manifest(state, manifest)


def vision_step(state: SegmentProcessingState) -> dict[str, Any]:
    """Call Azure OpenAI vision to understand the segment frames."""
    from .pipeline import run_vision_step

    manifest = _manifest_from_state(state)
    segment = _segment_from_manifest(manifest, state["segment_id"])

    if segment.status != SegmentStatus.FRAMES_EXTRACTED:
        return {}

    run_vision_step(manifest, segment)
    return _save_manifest(state, manifest)


def narration_step(state: SegmentProcessingState) -> dict[str, Any]:
    """Build narration text from vision result."""
    from .pipeline import run_narration_step

    manifest = _manifest_from_state(state)
    segment = _segment_from_manifest(manifest, state["segment_id"])

    if segment.status != SegmentStatus.FRAMES_EXTRACTED:
        return {}

    run_narration_step(manifest, segment)
    return _save_manifest(state, manifest)


def qa_gate_step(state: SegmentProcessingState) -> dict[str, Any]:
    """Evaluate narration quality and auto-rewrite once if needed."""
    from .pipeline import run_qa_gate_step

    manifest = _manifest_from_state(state)
    segment = _segment_from_manifest(manifest, state["segment_id"])

    run_qa_gate_step(manifest, segment)

    passed = segment.status == SegmentStatus.CONTENT_GENERATED
    return {
        **_save_manifest(state, manifest),
        "qa_passed": passed,
        "segment_status": segment.status.value,
    }


def tts_step(state: SegmentProcessingState) -> dict[str, Any]:
    """Synthesise speech and fit audio to the time budget."""
    from .pipeline import get_effective_style_policy, run_tts_step

    manifest = _manifest_from_state(state)
    segment = _segment_from_manifest(manifest, state["segment_id"])

    if segment.status != SegmentStatus.CONTENT_GENERATED:
        return {}

    style_policy = get_effective_style_policy(manifest)

    run_tts_step(
        segment,
        raw_audio_dir=Path(state["raw_audio_dir"]),
        fit_audio_dir=Path(state["fit_audio_dir"]),
        ffmpeg_bin=state["ffmpeg_bin"],
        ffprobe_bin=state["ffprobe_bin"],
        base_rate=style_policy["base_rate"],
        azure_style=style_policy["azure_style"],
        segment_buffer=state["segment_buffer"],
    )
    return _save_manifest(state, manifest)


def duration_gate_step(state: SegmentProcessingState) -> dict[str, Any]:
    """Check fitted audio against the time budget."""
    from .pipeline import run_duration_gate_step

    manifest = _manifest_from_state(state)
    segment = _segment_from_manifest(manifest, state["segment_id"])

    if segment.status != SegmentStatus.TTS_GENERATED:
        return {}

    run_duration_gate_step(segment)

    decision_value = segment.decision.value if segment.decision else "accept"
    return {
        **_save_manifest(state, manifest),
        "duration_decision": decision_value,
        "segment_status": segment.status.value,
    }


# ---------------------------------------------------------------------------
# Segment-level conditional edges
# ---------------------------------------------------------------------------

def route_after_boundary(state: SegmentProcessingState) -> Literal["frame_extraction", "__end__"]:
    if state.get("boundary_ok", True):
        return "frame_extraction"
    return END


def route_after_qa(state: SegmentProcessingState) -> Literal["tts", "__end__"]:
    if state.get("qa_passed", False):
        return "tts"
    return END


def route_after_duration(state: SegmentProcessingState) -> str:
    return END


# ---------------------------------------------------------------------------
# Segment subgraph builder
# ---------------------------------------------------------------------------

def build_segment_graph() -> StateGraph:
    """Build the per-segment processing subgraph.

    Flow::

        boundary → [accept?] → frame_extraction → vision → narration →
        qa_gate → [pass?] → tts → duration_gate → END
    """
    builder = StateGraph(SegmentProcessingState)

    builder.add_node("boundary", boundary_step)
    builder.add_node("frame_extraction", frame_extraction_step)
    builder.add_node("vision", vision_step)
    builder.add_node("narration", narration_step)
    builder.add_node("qa_gate", qa_gate_step)
    builder.add_node("tts", tts_step)
    builder.add_node("duration_gate", duration_gate_step)

    builder.add_edge(START, "boundary")
    builder.add_conditional_edges("boundary", route_after_boundary)
    builder.add_edge("frame_extraction", "vision")
    builder.add_edge("vision", "narration")
    builder.add_edge("narration", "qa_gate")
    builder.add_conditional_edges("qa_gate", route_after_qa)
    builder.add_edge("tts", "duration_gate")
    builder.add_conditional_edges("duration_gate", route_after_duration)

    return builder


# ---------------------------------------------------------------------------
# Pipeline-level graph nodes
# ---------------------------------------------------------------------------

def init_pipeline(state: PipelineState) -> dict[str, Any]:
    """Discover ffmpeg, resolve paths, load or create manifest."""
    from .pipeline import (
        find_ffmpeg,
        find_ffprobe,
        load_or_create_manifest,
        maybe_apply_redo,
        resolve_target_segment_ids,
        validate_target_segment_ids,
    )

    args = _args_from_dict(state["args_dict"])
    ffmpeg_bin = find_ffmpeg()
    ffprobe_bin = find_ffprobe(ffmpeg_bin)

    input_video = Path(args.input).expanduser().resolve()
    output_video = Path(args.output).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "frames").mkdir(exist_ok=True)
    (workdir / "tts_raw").mkdir(exist_ok=True)
    (workdir / "tts_fit").mkdir(exist_ok=True)
    manifest_path = workdir / "commentary_manifest.json"

    target_segment_ids = resolve_target_segment_ids(args)

    manifest = load_or_create_manifest(
        args,
        input_video=input_video,
        output_video=output_video,
        workdir=workdir,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        manifest_path=manifest_path,
    )
    validate_target_segment_ids(manifest, target_segment_ids)
    maybe_apply_redo(manifest, args, manifest_path=manifest_path, target_segment_ids=target_segment_ids)

    manifest.status = "running"
    manifest.save(manifest_path)

    return {
        "input_video": str(input_video),
        "output_video": str(output_video),
        "workdir": str(workdir),
        "ffmpeg_bin": ffmpeg_bin,
        "ffprobe_bin": ffprobe_bin,
        "manifest_dict": manifest.to_dict(),
        "manifest_path": str(manifest_path),
        "segments_total": len(manifest.segments),
        "current_segment_index": 0,
        "target_segment_ids": sorted(target_segment_ids) if target_segment_ids else None,
        "redo": args.redo,
    }


def process_segments(state: PipelineState) -> dict[str, Any]:
    """Iterate over segments, invoking the segment subgraph for each."""
    from .pipeline import should_process_segment

    segment_graph = build_segment_graph().compile()

    manifest = _manifest_from_state(state)
    target_ids = set(state["target_segment_ids"]) if state.get("target_segment_ids") else None

    workdir = Path(state["workdir"])

    for segment in manifest.segments:
        if not should_process_segment(segment, redo=state.get("redo"), target_segment_ids=target_ids):
            continue

        seg_input: SegmentProcessingState = {
            "segment_id": segment.id,
            "manifest_dict": manifest.to_dict(),
            "manifest_path": state["manifest_path"],
            "input_video": state["input_video"],
            "workdir": state["workdir"],
            "frames_dir": str(workdir / "frames"),
            "raw_audio_dir": str(workdir / "tts_raw"),
            "fit_audio_dir": str(workdir / "tts_fit"),
            "ffmpeg_bin": state["ffmpeg_bin"],
            "ffprobe_bin": state["ffprobe_bin"],
            "segment_buffer": manifest.segment_buffer,
            "base_rate": manifest.base_rate,
            "azure_style": manifest.azure_style,
            "boundary_ok": True,
            "qa_passed": False,
            "duration_decision": "",
            "tts_retry_count": 0,
            "narration_retry_count": 0,
            "max_tts_retries": 1,
            "max_narration_retries": 1,
            "segment_status": segment.status.value,
            "error": "",
        }

        result = segment_graph.invoke(seg_input)

        # Sync the manifest back from the segment subgraph output
        if result.get("manifest_dict"):
            manifest = Manifest.from_dict(result["manifest_dict"])

        seg = manifest.get_segment(segment.id)
        print(
            f"[segment {seg.id:03d}] {seg.start:.2f}-{seg.end:.2f}s | "
            f"status={seg.status.value} decision={(seg.decision.value if seg.decision else 'none')}"
        )

    return {"manifest_dict": manifest.to_dict()}


def finalize_pipeline(state: PipelineState) -> dict[str, Any]:
    """Compose audio, mux video, write SRT, and update manifest."""
    from .pipeline import finalize_outputs

    manifest = _manifest_from_state(state)
    result = finalize_outputs(
        manifest,
        ffmpeg_bin=state["ffmpeg_bin"],
        output_video=Path(state["output_video"]),
        workdir=Path(state["workdir"]),
    )
    return {"result": result, "manifest_dict": manifest.to_dict()}


# ---------------------------------------------------------------------------
# Pipeline-level graph builder
# ---------------------------------------------------------------------------

def build_pipeline_graph() -> StateGraph:
    """Build the top-level pipeline graph.

    Flow::

        init_pipeline → process_segments → finalize → END
    """
    builder = StateGraph(PipelineState)

    builder.add_node("init", init_pipeline)
    builder.add_node("process_segments", process_segments)
    builder.add_node("finalize", finalize_pipeline)

    builder.add_edge(START, "init")
    builder.add_edge("init", "process_segments")
    builder.add_edge("process_segments", "finalize")
    builder.add_edge("finalize", END)

    return builder


# ---------------------------------------------------------------------------
# Public entry point (drop-in replacement for narrate_video)
# ---------------------------------------------------------------------------

def _args_from_dict(d: dict[str, Any]) -> argparse.Namespace:
    """Reconstruct an argparse.Namespace from a plain dict."""
    return argparse.Namespace(**d)


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Serialise an argparse.Namespace to a JSON-safe dict."""
    raw = vars(args)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def narrate_video_graph(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full pipeline through the LangGraph StateGraph.

    This is a drop-in replacement for ``pipeline.narrate_video``.
    """
    graph = build_pipeline_graph().compile()

    initial_state: PipelineState = {
        "args_dict": args_to_dict(args),
        "current_segment_index": 0,
        "segments_total": 0,
    }

    final_state = graph.invoke(initial_state)
    return final_state.get("result", {})
