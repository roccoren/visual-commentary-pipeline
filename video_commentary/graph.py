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

Enhanced capabilities (opt-in via CLI flags):
* ``--use-content-understanding`` — Azure Content Understanding for video analysis
* ``--use-llm-critic`` — LLM-based QA critic with rewrite loops
* ``--use-doc-intel`` — Document Intelligence OCR for UI screenshots
* ``--use-llm-profiler`` — LLM-based video profiling from sampled frames
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
    """Understand the segment frames — enhanced with multi-model fusion.

    When ``use_content_understanding`` or ``use_doc_intel`` is set, delegates
    to the multi-model :func:`vision_enricher.enrich_segment_vision`.
    Otherwise falls back to the original Azure OpenAI vision call.
    """
    manifest = _manifest_from_state(state)
    segment = _segment_from_manifest(manifest, state["segment_id"])

    if segment.status != SegmentStatus.FRAMES_EXTRACTED:
        return {}

    use_cu = state.get("use_content_understanding", False)
    use_di = state.get("use_doc_intel", False)

    if use_cu or use_di:
        from .pipeline import get_effective_style_policy, get_previous_accepted_narration
        from .vision_enricher import enrich_segment_vision

        cu_data = state.get("cu_segment_data") or segment.vision_result
        if cu_data and cu_data.get("source") == "content_understanding":
            cu_data_for_enricher = cu_data
        else:
            cu_data_for_enricher = None

        enriched = enrich_segment_vision(
            segment,
            cu_data=cu_data_for_enricher,
            use_doc_intel=use_di,
            use_gpt4o_vision=True,
            previous_narration=get_previous_accepted_narration(manifest, state["segment_id"]),
            style_policy=get_effective_style_policy(manifest),
        )
        segment.vision_result = enriched
        segment.title = enriched.get("title", segment.title)
        segment.visible_points = enriched.get("visible_points", segment.visible_points)
        segment.on_screen_text = enriched.get("on_screen_text", segment.on_screen_text)

        from .pipeline import _assign_semantic_groups
        _assign_semantic_groups(manifest)
    else:
        from .pipeline import run_vision_step
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
    """Evaluate narration quality — two-layer when LLM critic is enabled.

    Layer 1 (always): rule-based checks from ``qa_gate.py``.
    Layer 2 (opt-in): LLM critic from ``llm_critic.py`` — runs only if rules
    pass so that fast deterministic checks still catch obvious problems for free.

    When the LLM critic fails narration and ``narration_retry_count`` is below
    ``max_narration_retries``, it triggers an LLM-guided rewrite and
    re-evaluates, implementing a proper critic-feedback loop.
    """
    from .pipeline import get_previous_accepted_narration, note_segment_decision, run_qa_gate_step

    manifest = _manifest_from_state(state)
    segment = _segment_from_manifest(manifest, state["segment_id"])
    use_llm = state.get("use_llm_critic", False)

    if not use_llm:
        # Original rule-based QA gate
        run_qa_gate_step(manifest, segment)
        passed = segment.status == SegmentStatus.CONTENT_GENERATED
        return {
            **_save_manifest(state, manifest),
            "qa_passed": passed,
            "segment_status": segment.status.value,
        }

    # Two-layer QA with LLM critic
    from .llm_critic import evaluate_narration_two_layer, rewrite_narration_llm

    previous_narration = get_previous_accepted_narration(manifest, state["segment_id"])
    max_retries = state.get("max_narration_retries", 2)
    retry_count = state.get("narration_retry_count", 0)

    passed, decision, reason, feedback, critic_result = evaluate_narration_two_layer(
        narration=segment.selected_draft,
        segment=segment,
        previous_narration=previous_narration,
        use_llm=True,
    )

    if passed:
        segment.final_decision = Decision.ACCEPT
        segment.critic_feedback = feedback
        return {
            **_save_manifest(state, manifest),
            "qa_passed": True,
            "segment_status": segment.status.value,
        }

    # Failed — try LLM rewrite if we have retries left
    segment.critic_feedback = feedback
    note_segment_decision(segment, decision=decision, reason=reason)

    if critic_result and retry_count < max_retries and decision == Decision.RETRY_NARRATION:
        rewrite_result = rewrite_narration_llm(
            narration=segment.selected_draft,
            segment=segment,
            critic_result=critic_result,
            previous_narration=previous_narration,
        )

        if rewrite_result.confidence > 0.0:
            segment.rewritten_draft = rewrite_result.narration_zh
            segment.selected_draft = rewrite_result.narration_zh
            segment.draft_candidates.append(rewrite_result.narration_zh)
            segment.rewrite_attempt_count += 1

            # Re-evaluate the rewrite
            passed2, decision2, reason2, feedback2, _ = evaluate_narration_two_layer(
                narration=segment.selected_draft,
                segment=segment,
                previous_narration=previous_narration,
                use_llm=True,
            )
            segment.critic_feedback = feedback2
            note_segment_decision(
                segment, decision=decision2,
                reason=f"post-llm-rewrite: {reason2}",
                details={"changes": rewrite_result.changes_made},
            )

            if passed2:
                segment.final_decision = Decision.ACCEPT
                return {
                    **_save_manifest(state, manifest),
                    "qa_passed": True,
                    "narration_retry_count": retry_count + 1,
                    "segment_status": segment.status.value,
                }

    # Exhausted retries or high-severity issue
    segment.final_decision = decision
    segment.status = SegmentStatus.NEEDS_HUMAN_REVIEW
    segment.human_review_status = "llm-qa-failed"

    return {
        **_save_manifest(state, manifest),
        "qa_passed": False,
        "narration_retry_count": retry_count + 1,
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
    """Discover ffmpeg, resolve paths, load or create manifest.

    When ``--use-content-understanding`` is enabled, runs Azure Content
    Understanding analysis on the video and uses the result for segmentation.

    When ``--use-llm-profiler`` is enabled, uses LLM vision analysis of
    sampled frames for video type inference instead of filename heuristics.
    """
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

    # Feature flags from CLI
    use_cu = getattr(args, "use_content_understanding", False)
    use_llm_critic = getattr(args, "use_llm_critic", False)
    use_doc_intel = getattr(args, "use_doc_intel", False)
    use_llm_profiler = getattr(args, "use_llm_profiler", False)

    cu_result: dict[str, Any] = {}

    # Phase 2: Content Understanding analysis (if enabled and creating new manifest)
    if use_cu and not getattr(args, "resume_from_manifest", None) and (
        not manifest_path.exists() or getattr(args, "force_replan", False)
    ):
        try:
            from .content_understanding import analyze_video, content_understanding_to_segments
            from .pipeline import build_manifest, ffprobe_duration

            duration = ffprobe_duration(ffprobe_bin, input_video)
            cu_result = analyze_video(input_video)

            cu_segments = content_understanding_to_segments(
                cu_result,
                video_duration=duration,
                min_segment=args.min_segment,
                max_segment=args.max_segment,
            )

            if cu_segments:
                # Build manifest from CU-derived segments
                from .core import Segment

                segments_for_manifest = [
                    Segment(id=s.id, start=s.start, end=s.end, duration=s.duration)
                    for s in cu_segments
                ]
                manifest = build_manifest(
                    args,
                    input_video=input_video,
                    output_video=output_video,
                    workdir=workdir,
                    duration=duration,
                    segments=segments_for_manifest,
                )
                # Enrich with CU metadata
                for cu_seg in cu_segments:
                    try:
                        m_seg = manifest.get_segment(cu_seg.id)
                        m_seg.title = cu_seg.title
                        m_seg.visible_points = cu_seg.visible_points
                        m_seg.on_screen_text = cu_seg.on_screen_text
                        m_seg.vision_result = cu_seg.vision_result
                        m_seg.frame_paths = cu_seg.frame_paths
                        m_seg.status = cu_seg.status
                    except KeyError:
                        pass

                manifest.status = "planned"
                manifest.save(manifest_path)

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
                    "cu_result": cu_result,
                    "use_content_understanding": use_cu,
                    "use_llm_critic": use_llm_critic,
                    "use_doc_intel": use_doc_intel,
                    "use_llm_profiler": use_llm_profiler,
                    "segments_total": len(manifest.segments),
                    "current_segment_index": 0,
                    "target_segment_ids": sorted(target_segment_ids) if target_segment_ids else None,
                    "redo": args.redo,
                }
        except Exception as exc:
            print(f"[warn] Content Understanding failed, falling back to ffmpeg: {exc}")
            cu_result = {}

    # Standard manifest creation (ffmpeg-based)
    manifest = load_or_create_manifest(
        args,
        input_video=input_video,
        output_video=output_video,
        workdir=workdir,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        manifest_path=manifest_path,
    )

    # Phase 4: LLM-based video profiler (if enabled)
    if use_llm_profiler and manifest.video_profile:
        try:
            from .vision_enricher import profile_video_with_llm

            sample_frames = _collect_sample_frames(manifest)
            if sample_frames:
                llm_profile = profile_video_with_llm(
                    sample_frames,
                    requested_scene_threshold=args.scene_threshold,
                    requested_min_segment=args.min_segment,
                    requested_max_segment=args.max_segment,
                    requested_base_rate=args.base_rate,
                    requested_azure_style=args.azure_style,
                )
                if llm_profile.confidence > manifest.video_profile.confidence:
                    manifest.video_profile = llm_profile
                    manifest.save(manifest_path)
        except Exception as exc:
            print(f"[warn] LLM profiler failed, keeping heuristic profile: {exc}")

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
        "cu_result": cu_result,
        "use_content_understanding": use_cu,
        "use_llm_critic": use_llm_critic,
        "use_doc_intel": use_doc_intel,
        "use_llm_profiler": use_llm_profiler,
        "segments_total": len(manifest.segments),
        "current_segment_index": 0,
        "target_segment_ids": sorted(target_segment_ids) if target_segment_ids else None,
        "redo": args.redo,
    }


def _collect_sample_frames(manifest: Manifest) -> list[Path]:
    """Collect up to 8 frame paths from accepted or extracted segments."""
    frames: list[Path] = []
    for segment in manifest.segments:
        for fp in segment.frame_paths:
            p = Path(fp)
            if p.exists():
                frames.append(p)
                if len(frames) >= 8:
                    return frames
    return frames


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
            # Feature flags (Phase 2/3/4)
            "use_content_understanding": state.get("use_content_understanding", False),
            "use_llm_critic": state.get("use_llm_critic", False),
            "use_doc_intel": state.get("use_doc_intel", False),
            "cu_segment_data": {},
            # Control flow
            "boundary_ok": True,
            "qa_passed": False,
            "duration_decision": "",
            "tts_retry_count": 0,
            "narration_retry_count": 0,
            "max_tts_retries": 1,
            "max_narration_retries": 2 if state.get("use_llm_critic", False) else 1,
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
