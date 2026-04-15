"""Tests for the LangGraph orchestration layer."""

import json
from pathlib import Path

import pytest

from video_commentary.graph import (
    args_to_dict,
    build_pipeline_graph,
    build_segment_graph,
    narrate_video_graph,
    boundary_step,
    frame_extraction_step,
    vision_step,
    narration_step,
    qa_gate_step,
    tts_step,
    duration_gate_step,
    route_after_boundary,
    route_after_qa,
    route_after_duration,
)
from video_commentary.graph_state import PipelineState, SegmentProcessingState
from video_commentary.pipeline import build_arg_parser
from video_commentary.planner import VideoProfile
from video_commentary.state import Decision, Manifest, SegmentState, SegmentStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest(tmp_path: Path, segments: list[SegmentState] | None = None) -> Manifest:
    return Manifest(
        version="2.0",
        input_video=str(tmp_path / "in.mp4"),
        output_video=str(tmp_path / "out.mp4"),
        workdir=str(tmp_path),
        scene_threshold=0.32,
        min_segment=3.0,
        max_segment=12.0,
        segment_buffer=0.35,
        base_rate="+0%",
        azure_style="professional",
        duration=20.0,
        segments=segments or [],
    )


def _make_seg_state(manifest: Manifest, segment: SegmentState, tmp_path: Path) -> SegmentProcessingState:
    manifest_path = tmp_path / "commentary_manifest.json"
    return {
        "segment_id": segment.id,
        "manifest_dict": manifest.to_dict(),
        "manifest_path": str(manifest_path),
        "input_video": str(tmp_path / "in.mp4"),
        "workdir": str(tmp_path),
        "frames_dir": str(tmp_path / "frames"),
        "raw_audio_dir": str(tmp_path / "tts_raw"),
        "fit_audio_dir": str(tmp_path / "tts_fit"),
        "ffmpeg_bin": "ffmpeg",
        "ffprobe_bin": "ffprobe",
        "segment_buffer": 0.35,
        "base_rate": "+0%",
        "azure_style": "professional",
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


# ---------------------------------------------------------------------------
# Graph construction tests
# ---------------------------------------------------------------------------

class TestGraphConstruction:
    def test_segment_graph_compiles(self):
        graph = build_segment_graph().compile()
        assert graph is not None

    def test_pipeline_graph_compiles(self):
        graph = build_pipeline_graph().compile()
        assert graph is not None

    def test_segment_graph_has_expected_nodes(self):
        builder = build_segment_graph()
        node_names = set(builder.nodes.keys())
        expected = {"boundary", "frame_extraction", "vision", "narration", "qa_gate", "tts", "duration_gate"}
        assert expected == node_names

    def test_pipeline_graph_has_expected_nodes(self):
        builder = build_pipeline_graph()
        node_names = set(builder.nodes.keys())
        expected = {"init", "process_segments", "finalize"}
        assert expected == node_names


# ---------------------------------------------------------------------------
# Routing logic tests
# ---------------------------------------------------------------------------

class TestRouting:
    def test_route_after_boundary_accept(self):
        assert route_after_boundary({"boundary_ok": True}) == "frame_extraction"

    def test_route_after_boundary_skip(self):
        assert route_after_boundary({"boundary_ok": False}) == "__end__"

    def test_route_after_boundary_default_is_accept(self):
        assert route_after_boundary({}) == "frame_extraction"

    def test_route_after_qa_pass(self):
        assert route_after_qa({"qa_passed": True}) == "tts"

    def test_route_after_qa_fail(self):
        assert route_after_qa({"qa_passed": False}) == "__end__"

    def test_route_after_duration_always_ends(self):
        assert route_after_duration({"duration_decision": "accept"}) == "__end__"
        assert route_after_duration({"duration_decision": "retry_tts"}) == "__end__"


# ---------------------------------------------------------------------------
# Boundary step tests
# ---------------------------------------------------------------------------

class TestBoundaryStep:
    def test_skip_tiny_segment(self, tmp_path: Path):
        segment = SegmentState(id=1, start=0.0, end=0.5, duration=0.5)
        manifest = _make_manifest(tmp_path, segments=[segment])
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest.save(manifest_path)

        state = _make_seg_state(manifest, segment, tmp_path)
        state["manifest_path"] = str(manifest_path)
        result = boundary_step(state)

        assert result["boundary_ok"] is False
        assert result["segment_status"] == "skipped"

    def test_accept_normal_segment(self, tmp_path: Path):
        segment = SegmentState(id=1, start=0.0, end=5.0, duration=5.0)
        manifest = _make_manifest(tmp_path, segments=[segment])
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest.save(manifest_path)

        state = _make_seg_state(manifest, segment, tmp_path)
        state["manifest_path"] = str(manifest_path)
        result = boundary_step(state)

        assert result["boundary_ok"] is True

    def test_merge_short_segment_with_previous(self, tmp_path: Path):
        prev = SegmentState(id=1, start=0.0, end=5.0, duration=5.0)
        segment = SegmentState(id=2, start=5.0, end=6.2, duration=1.2)
        manifest = _make_manifest(tmp_path, segments=[prev, segment])
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest.save(manifest_path)

        state = _make_seg_state(manifest, segment, tmp_path)
        state["manifest_path"] = str(manifest_path)
        result = boundary_step(state)

        assert result["boundary_ok"] is False
        assert result["segment_status"] == "needs_human_review"


# ---------------------------------------------------------------------------
# QA gate step tests
# ---------------------------------------------------------------------------

class TestQAGateStep:
    def test_qa_passes_good_narration(self, tmp_path: Path):
        segment = SegmentState(
            id=1, start=0.0, end=5.0, duration=5.0,
            status=SegmentStatus.CONTENT_GENERATED,
            selected_draft="这里展示了 Azure 门户的核心功能。",
        )
        manifest = _make_manifest(tmp_path, segments=[segment])
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest.save(manifest_path)

        state = _make_seg_state(manifest, segment, tmp_path)
        state["manifest_path"] = str(manifest_path)
        result = qa_gate_step(state)

        assert result["qa_passed"] is True

    def test_qa_fails_empty_narration_and_rewrites(self, tmp_path: Path):
        segment = SegmentState(
            id=1, start=0.0, end=5.0, duration=5.0,
            status=SegmentStatus.CONTENT_GENERATED,
            selected_draft="   ",
        )
        manifest = _make_manifest(tmp_path, segments=[segment])
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest.save(manifest_path)

        state = _make_seg_state(manifest, segment, tmp_path)
        state["manifest_path"] = str(manifest_path)
        result = qa_gate_step(state)

        # Auto-rewrite should have made it pass
        assert result["qa_passed"] is True


# ---------------------------------------------------------------------------
# Segment subgraph integration test (with mocked external calls)
# ---------------------------------------------------------------------------

class TestSegmentSubgraph:
    def test_boundary_skip_ends_early(self, tmp_path: Path):
        """Tiny segment should be skipped by the boundary check and end the subgraph."""
        segment = SegmentState(id=1, start=0.0, end=0.5, duration=0.5)
        manifest = _make_manifest(tmp_path, segments=[segment])
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest.save(manifest_path)

        graph = build_segment_graph().compile()
        state = _make_seg_state(manifest, segment, tmp_path)
        state["manifest_path"] = str(manifest_path)

        result = graph.invoke(state)

        assert result["boundary_ok"] is False
        assert result["segment_status"] == "skipped"

    def test_boundary_accept_reaches_frame_extraction_but_skips_pending(self, tmp_path: Path, monkeypatch):
        """Normal segment passes boundary and reaches frame_extraction.
        Since we don't have real ffmpeg/Azure, we mock the underlying pipeline calls.
        """
        segment = SegmentState(
            id=1, start=0.0, end=5.0, duration=5.0,
            status=SegmentStatus.PENDING,
        )
        manifest = _make_manifest(tmp_path, segments=[segment])
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest.save(manifest_path)

        # Mock all external pipeline calls
        monkeypatch.setattr("video_commentary.pipeline.extract_frame", lambda *a, **kw: None)
        monkeypatch.setattr(
            "video_commentary.pipeline.call_azure_openai_vision",
            lambda **kw: {
                "title": "Mock Title",
                "visible_points": ["point1"],
                "on_screen_text": ["text1"],
                "narration_zh": "这里展示了模拟内容。",
            },
        )
        monkeypatch.setattr("video_commentary.pipeline.synthesize_azure_tts", lambda *a, **kw: None)
        monkeypatch.setattr("video_commentary.pipeline.fit_audio_to_budget", lambda *a, **kw: 4.5)
        monkeypatch.setattr("video_commentary.pipeline.ffprobe_duration", lambda *a, **kw: 4.5)

        # Create fake frame files so frame_paths are valid
        frames_dir = tmp_path / "frames" / "seg_001"
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "frame_01.jpg").write_bytes(b"fake")

        graph = build_segment_graph().compile()

        state = _make_seg_state(manifest, segment, tmp_path)

        result = graph.invoke(state)

        # Should have passed through all stages and been accepted by duration gate
        result_manifest = Manifest.from_dict(result["manifest_dict"])
        seg = result_manifest.get_segment(1)
        assert seg.status == SegmentStatus.ACCEPTED
        assert seg.selected_draft
        assert result["segment_status"] == "accepted"


# ---------------------------------------------------------------------------
# args_to_dict round-trip
# ---------------------------------------------------------------------------

class TestArgsRoundTrip:
    def test_args_serialization(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--input", "in.mp4",
            "--output", "out.mp4",
            "--workdir", "work",
            "--scene-threshold", "0.36",
            "--base-rate", "+5%",
        ])

        d = args_to_dict(args)
        assert d["input"] == "in.mp4"
        assert d["output"] == "out.mp4"
        assert d["scene_threshold"] == 0.36
        assert d["base_rate"] == "+5%"

        from video_commentary.graph import _args_from_dict
        restored = _args_from_dict(d)
        assert restored.input == "in.mp4"
        assert restored.scene_threshold == 0.36


# ---------------------------------------------------------------------------
# Pipeline graph integration test (fully mocked)
# ---------------------------------------------------------------------------

class TestPipelineGraph:
    def test_full_pipeline_graph_with_mocked_externals(self, tmp_path: Path, monkeypatch):
        """End-to-end pipeline graph test with all external calls mocked."""
        from video_commentary.core import Segment

        input_video = tmp_path / "in.mp4"
        input_video.write_bytes(b"fake")
        output_video = tmp_path / "out.mp4"

        # Mock all external dependencies
        monkeypatch.setattr("video_commentary.pipeline.find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr("video_commentary.pipeline.find_ffprobe", lambda _: "ffprobe")
        monkeypatch.setattr("video_commentary.pipeline.ffprobe_duration", lambda *a, **kw: 10.0)
        monkeypatch.setattr("video_commentary.pipeline.detect_scene_cuts", lambda *a, **kw: [5.0])
        monkeypatch.setattr(
            "video_commentary.pipeline.build_segments",
            lambda duration, cuts, min_segment, max_segment: [
                Segment(id=1, start=0.0, end=5.0, duration=5.0),
                Segment(id=2, start=5.0, end=10.0, duration=5.0),
            ],
        )

        # Mock vision and TTS so segments get processed without real APIs
        monkeypatch.setattr(
            "video_commentary.pipeline.call_azure_openai_vision",
            lambda **kw: {
                "title": "Test",
                "visible_points": ["p1"],
                "on_screen_text": ["t1"],
                "narration_zh": "这是测试内容。",
            },
        )
        monkeypatch.setattr("video_commentary.pipeline.extract_frame", lambda *a, **kw: None)
        monkeypatch.setattr("video_commentary.pipeline.synthesize_azure_tts", lambda *a, **kw: None)
        monkeypatch.setattr("video_commentary.pipeline.fit_audio_to_budget", lambda *a, **kw: 4.5)
        monkeypatch.setattr("video_commentary.pipeline.ffprobe_duration", lambda *a, **kw: 4.5)
        monkeypatch.setattr("video_commentary.pipeline.compose_commentary_track", lambda *a, **kw: None)
        monkeypatch.setattr("video_commentary.pipeline.mux_video", lambda *a, **kw: None)
        monkeypatch.setattr(
            "video_commentary.pipeline.write_srt_file",
            lambda narrations, out_path: out_path.write_text("fake srt", encoding="utf-8"),
        )

        args = build_arg_parser().parse_args([
            "--input", str(input_video),
            "--output", str(output_video),
            "--workdir", str(tmp_path / "work"),
        ])

        result = narrate_video_graph(args)

        assert result["segments"] >= 0
        assert "output_video" in result
