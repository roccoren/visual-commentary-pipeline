"""Tests for Phase 2/3/4: Content Understanding, LLM Critic, Vision Enricher."""

import json
from pathlib import Path

import pytest

from video_commentary.content_understanding import (
    _parse_mm_ss,
    _parse_markdown_segments,
    _parse_timestamp_ms,
    content_understanding_to_segments,
    extract_shot_boundaries,
    infer_video_type_from_cu,
    parse_cu_segments,
)
from video_commentary.llm_critic import (
    CriticIssue,
    CriticResult,
    RewriteResult,
    evaluate_narration_two_layer,
)
from video_commentary.state import Decision, SegmentState, SegmentStatus
from video_commentary.vision_enricher import (
    _parse_doc_intel_result,
    enrich_segment_vision,
)


# ===================================================================
# Phase 2: Content Understanding
# ===================================================================

class TestTimestampParsing:
    def test_parse_timestamp_ms(self):
        assert _parse_timestamp_ms(5000) == 5.0
        assert _parse_timestamp_ms(1500) == 1.5
        assert _parse_timestamp_ms(0) == 0.0

    def test_parse_mm_ss(self):
        assert _parse_mm_ss("00:06.000") == 6.0
        assert _parse_mm_ss("01:30.500") == 90.5
        assert _parse_mm_ss("00:00.000") == 0.0
        assert _parse_mm_ss("invalid") == 0.0


class TestParseMarkdownSegments:
    def test_parses_video_segments_from_markdown(self):
        markdown = (
            "# Video: 00:00.000 => 00:06.000\n"
            "A lively room filled with people.\n\n"
            "# Video: 00:06.000 => 00:10.080\n"
            "The scene transitions to a more vibrant setting.\n"
        )
        segments = _parse_markdown_segments(markdown, video_duration=10.08)
        assert len(segments) == 2
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 6.0
        assert segments[0]["duration"] == 6.0
        assert "lively room" in segments[0]["description"]
        assert segments[1]["start"] == 6.0
        assert segments[1]["end"] == 10.08

    def test_empty_markdown_returns_empty(self):
        assert _parse_markdown_segments("", video_duration=10.0) == []


class TestParseCUSegments:
    def test_parses_structured_json_segments(self):
        cu_result = {
            "contents": [
                {
                    "startTimeMs": 0,
                    "endTimeMs": 5000,
                    "fields": {
                        "title": {"value": "Introduction"},
                        "narration_zh": {"value": "这里展示了 Azure 门户。"},
                        "visible_points": {"values": [{"value": "Top nav"}, {"value": "Side menu"}]},
                        "on_screen_text": {"values": [{"value": "Azure Portal"}]},
                        "video_type_hint": {"value": "portal_walkthrough"},
                    },
                    "keyFrames": [
                        {"url": "https://example.com/frame1.jpg"},
                        {"path": "/tmp/frame2.jpg"},
                    ],
                },
                {
                    "startTimeMs": 5000,
                    "endTimeMs": 10000,
                    "fields": {
                        "title": {"value": "Configuration"},
                        "narration_zh": {"value": "接下来配置资源。"},
                    },
                },
            ]
        }

        segments = parse_cu_segments(cu_result, video_duration=10.0)
        assert len(segments) == 2
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 5.0
        assert segments[0]["title"] == "Introduction"
        assert segments[0]["narration_zh"] == "这里展示了 Azure 门户。"
        assert segments[0]["visible_points"] == ["Top nav", "Side menu"]
        assert segments[0]["on_screen_text"] == ["Azure Portal"]
        assert segments[0]["video_type_hint"] == "portal_walkthrough"
        assert len(segments[0]["keyframe_paths"]) == 2

    def test_handles_empty_result(self):
        assert parse_cu_segments({}, video_duration=10.0) == []

    def test_handles_markdown_string_content(self):
        cu_result = {
            "content": (
                "# Video: 00:00.000 => 00:05.000\n"
                "Showing Azure Portal.\n"
            )
        }
        segments = parse_cu_segments(cu_result, video_duration=5.0)
        assert len(segments) == 1
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 5.0


class TestContentUnderstandingToSegments:
    def test_converts_cu_result_to_segment_states(self):
        cu_result = {
            "contents": [
                {
                    "startTimeMs": 0,
                    "endTimeMs": 5000,
                    "fields": {
                        "title": {"value": "Intro"},
                        "narration_zh": {"value": "这里展示了首页。"},
                        "visible_points": {"values": []},
                        "on_screen_text": {"values": []},
                    },
                    "keyFrames": [{"url": "frame1.jpg"}],
                },
                {
                    "startTimeMs": 5000,
                    "endTimeMs": 12000,
                    "fields": {
                        "title": {"value": "Detail"},
                        "narration_zh": {"value": "接下来看细节。"},
                    },
                    "keyFrames": [{"url": "frame2.jpg"}],
                },
            ]
        }

        states = content_understanding_to_segments(
            cu_result, video_duration=12.0, min_segment=3.0, max_segment=12.0
        )

        assert len(states) >= 2
        assert all(isinstance(s, SegmentState) for s in states)
        # First segment should have CU metadata
        assert states[0].title == "Intro"
        assert states[0].status == SegmentStatus.FRAMES_EXTRACTED

    def test_empty_cu_result_returns_empty(self):
        assert content_understanding_to_segments({}, video_duration=10.0) == []


class TestShotBoundaries:
    def test_extract_shot_boundaries(self):
        cu_result = {
            "details": {
                "cameraShotTimesMs": [0, 3500, 7000, 12000]
            }
        }
        boundaries = extract_shot_boundaries(cu_result)
        assert boundaries == [0.0, 3.5, 7.0, 12.0]

    def test_empty_details(self):
        assert extract_shot_boundaries({}) == []


class TestInferVideoType:
    def test_infer_from_segment_hints(self):
        cu_result = {
            "contents": [
                {"fields": {"video_type_hint": {"value": "portal_walkthrough"}}},
                {"fields": {"video_type_hint": {"value": "portal_walkthrough"}}},
                {"fields": {"video_type_hint": {"value": "product_demo"}}},
            ]
        }
        assert infer_video_type_from_cu(cu_result) == "portal_walkthrough"

    def test_defaults_to_mixed(self):
        assert infer_video_type_from_cu({}) == "mixed_visual_demo"


# ===================================================================
# Phase 3: LLM Critic
# ===================================================================

class TestCriticModels:
    def test_critic_result_validation(self):
        result = CriticResult(
            passed=True,
            confidence=0.95,
            issues=[],
            overall_feedback="Good narration",
        )
        assert result.passed is True
        assert result.confidence == 0.95

    def test_critic_issue_model(self):
        issue = CriticIssue(
            category="accuracy",
            description="Mentions Azure but screen shows AWS",
            suggestion="Change Azure to AWS",
            severity="high",
        )
        assert issue.severity == "high"

    def test_rewrite_result_model(self):
        result = RewriteResult(
            narration_zh="改写后的旁白。",
            changes_made=["修正了术语"],
            confidence=0.8,
        )
        assert result.narration_zh == "改写后的旁白。"


class TestTwoLayerQA:
    def test_rule_based_catches_empty_narration(self):
        segment = SegmentState(id=1, start=0.0, end=5.0, duration=5.0)
        passed, decision, reason, feedback, critic = evaluate_narration_two_layer(
            narration="   ",
            segment=segment,
            previous_narration="上一段",
            use_llm=False,
        )
        assert passed is False
        assert decision == Decision.RETRY_NARRATION
        assert "empty narration" in feedback
        assert critic is None

    def test_rule_based_catches_repetitive_narration(self):
        segment = SegmentState(id=2, start=5.0, end=10.0, duration=5.0)
        passed, decision, reason, feedback, critic = evaluate_narration_two_layer(
            narration="这里展示Azure门户。",
            segment=segment,
            previous_narration="这里展示 Azure 门户。",
            use_llm=False,
        )
        assert passed is False
        assert decision == Decision.RETRY_NARRATION

    def test_accepts_good_narration_without_llm(self):
        segment = SegmentState(id=1, start=0.0, end=5.0, duration=5.0)
        passed, decision, reason, feedback, critic = evaluate_narration_two_layer(
            narration="这里展示了 Azure 门户的核心功能。",
            segment=segment,
            previous_narration="",
            use_llm=False,
        )
        assert passed is True
        assert decision == Decision.ACCEPT
        assert critic is None

    def test_llm_layer_graceful_degradation(self):
        """When LLM is requested but credentials aren't set, falls back gracefully."""
        segment = SegmentState(
            id=1, start=0.0, end=5.0, duration=5.0,
            visible_points=["menu"],
            on_screen_text=["Settings"],
        )
        passed, decision, reason, feedback, critic = evaluate_narration_two_layer(
            narration="这里展示了设置页面的核心功能。",
            segment=segment,
            previous_narration="",
            use_llm=True,
        )
        # Should pass (LLM unavailable returns passed=True with confidence=0.0)
        assert passed is True
        assert decision == Decision.ACCEPT


# ===================================================================
# Phase 4: Vision Enricher
# ===================================================================

class TestDocIntelParsing:
    def test_parse_doc_intel_result(self):
        analyze_result = {
            "content": "Azure Portal\nSettings > Configuration",
            "pages": [
                {
                    "lines": [
                        {"content": "Azure Portal", "confidence": 0.99},
                        {"content": "Settings > Configuration", "confidence": 0.95},
                    ],
                    "words": [
                        {"content": "Azure", "confidence": 0.99},
                        {"content": "Portal", "confidence": 0.99},
                    ],
                }
            ],
        }
        result = _parse_doc_intel_result(analyze_result)
        assert result["text"] == "Azure Portal\nSettings > Configuration"
        assert len(result["lines"]) == 2
        assert result["lines"][0]["text"] == "Azure Portal"
        assert result["lines"][0]["confidence"] == 0.99

    def test_empty_result(self):
        result = _parse_doc_intel_result({})
        assert result["text"] == ""
        assert result["lines"] == []


class TestVisionEnricher:
    def test_enriches_with_cu_data_only(self):
        segment = SegmentState(
            id=1, start=0.0, end=5.0, duration=5.0,
            visible_points=[], on_screen_text=[],
        )
        cu_data = {
            "title": "CU Title",
            "description": "CU Description",
            "narration_zh": "CU 旁白内容。",
            "visible_points": ["CU point 1", "CU point 2"],
            "on_screen_text": ["CU text 1"],
        }

        result = enrich_segment_vision(
            segment,
            cu_data=cu_data,
            use_doc_intel=False,
            use_gpt4o_vision=False,
        )

        assert result["title"] == "CU Title"
        assert result["description"] == "CU Description"
        assert result["narration_zh"] == "CU 旁白内容。"
        assert "CU point 1" in result["visible_points"]
        assert "content_understanding" in result["sources"]

    def test_enriches_without_any_external_data(self):
        segment = SegmentState(
            id=1, start=0.0, end=5.0, duration=5.0,
            title="Existing Title",
            visible_points=["existing_vp"],
            on_screen_text=["existing_ost"],
        )

        result = enrich_segment_vision(
            segment,
            cu_data=None,
            use_doc_intel=False,
            use_gpt4o_vision=False,
        )

        assert result["title"] == "Existing Title"
        assert result["visible_points"] == ["existing_vp"]
        assert result["sources"] == []

    def test_cu_data_fills_empty_fields(self):
        segment = SegmentState(
            id=1, start=0.0, end=5.0, duration=5.0,
            title="", visible_points=[], on_screen_text=[],
        )

        cu_data = {
            "title": "From CU",
            "visible_points": ["vp1"],
            "on_screen_text": ["ost1"],
            "narration_zh": "CU narration",
            "description": "CU desc",
        }

        result = enrich_segment_vision(
            segment,
            cu_data=cu_data,
            use_doc_intel=False,
            use_gpt4o_vision=False,
        )

        assert result["title"] == "From CU"
        assert result["visible_points"] == ["vp1"]
        assert result["on_screen_text"] == ["ost1"]
        assert result["narration_zh"] == "CU narration"


# ===================================================================
# Enhanced graph integration tests
# ===================================================================

class TestEnhancedGraphNodes:
    def test_vision_step_without_enhancements(self, tmp_path: Path, monkeypatch):
        """Vision step uses original path when no enhancements enabled."""
        from video_commentary.graph import vision_step, _save_manifest
        from video_commentary.state import Manifest

        segment = SegmentState(
            id=1, start=0.0, end=5.0, duration=5.0,
            status=SegmentStatus.FRAMES_EXTRACTED,
            frame_paths=["fake.jpg"],
        )
        manifest = Manifest(
            version="2.0", input_video="in.mp4", output_video="out.mp4",
            workdir=str(tmp_path), scene_threshold=0.32, min_segment=3.0,
            max_segment=12.0, segment_buffer=0.35, base_rate="+0%",
            azure_style="professional", duration=20.0, segments=[segment],
        )
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest.save(manifest_path)

        monkeypatch.setattr(
            "video_commentary.pipeline.call_azure_openai_vision",
            lambda **kw: {
                "title": "Mock",
                "visible_points": ["p1"],
                "on_screen_text": ["t1"],
                "narration_zh": "模拟内容。",
            },
        )

        state = {
            "segment_id": 1,
            "manifest_dict": manifest.to_dict(),
            "manifest_path": str(manifest_path),
            "use_content_understanding": False,
            "use_doc_intel": False,
        }

        result = vision_step(state)
        assert "manifest_dict" in result

    def test_qa_gate_step_without_llm_critic(self, tmp_path: Path):
        """QA step uses original rule-based gate when LLM critic disabled."""
        from video_commentary.graph import qa_gate_step
        from video_commentary.state import Manifest

        segment = SegmentState(
            id=1, start=0.0, end=5.0, duration=5.0,
            status=SegmentStatus.CONTENT_GENERATED,
            selected_draft="这里展示了 Azure 门户的核心功能。",
        )
        manifest = Manifest(
            version="2.0", input_video="in.mp4", output_video="out.mp4",
            workdir=str(tmp_path), scene_threshold=0.32, min_segment=3.0,
            max_segment=12.0, segment_buffer=0.35, base_rate="+0%",
            azure_style="professional", duration=20.0, segments=[segment],
        )
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest.save(manifest_path)

        state = {
            "segment_id": 1,
            "manifest_dict": manifest.to_dict(),
            "manifest_path": str(manifest_path),
            "use_llm_critic": False,
        }

        result = qa_gate_step(state)
        assert result["qa_passed"] is True

    def test_qa_gate_step_with_llm_critic_graceful_degradation(self, tmp_path: Path):
        """QA step with LLM critic passes when LLM is unavailable (no credentials)."""
        from video_commentary.graph import qa_gate_step
        from video_commentary.state import Manifest

        segment = SegmentState(
            id=1, start=0.0, end=5.0, duration=5.0,
            status=SegmentStatus.CONTENT_GENERATED,
            selected_draft="这里展示了 Azure 门户的核心功能。",
        )
        manifest = Manifest(
            version="2.0", input_video="in.mp4", output_video="out.mp4",
            workdir=str(tmp_path), scene_threshold=0.32, min_segment=3.0,
            max_segment=12.0, segment_buffer=0.35, base_rate="+0%",
            azure_style="professional", duration=20.0, segments=[segment],
        )
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest.save(manifest_path)

        state = {
            "segment_id": 1,
            "manifest_dict": manifest.to_dict(),
            "manifest_path": str(manifest_path),
            "use_llm_critic": True,
            "narration_retry_count": 0,
            "max_narration_retries": 2,
        }

        result = qa_gate_step(state)
        # Should pass — rule-based passes, LLM degrades gracefully
        assert result["qa_passed"] is True


class TestCLIFlags:
    def test_new_cli_flags_are_parsed(self):
        from video_commentary.pipeline import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args([
            "--input", "in.mp4",
            "--output", "out.mp4",
            "--workdir", "work",
            "--graph",
            "--use-content-understanding",
            "--use-llm-critic",
            "--use-doc-intel",
            "--use-llm-profiler",
        ])
        assert args.graph is True
        assert args.use_content_understanding is True
        assert args.use_llm_critic is True
        assert args.use_doc_intel is True
        assert args.use_llm_profiler is True

    def test_flags_default_to_false(self):
        from video_commentary.pipeline import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args([
            "--input", "in.mp4",
            "--output", "out.mp4",
            "--workdir", "work",
        ])
        assert args.graph is False
        assert args.use_content_understanding is False
        assert args.use_llm_critic is False
        assert args.use_doc_intel is False
        assert args.use_llm_profiler is False
