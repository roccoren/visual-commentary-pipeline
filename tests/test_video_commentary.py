"""Tests for the video commentary engineering helpers."""

import json
from pathlib import Path

from video_commentary.core import (
    DEFAULT_TERM_MAP,
    Segment,
    atempo_chain,
    build_azure_tts_ssml,
    build_segments,
    build_srt_text,
    format_srt_timestamp,
    normalize_terms,
    sample_times,
)
from video_commentary.duration_policy import decide_duration_action
from video_commentary.pipeline import (
    accepted_narrations_from_manifest,
    accepted_segments_from_manifest,
    azure_openai_uses_responses_api,
    build_azure_openai_vision_request,
    build_manifest,
    build_arg_parser,
    extract_azure_openai_text,
    get_effective_segmentation_policy,
    get_effective_style_policy,
    get_previous_accepted_narration,
    get_previous_segment,
    load_or_create_manifest,
    maybe_apply_redo,
    note_segment_decision,
    parse_segment_id_list,
    resolve_target_segment_ids,
    reset_segment_for_redo,
    rewrite_narration_once,
    run_qa_gate_step,
    segment_state_to_narration,
    should_process_segment,
    validate_target_segment_ids,
)
from video_commentary.planner import VideoProfile, normalize_video_profile, plan_video_profile
from video_commentary.qa_gate import evaluate_narration_quality
from video_commentary.segment_policy import decide_segment_action
from video_commentary.state import Decision, Manifest, RetryEntry, SegmentState, SegmentStatus


class TestBuildSegments:
    def test_merges_short_segments_into_previous_and_splits_long_ones(self):
        segments = build_segments(
            duration=30.0,
            cuts=[2.0, 5.0, 21.0],
            min_segment=3.0,
            max_segment=8.0,
        )

        assert [(s.start, s.end, s.duration) for s in segments] == [
            (0.0, 5.0, 5.0),
            (5.0, 13.0, 8.0),
            (13.0, 21.0, 8.0),
            (21.0, 25.5, 4.5),
            (25.5, 30.0, 4.5),
        ]

    def test_ignores_duplicate_and_tiny_boundaries(self):
        segments = build_segments(
            duration=10.0,
            cuts=[0.0, 0.01, 4.0, 4.0, 9.98],
            min_segment=1.0,
            max_segment=10.0,
        )

        assert [(s.start, s.end, s.duration) for s in segments] == [
            (0.0, 4.0, 4.0),
            (4.0, 10.0, 6.0),
        ]


class TestSampleTimes:
    def test_short_segment_uses_midpoint(self):
        segment = Segment(id=1, start=4.0, end=7.0, duration=3.0)
        assert sample_times(segment) == [5.5]

    def test_longer_segment_uses_three_samples(self):
        segment = Segment(id=2, start=10.0, end=20.0, duration=10.0)
        assert sample_times(segment) == [11.2, 15.0, 18.8]


class TestNormalizeTerms:
    def test_applies_default_glossary_and_punctuation_spacing_cleanup(self):
        text = "ashure  openaai ， deep seek !"
        assert normalize_terms(text, DEFAULT_TERM_MAP) == "Azure OpenAI， DeepSeek!"

    def test_accepts_custom_term_map(self):
        assert normalize_terms("foo demo", {"foo": "Bar"}) == "Bar demo"


class TestTempoChain:
    def test_single_stage(self):
        assert atempo_chain(1.25) == "atempo=1.25000"

    def test_multi_stage_for_large_speedup(self):
        assert atempo_chain(4.0) == "atempo=2.0,atempo=2.0"


class TestSrtHelpers:
    def test_format_srt_timestamp(self):
        assert format_srt_timestamp(3661.275) == "01:01:01,275"

    def test_build_srt_text(self):
        items = [
            {"start": 0.0, "end": 2.5, "narration_zh": "第一页概览。"},
            {"start": 2.5, "end": 5.0, "narration_zh": "这里进入细节。"},
        ]

        assert build_srt_text(items) == (
            "1\n"
            "00:00:00,000 --> 00:00:02,500\n"
            "第一页概览。\n\n"
            "2\n"
            "00:00:02,500 --> 00:00:05,000\n"
            "这里进入细节。\n"
        )


class TestAzureSsml:
    def test_builds_ssml_with_optional_duration_and_escaping(self):
        ssml = build_azure_tts_ssml(
            text="AT&T <Azure>",
            voice="zh-CN-XiaoxiaoNeural",
            rate="+5%",
            target_duration_ms=18000,
            style="professional",
        )

        assert 'mstts:audioduration value="18000ms"' in ssml
        assert '<prosody rate="+5%">AT&amp;T &lt;Azure&gt;</prosody>' in ssml
        assert 'style="professional"' in ssml
        assert 'xmlns:mstts="https://www.w3.org/2001/mstts"' in ssml

    def test_omits_optional_nodes_when_not_requested(self):
        ssml = build_azure_tts_ssml(
            text="讲解内容",
            voice="zh-CN-XiaoxiaoNeural",
        )

        assert "mstts:audioduration" not in ssml
        assert "mstts:express-as" not in ssml
        assert '<prosody rate="+0%">讲解内容</prosody>' in ssml


class TestAzureOpenAIRequestRouting:
    def test_uses_responses_api_for_2025_preview(self, tmp_path: Path):
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"fake-jpeg")

        url, payload, api_kind = build_azure_openai_vision_request(
            endpoint="https://example.openai.azure.com",
            deployment="gpt-5.4-nano",
            api_version="2025-04-01-preview",
            user_prompt="describe the frame",
            frame_paths=[frame],
        )

        assert azure_openai_uses_responses_api("2025-04-01-preview") is True
        assert api_kind == "responses"
        assert url == "https://example.openai.azure.com/openai/responses?api-version=2025-04-01-preview"
        assert payload["model"] == "gpt-5.4-nano"
        assert payload["input"][0]["content"][0] == {"type": "input_text", "text": "describe the frame"}
        assert payload["input"][0]["content"][1]["type"] == "input_image"

    def test_uses_chat_completions_for_2024_api(self, tmp_path: Path):
        frame = tmp_path / "frame.png"
        frame.write_bytes(b"fake-png")

        url, payload, api_kind = build_azure_openai_vision_request(
            endpoint="https://example.openai.azure.com",
            deployment="gpt-4o-mini",
            api_version="2024-10-21",
            user_prompt="describe the frame",
            frame_paths=[frame],
        )

        assert azure_openai_uses_responses_api("2024-10-21") is False
        assert api_kind == "chat_completions"
        assert url == (
            "https://example.openai.azure.com/openai/deployments/gpt-4o-mini/"
            "chat/completions?api-version=2024-10-21"
        )
        assert payload["messages"][1]["content"][0] == {"type": "text", "text": "describe the frame"}
        assert payload["messages"][1]["content"][1]["type"] == "image_url"


class TestAzureOpenAIResponseParsing:
    def test_extracts_output_text_from_responses_api(self):
        body = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"narration_zh": "这里展示了 Azure 门户。"}',
                        }
                    ],
                }
            ]
        }

        assert extract_azure_openai_text(body, "responses") == '{"narration_zh": "这里展示了 Azure 门户。"}'

    def test_extracts_message_content_from_chat_completions(self):
        body = {
            "choices": [
                {
                    "message": {
                        "content": '{"narration_zh": "这里继续演示配置过程。"}'
                    }
                }
            ]
        }

        assert extract_azure_openai_text(body, "chat_completions") == '{"narration_zh": "这里继续演示配置过程。"}'


class TestVideoPlanner:
    def test_video_profile_round_trip_through_manifest(self, tmp_path: Path):
        profile = VideoProfile(
            video_type="portal_walkthrough",
            confidence=0.82,
            segmentation_policy={"scene_threshold": 0.36, "min_segment": 2.5, "max_segment": 9.0},
            style_policy={
                "narration_density": "balanced",
                "narration_focus": "operation_step",
                "azure_style": "professional",
                "base_rate": "+2%",
            },
            rationale=["screen is portal-like"],
        )
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir=str(tmp_path),
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=20.0,
            video_profile=profile,
            segments=[],
        )

        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = Manifest.load(path)

        assert loaded.video_profile is not None
        assert loaded.video_profile.video_type == "portal_walkthrough"
        assert loaded.video_profile.style_policy["base_rate"] == "+2%"

    def test_normalize_video_profile_fallback_whitelist_and_clamp(self):
        profile = normalize_video_profile(
            {
                "video_type": "unknown_type",
                "confidence": 2.4,
                "segmentation_policy": {"scene_threshold": 0.9, "min_segment": 0.2, "max_segment": 0.5},
                "style_policy": {
                    "narration_density": "verbose",
                    "narration_focus": "everything",
                    "azure_style": "professional",
                    "base_rate": "+99%",
                },
            },
            requested_scene_threshold=0.32,
            requested_min_segment=3.0,
            requested_max_segment=12.0,
            requested_base_rate="+0%",
            requested_azure_style="professional",
        )

        assert profile.video_type == "mixed_visual_demo"
        assert profile.confidence == 1.0
        assert profile.segmentation_policy["scene_threshold"] == 0.5
        assert profile.segmentation_policy["min_segment"] == 1.0
        assert profile.segmentation_policy["max_segment"] >= profile.segmentation_policy["min_segment"]
        assert profile.style_policy["narration_density"] == "balanced"
        assert profile.style_policy["narration_focus"] == "screen_change"
        assert profile.style_policy["base_rate"] == "+25%"

    def test_plan_video_profile_infers_known_video_type(self):
        profile = plan_video_profile(
            input_video=Path("azure-portal-demo.mp4"),
            requested_scene_threshold=0.32,
            requested_min_segment=3.0,
            requested_max_segment=12.0,
            requested_base_rate="+0%",
            requested_azure_style="professional",
        )

        assert profile.video_type == "portal_walkthrough"
        assert profile.style_policy["narration_focus"] == "operation_step"

    def test_effective_policy_helpers_read_from_manifest_video_profile(self):
        profile = VideoProfile(
            video_type="portal_walkthrough",
            confidence=0.8,
            segmentation_policy={"scene_threshold": 0.36, "min_segment": 2.5, "max_segment": 9.0},
            style_policy={
                "narration_density": "balanced",
                "narration_focus": "operation_step",
                "azure_style": "professional",
                "base_rate": "+2%",
            },
            rationale=[],
        )
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir="/tmp/work",
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="calm",
            duration=20.0,
            video_profile=profile,
            segments=[],
        )

        assert get_effective_segmentation_policy(manifest) == {
            "scene_threshold": 0.36,
            "min_segment": 2.5,
            "max_segment": 9.0,
        }
        assert get_effective_style_policy(manifest)["base_rate"] == "+2%"
        assert get_effective_style_policy(manifest)["azure_style"] == "professional"

    def test_load_or_create_manifest_reuses_existing_profile_on_resume(self, tmp_path: Path, monkeypatch):
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir=str(tmp_path),
            scene_threshold=0.36,
            min_segment=2.5,
            max_segment=9.0,
            segment_buffer=0.35,
            base_rate="+2%",
            azure_style="professional",
            duration=20.0,
            video_profile=VideoProfile(
                video_type="portal_walkthrough",
                confidence=0.8,
                segmentation_policy={"scene_threshold": 0.36, "min_segment": 2.5, "max_segment": 9.0},
                style_policy={
                    "narration_density": "balanced",
                    "narration_focus": "operation_step",
                    "azure_style": "professional",
                    "base_rate": "+2%",
                },
                rationale=["saved from previous run"],
            ),
            segments=[],
        )
        manifest.save(manifest_path)

        class Args:
            resume_from_manifest = None
            force_replan = False
            scene_threshold = 0.32
            min_segment = 3.0
            max_segment = 12.0
            segment_buffer = 0.35
            base_rate = "+0%"
            azure_style = "calm"

        def should_not_plan(**kwargs):
            raise AssertionError("planner should not rerun when manifest already has video_profile")

        monkeypatch.setattr("video_commentary.pipeline.plan_video_profile", should_not_plan)

        loaded = load_or_create_manifest(
            Args(),
            input_video=tmp_path / "input.mp4",
            output_video=tmp_path / "output.mp4",
            workdir=tmp_path,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            manifest_path=manifest_path,
        )

        assert loaded.video_profile is not None
        assert loaded.video_profile.video_type == "portal_walkthrough"
        assert loaded.video_profile.style_policy["base_rate"] == "+2%"

    def test_load_or_create_manifest_applies_planner_policy_to_manifest_defaults(self, tmp_path: Path, monkeypatch):
        input_video = tmp_path / "azure-portal-demo.mp4"
        output_video = tmp_path / "output.mp4"
        manifest_path = tmp_path / "commentary_manifest.json"
        input_video.write_bytes(b"fake")

        class Args:
            resume_from_manifest = None
            force_replan = False
            scene_threshold = 0.32
            min_segment = 3.0
            max_segment = 12.0
            segment_buffer = 0.35
            base_rate = "+0%"
            azure_style = "calm"

        monkeypatch.setattr("video_commentary.pipeline.ffprobe_duration", lambda *_args, **_kwargs: 20.0)
        monkeypatch.setattr("video_commentary.pipeline.detect_scene_cuts", lambda *_args, **_kwargs: [5.0, 10.0])
        monkeypatch.setattr(
            "video_commentary.pipeline.build_segments",
            lambda duration, cuts, min_segment, max_segment: [
                Segment(id=1, start=0.0, end=5.0, duration=5.0),
                Segment(id=2, start=5.0, end=10.0, duration=5.0),
            ],
        )
        monkeypatch.setattr(
            "video_commentary.pipeline.plan_video_profile",
            lambda **_kwargs: VideoProfile(
                video_type="portal_walkthrough",
                confidence=0.8,
                segmentation_policy={"scene_threshold": 0.36, "min_segment": 2.5, "max_segment": 9.0},
                style_policy={
                    "narration_density": "balanced",
                    "narration_focus": "operation_step",
                    "azure_style": "professional",
                    "base_rate": "+2%",
                },
                rationale=["planner test profile"],
            ),
        )

        loaded = load_or_create_manifest(
            Args(),
            input_video=input_video,
            output_video=output_video,
            workdir=tmp_path,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            manifest_path=manifest_path,
        )

        assert loaded.video_profile is not None
        assert loaded.scene_threshold == 0.36
        assert loaded.min_segment == 2.5
        assert loaded.max_segment == 9.0
        assert loaded.base_rate == "+2%"
        assert loaded.azure_style == "professional"


class TestStateModels:
    def test_manifest_round_trip(self, tmp_path: Path):
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir=str(tmp_path),
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=20.0,
            segments=[
                SegmentState(
                    id=1,
                    start=0.0,
                    end=5.0,
                    duration=5.0,
                    status=SegmentStatus.ACCEPTED,
                    selected_draft="第一页介绍。",
                    decision=Decision.ACCEPT,
                    retry_history=[RetryEntry(action=Decision.ACCEPT, reason="ok")],
                )
            ],
        )

        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = Manifest.load(path)

        assert loaded.version == "2.0"
        assert loaded.segments[0].status == SegmentStatus.ACCEPTED
        assert loaded.segments[0].decision == Decision.ACCEPT
        assert loaded.segments[0].retry_history[0].action == Decision.ACCEPT

    def test_manifest_loads_legacy_segment_list(self, tmp_path: Path):
        path = tmp_path / "commentary_manifest.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": 1,
                        "start": 0.0,
                        "end": 5.0,
                        "duration": 5.0,
                        "title": "第一页",
                        "visible_points": ["a"],
                        "on_screen_text": ["b"],
                        "narration_zh": "第一页介绍。",
                        "frame_paths": ["f1.jpg"],
                        "audio_path": "raw1.mp3",
                        "audio_duration": 4.8,
                        "fitted_audio_path": "fit1.mp3",
                        "fitted_audio_duration": 4.8,
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        loaded = Manifest.load(path)

        assert loaded.version == "1.0-legacy"
        assert loaded.duration == 5.0
        assert loaded.segments[0].status == SegmentStatus.ACCEPTED
        assert loaded.segments[0].selected_draft == "第一页介绍。"
        assert loaded.segments[0].fitted_audio_path == "fit1.mp3"

    def test_load_or_create_manifest_accepts_legacy_resume_manifest(self, tmp_path: Path):
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest_path.write_text(
            json.dumps(
                [
                    {
                        "id": 2,
                        "start": 7.341,
                        "end": 16.683,
                        "duration": 9.342,
                        "title": "旧格式片段",
                        "visible_points": ["step one"],
                        "on_screen_text": ["Run agent"],
                        "narration_zh": "这里继续演示旧格式清单。",
                        "frame_paths": ["frame.jpg"],
                        "audio_path": "raw.mp3",
                        "audio_duration": 9.0,
                        "fitted_audio_path": "fit.mp3",
                        "fitted_audio_duration": 9.0,
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        class Args:
            resume_from_manifest = str(manifest_path)
            force_replan = False
            scene_threshold = 0.32
            min_segment = 3.0
            max_segment = 12.0
            segment_buffer = 0.35
            base_rate = "+0%"
            azure_style = "professional"

        loaded = load_or_create_manifest(
            Args(),
            input_video=tmp_path / "input.mp4",
            output_video=tmp_path / "output.mp4",
            workdir=tmp_path,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            manifest_path=tmp_path / "unused.json",
        )

        assert loaded.version == "1.0-legacy"
        assert loaded.input_video == str(tmp_path / "input.mp4")
        assert loaded.output_video == str(tmp_path / "output.mp4")
        assert loaded.workdir == str(tmp_path)
        assert loaded.segments[0].id == 2
        assert loaded.segments[0].status == SegmentStatus.ACCEPTED

    def test_segment_state_to_narration(self):
        segment = SegmentState(
            id=3,
            start=10.0,
            end=13.0,
            duration=3.0,
            title="标题",
            visible_points=["a"],
            on_screen_text=["b"],
            selected_draft="这里展示了关键步骤。",
            frame_paths=["f1.jpg"],
            raw_audio_path="raw.mp3",
            raw_audio_duration=2.8,
            fitted_audio_path="fit.mp3",
            fitted_audio_duration=2.7,
        )

        narration = segment_state_to_narration(segment)
        assert narration.narration_zh == "这里展示了关键步骤。"
        assert narration.fitted_audio_path == "fit.mp3"

    def test_manifest_helpers_use_manifest_as_single_source_of_truth(self):
        accepted = SegmentState(
            id=1,
            start=0.0,
            end=5.0,
            duration=5.0,
            status=SegmentStatus.ACCEPTED,
            title="第一页",
            selected_draft="第一页介绍。",
            fitted_audio_path="fit1.mp3",
            raw_audio_path="raw1.mp3",
        )
        pending = SegmentState(
            id=2,
            start=5.0,
            end=9.0,
            duration=4.0,
            status=SegmentStatus.CONTENT_GENERATED,
            selected_draft="第二页草稿。",
            fitted_audio_path="",
        )
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir="/tmp/work",
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=20.0,
            segments=[accepted, pending],
        )

        accepted_segments = accepted_segments_from_manifest(manifest)
        accepted_narrations = accepted_narrations_from_manifest(manifest)

        assert [segment.id for segment in accepted_segments] == [1]
        assert [item.id for item in accepted_narrations] == [1]
        assert accepted_narrations[0].narration_zh == "第一页介绍。"

    def test_previous_segment_helpers_are_manifest_based(self):
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir="/tmp/work",
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=20.0,
            segments=[
                SegmentState(id=1, start=0.0, end=3.0, duration=3.0, status=SegmentStatus.ACCEPTED, selected_draft="A"),
                SegmentState(id=2, start=3.0, end=6.0, duration=3.0, status=SegmentStatus.PENDING, selected_draft=""),
                SegmentState(id=3, start=6.0, end=9.0, duration=3.0, status=SegmentStatus.ACCEPTED, selected_draft="C"),
            ],
        )

        previous = get_previous_segment(manifest, 3)
        previous_text = get_previous_accepted_narration(manifest, 3)

        assert previous is not None and previous.id == 2
        assert previous_text == "A"


class TestDecisionHelpers:
    def test_note_segment_decision_updates_reason_and_history(self):
        segment = SegmentState(id=1, start=0.0, end=2.0, duration=2.0)
        note_segment_decision(segment, decision=Decision.RETRY_TTS, reason="need better timing", details={"gap_ms": 420})

        assert segment.decision == Decision.RETRY_TTS
        assert segment.decision_reason == "need better timing"
        assert segment.retry_history[-1].action == Decision.RETRY_TTS
        assert segment.retry_history[-1].details == {"gap_ms": 420}


class TestQAGate:
    def test_empty_narration_requests_retry(self):
        result = evaluate_narration_quality(narration="   ", previous_narration="上一段", duration_seconds=5.0)
        assert result.passed is False
        assert result.decision == Decision.RETRY_NARRATION
        assert "empty narration" in result.feedback

    def test_repetitive_narration_requests_retry(self):
        result = evaluate_narration_quality(
            narration="这里展示 Azure 门户。",
            previous_narration="这里展示Azure门户。",
            duration_seconds=5.0,
        )
        assert result.passed is False
        assert result.decision == Decision.RETRY_NARRATION
        assert "repetitive narration" in result.feedback

    def test_too_dense_narration_requests_retry_or_review(self):
        retry_result = evaluate_narration_quality(
            narration="这是一段明显过长的讲解文案" * 5,
            previous_narration="上一段不同内容",
            duration_seconds=6.0,
        )
        review_result = evaluate_narration_quality(
            narration="极其冗长的讲解文案" * 10,
            previous_narration="上一段不同内容",
            duration_seconds=4.0,
        )
        assert retry_result.passed is False
        assert retry_result.decision == Decision.RETRY_NARRATION
        assert review_result.passed is False
        assert review_result.decision == Decision.NEEDS_HUMAN_REVIEW

    def test_rewrite_narration_once_fixes_empty_text(self):
        rewritten = rewrite_narration_once(
            narration="   ",
            critic_feedback=["empty narration"],
            decision_reason="narration is empty after normalization",
            duration_seconds=4.0,
            previous_narration="上一段",
        )
        assert rewritten
        assert rewritten != "上一段"

    def test_run_qa_gate_step_rewrites_empty_once_and_passes(self):
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir="/tmp/work",
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=20.0,
            segments=[
                SegmentState(id=1, start=0.0, end=4.0, duration=4.0, status=SegmentStatus.CONTENT_GENERATED, selected_draft="   "),
            ],
        )
        segment = manifest.segments[0]

        run_qa_gate_step(manifest, segment)

        assert segment.status == SegmentStatus.CONTENT_GENERATED
        assert segment.auto_retry_attempted is True
        assert segment.rewrite_attempt_count == 1
        assert segment.original_draft == ""
        assert segment.rewritten_draft
        assert segment.final_decision == Decision.ACCEPT

    def test_run_qa_gate_step_rewrites_repetitive_once_and_passes(self):
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir="/tmp/work",
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=20.0,
            segments=[
                SegmentState(id=1, start=0.0, end=4.0, duration=4.0, status=SegmentStatus.ACCEPTED, selected_draft="上一段"),
                SegmentState(id=2, start=4.0, end=8.0, duration=4.0, status=SegmentStatus.CONTENT_GENERATED, selected_draft="上一段", original_draft="上一段"),
            ],
        )
        segment = manifest.segments[1]

        run_qa_gate_step(manifest, segment)

        assert segment.status == SegmentStatus.CONTENT_GENERATED
        assert segment.auto_retry_attempted is True
        assert segment.rewrite_attempt_count == 1
        assert segment.rewritten_draft
        assert segment.rewritten_draft != "上一段"
        assert segment.final_decision == Decision.ACCEPT

    def test_run_qa_gate_step_too_dense_after_rewrite_needs_review(self):
        original = "一二三四五六七八九十一二三四五六七八九十"
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir="/tmp/work",
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=20.0,
            segments=[
                SegmentState(
                    id=1,
                    start=0.0,
                    end=2.0,
                    duration=2.0,
                    status=SegmentStatus.CONTENT_GENERATED,
                    selected_draft=original,
                    original_draft=original,
                ),
            ],
        )
        segment = manifest.segments[0]

        run_qa_gate_step(manifest, segment)

        assert segment.auto_retry_attempted is True
        assert segment.rewrite_attempt_count == 1
        assert segment.status == SegmentStatus.NEEDS_HUMAN_REVIEW
        assert segment.decision == Decision.NEEDS_HUMAN_REVIEW
        assert segment.final_decision == Decision.NEEDS_HUMAN_REVIEW
        assert segment.human_review_status == "qa-rewrite-failed"

    def test_run_qa_gate_step_never_rewrites_more_than_once(self):
        original = "一二三四五六七八九十一二三四五六七八九十"
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir="/tmp/work",
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=20.0,
            segments=[
                SegmentState(
                    id=1,
                    start=0.0,
                    end=2.0,
                    duration=2.0,
                    status=SegmentStatus.CONTENT_GENERATED,
                    selected_draft=original,
                    original_draft=original,
                    rewrite_attempt_count=1,
                    auto_retry_attempted=True,
                ),
            ],
        )
        segment = manifest.segments[0]

        run_qa_gate_step(manifest, segment)

        assert segment.rewrite_attempt_count == 1
        assert segment.status == SegmentStatus.NEEDS_HUMAN_REVIEW
        assert segment.final_decision == Decision.RETRY_NARRATION


class TestPolicies:
    def test_duration_policy_accepts_small_gap(self):
        result = decide_duration_action(budget_seconds=10.0, fitted_duration_seconds=10.2)
        assert result.decision == Decision.ACCEPT

    def test_duration_policy_requests_tts_retry_for_medium_gap(self):
        result = decide_duration_action(budget_seconds=10.0, fitted_duration_seconds=10.8)
        assert result.decision == Decision.RETRY_TTS

    def test_duration_policy_requests_narration_retry_for_large_gap(self):
        result = decide_duration_action(budget_seconds=10.0, fitted_duration_seconds=11.5)
        assert result.decision == Decision.RETRY_NARRATION

    def test_segment_policy_skips_tiny_segment(self):
        segment = SegmentState(id=1, start=0.0, end=0.6, duration=0.6)
        result = decide_segment_action(segment)
        assert result.decision == Decision.SKIP_SEGMENT

    def test_segment_policy_merges_very_short_segment_with_previous(self):
        previous = SegmentState(id=1, start=0.0, end=5.0, duration=5.0)
        segment = SegmentState(id=2, start=5.0, end=6.2, duration=1.2)
        result = decide_segment_action(segment, previous_segment=previous)
        assert result.decision == Decision.MERGE_WITH_PREVIOUS


class TestResumeRedoHelpers:
    def test_should_process_segment_defaults_to_unfinished_only(self):
        accepted = SegmentState(id=1, start=0.0, end=1.0, duration=1.0, status=SegmentStatus.ACCEPTED)
        pending = SegmentState(id=2, start=1.0, end=2.0, duration=1.0, status=SegmentStatus.PENDING)

        assert should_process_segment(accepted, redo=None, target_segment_ids=None) is False
        assert should_process_segment(pending, redo=None, target_segment_ids=None) is True

    def test_should_process_segment_honors_target_and_redo(self):
        accepted = SegmentState(id=7, start=0.0, end=1.0, duration=1.0, status=SegmentStatus.ACCEPTED)
        assert should_process_segment(accepted, redo="tts", target_segment_ids={7, 9}) is True
        assert should_process_segment(accepted, redo="tts", target_segment_ids={8, 9}) is False

    def test_parse_segment_id_list(self):
        assert parse_segment_id_list("2, 4,17") == [2, 4, 17]

    def test_resolve_target_segment_ids_merges_single_and_multi(self):
        args = build_arg_parser().parse_args(
            [
                "--input",
                "in.mp4",
                "--output",
                "out.mp4",
                "--workdir",
                "work",
                "--segment-id",
                "7",
                "--segment-ids",
                "2,4,7",
            ]
        )

        assert resolve_target_segment_ids(args) == {2, 4, 7}

    def test_resolve_target_segment_ids_requires_target_for_redo(self):
        args = build_arg_parser().parse_args(
            [
                "--input",
                "in.mp4",
                "--output",
                "out.mp4",
                "--workdir",
                "work",
                "--redo",
                "narration",
            ]
        )

        try:
            resolve_target_segment_ids(args)
        except SystemExit as exc:
            assert str(exc) == "--redo requires --segment-id or --segment-ids"
        else:  # pragma: no cover
            raise AssertionError("expected SystemExit for redo without segment target")

    def test_maybe_apply_redo_resets_multiple_target_segments(self, tmp_path: Path):
        manifest_path = tmp_path / "commentary_manifest.json"
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir=str(tmp_path),
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=20.0,
            segments=[
                SegmentState(
                    id=2,
                    start=0.0,
                    end=5.0,
                    duration=5.0,
                    status=SegmentStatus.ACCEPTED,
                    selected_draft="seg2",
                    raw_audio_path="raw2.mp3",
                    fitted_audio_path="fit2.mp3",
                ),
                SegmentState(
                    id=4,
                    start=5.0,
                    end=10.0,
                    duration=5.0,
                    status=SegmentStatus.ACCEPTED,
                    selected_draft="seg4",
                    raw_audio_path="raw4.mp3",
                    fitted_audio_path="fit4.mp3",
                ),
                SegmentState(
                    id=5,
                    start=10.0,
                    end=15.0,
                    duration=5.0,
                    status=SegmentStatus.ACCEPTED,
                    selected_draft="seg5",
                    raw_audio_path="raw5.mp3",
                    fitted_audio_path="fit5.mp3",
                ),
            ],
        )

        class Args:
            redo = "narration"

        maybe_apply_redo(manifest, Args(), manifest_path=manifest_path, target_segment_ids={2, 4})

        assert manifest.get_segment(2).status == SegmentStatus.FRAMES_EXTRACTED
        assert manifest.get_segment(4).status == SegmentStatus.FRAMES_EXTRACTED
        assert manifest.get_segment(5).status == SegmentStatus.ACCEPTED

    def test_validate_target_segment_ids_rejects_unknown_ids(self):
        manifest = Manifest(
            version="2.0",
            input_video="in.mp4",
            output_video="out.mp4",
            workdir=".",
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=20.0,
            segments=[
                SegmentState(id=2, start=0.0, end=5.0, duration=5.0),
                SegmentState(id=4, start=5.0, end=10.0, duration=5.0),
            ],
        )

        try:
            validate_target_segment_ids(manifest, {2, 17})
        except SystemExit as exc:
            assert str(exc) == "Unknown segment id(s): 17"
        else:  # pragma: no cover
            raise AssertionError("expected SystemExit for unknown target segment ids")

    def test_reset_segment_for_redo(self):
        segment = SegmentState(
            id=5,
            start=0.0,
            end=3.0,
            duration=3.0,
            status=SegmentStatus.ACCEPTED,
            title="标题",
            selected_draft="讲稿",
            raw_audio_path="raw.mp3",
            fitted_audio_path="fit.mp3",
            decision=Decision.ACCEPT,
            decision_reason="ok",
            errors=["x"],
        )

        reset_segment_for_redo(segment, "tts")
        assert segment.status == SegmentStatus.CONTENT_GENERATED
        assert segment.raw_audio_path == ""
        assert segment.fitted_audio_path == ""
        assert segment.decision is None
        assert segment.errors == []

        reset_segment_for_redo(segment, "narration")
        assert segment.status == SegmentStatus.FRAMES_EXTRACTED
        assert segment.selected_draft == ""

        reset_segment_for_redo(segment, "vision")
        assert segment.status == SegmentStatus.PENDING
        assert segment.frame_paths == []
