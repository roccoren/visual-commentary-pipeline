"""Tests for the video commentary engineering helpers."""

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
from video_commentary.pipeline import (
    azure_openai_uses_responses_api,
    build_azure_openai_vision_request,
    extract_azure_openai_text,
)


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

        assert "mstts:audioduration value=\"18000ms\"" in ssml
        assert "<prosody rate=\"+5%\">AT&amp;T &lt;Azure&gt;</prosody>" in ssml
        assert 'style="professional"' in ssml
        assert "xmlns:mstts=\"https://www.w3.org/2001/mstts\"" in ssml

    def test_omits_optional_nodes_when_not_requested(self):
        ssml = build_azure_tts_ssml(
            text="讲解内容",
            voice="zh-CN-XiaoxiaoNeural",
        )

        assert "mstts:audioduration" not in ssml
        assert "mstts:express-as" not in ssml
        assert "<prosody rate=\"+0%\">讲解内容</prosody>" in ssml


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

