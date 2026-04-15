"""Azure Content Understanding integration for video analysis.

Provides an alternative to the custom ffmpeg scene-detection + keyframe
extraction + Azure OpenAI vision chain.  A single API call to Azure Content
Understanding returns:

* Automatic scene segmentation with timestamps
* Semantically-selected keyframes per segment
* Transcription (WebVTT) with diarization
* Custom field extraction (UI elements, product names, visible text)
* Shot boundaries (cameraShotTimesMs)

The module exposes two public helpers:

``analyze_video``
    Call the Content Understanding REST API and return the raw result dict.

``content_understanding_to_segments``
    Convert the raw CU result into a list of ``SegmentState`` objects that
    the rest of the pipeline can consume directly.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from .azure_auth import cognitive_services_auth_headers
from .core import Segment, build_segments, normalize_terms
from .state import SegmentState, SegmentStatus


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CU_API_VERSION = "2025-11-01"
DEFAULT_ANALYZER = "prebuilt-videoAnalysis"


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------

def _cu_endpoint() -> str:
    endpoint = os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "").rstrip("/")
    if not endpoint:
        raise SystemExit(
            "Missing required environment variable: AZURE_CONTENT_UNDERSTANDING_ENDPOINT"
        )
    return endpoint


def _cu_api_key() -> str | None:
    """Return the Content Understanding API key, or ``None`` for MSI auth."""
    return os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY") or None


# ---------------------------------------------------------------------------
# Custom analyzer creation
# ---------------------------------------------------------------------------

VISUAL_COMMENTARY_FIELD_SCHEMA = [
    {
        "name": "title",
        "type": "string",
        "description": "A one-sentence title describing the key content shown in this video segment.",
    },
    {
        "name": "visible_points",
        "type": "array",
        "items": {"type": "string"},
        "description": "Key visual elements, UI controls, or notable items visible on screen.",
    },
    {
        "name": "on_screen_text",
        "type": "array",
        "items": {"type": "string"},
        "description": "Important text visible on screen (menus, labels, headings). Do not OCR every word — only key labels.",
    },
    {
        "name": "narration_zh",
        "type": "string",
        "description": (
            "1–3 sentences of natural Chinese narration describing what this segment shows. "
            "Focus on what changed from the previous segment. Keep English proper nouns in English "
            "(e.g. Azure, OpenAI, Copilot Studio). Suitable for voiceover at normal speech rate."
        ),
    },
    {
        "name": "video_type_hint",
        "type": "string",
        "description": (
            "One of: deck_recording, product_demo, portal_walkthrough, dashboard_demo, mixed_visual_demo. "
            "Inferred from the visual content of this segment."
        ),
    },
]


def create_custom_analyzer(
    *,
    analyzer_id: str = "visual-commentary-analyzer",
    model_deployment: str | None = None,
) -> dict[str, Any]:
    """Create or update a custom Content Understanding analyzer with our field schema.

    Returns the analyzer definition dict.
    """
    endpoint = _cu_endpoint()
    api_key = _cu_api_key()
    api_version = os.getenv("AZURE_CU_API_VERSION", DEFAULT_CU_API_VERSION)

    model = model_deployment or os.getenv("AZURE_CU_MODEL_DEPLOYMENT", "gpt-4o")

    analyzer_def: dict[str, Any] = {
        "description": "Visual commentary pipeline analyzer for demo/slide/portal video narration",
        "scenario": "videoAnalysis",
        "fieldSchema": VISUAL_COMMENTARY_FIELD_SCHEMA,
        "config": {
            "returnDetails": True,
        },
    }

    url = f"{endpoint}/contentunderstanding/analyzers/{analyzer_id}?api-version={api_version}"
    response = requests.put(
        url,
        headers={**cognitive_services_auth_headers(api_key), "Content-Type": "application/json"},
        json=analyzer_def,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: Path,
    *,
    analyzer_id: str | None = None,
    poll_interval: float = 5.0,
    max_wait: float = 600.0,
) -> dict[str, Any]:
    """Submit a video for analysis and poll until complete.

    Parameters
    ----------
    video_path:
        Path to the local video file.
    analyzer_id:
        Analyzer to use.  Defaults to the prebuilt video analyzer.
    poll_interval:
        Seconds between status polls.
    max_wait:
        Maximum seconds to wait before raising a timeout error.

    Returns
    -------
    dict
        The full Content Understanding result payload containing segments,
        keyframes, transcript, and extracted fields.
    """
    endpoint = _cu_endpoint()
    api_key = _cu_api_key()
    api_version = os.getenv("AZURE_CU_API_VERSION", DEFAULT_CU_API_VERSION)
    analyzer = analyzer_id or DEFAULT_ANALYZER

    headers = {
        **cognitive_services_auth_headers(api_key),
        "Content-Type": "application/octet-stream",
    }

    # Submit the video for analysis
    url = f"{endpoint}/contentunderstanding/analyzers/{analyzer}:analyze?api-version={api_version}"
    with open(video_path, "rb") as f:
        response = requests.post(url, headers=headers, data=f, timeout=120)
    response.raise_for_status()

    # Get the operation location for polling
    operation_url = response.headers.get("Operation-Location", "")
    if not operation_url:
        # Synchronous result
        return response.json()

    # Poll for completion
    poll_headers = cognitive_services_auth_headers(api_key)
    elapsed = 0.0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        poll_response = requests.get(operation_url, headers=poll_headers, timeout=30)
        poll_response.raise_for_status()
        result = poll_response.json()
        status = result.get("status", "")
        if status in ("succeeded", "Succeeded"):
            return result.get("result", result)
        if status in ("failed", "Failed"):
            raise RuntimeError(
                f"Content Understanding analysis failed: {json.dumps(result, ensure_ascii=False)[:2000]}"
            )

    raise TimeoutError(
        f"Content Understanding analysis did not complete within {max_wait}s"
    )


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_timestamp_ms(ms_value: int | float) -> float:
    """Convert milliseconds to seconds."""
    return round(float(ms_value) / 1000.0, 3)


def parse_cu_segments(
    cu_result: dict[str, Any],
    *,
    video_duration: float,
    min_segment: float = 3.0,
    max_segment: float = 12.0,
) -> list[dict[str, Any]]:
    """Extract segment information from a Content Understanding result.

    Returns a list of dicts with keys:
    ``start``, ``end``, ``duration``, ``title``, ``description``,
    ``visible_points``, ``on_screen_text``, ``narration_zh``,
    ``keyframe_paths``, ``video_type_hint``, ``transcript``.
    """
    segments: list[dict[str, Any]] = []

    # CU segments can appear under different keys depending on the analyzer
    content = cu_result.get("contents", cu_result.get("content", []))
    if isinstance(content, str):
        # Markdown output — parse segment boundaries from headers
        return _parse_markdown_segments(content, video_duration=video_duration)

    # Structured JSON segments
    for item in content if isinstance(content, list) else []:
        seg: dict[str, Any] = {}

        # Time boundaries
        start_ms = item.get("startTimeMs", item.get("offset", 0))
        end_ms = item.get("endTimeMs", item.get("offset", 0) + item.get("duration", 0))
        seg["start"] = _parse_timestamp_ms(start_ms)
        seg["end"] = _parse_timestamp_ms(end_ms)
        seg["duration"] = round(seg["end"] - seg["start"], 3)

        # Fields from custom or prebuilt schema
        fields = item.get("fields", {})
        seg["title"] = normalize_terms(str(fields.get("title", {}).get("value", "")))
        seg["description"] = str(fields.get("description", {}).get("value", ""))
        seg["narration_zh"] = normalize_terms(
            str(fields.get("narration_zh", {}).get("value", ""))
        )
        seg["video_type_hint"] = str(
            fields.get("video_type_hint", {}).get("value", "mixed_visual_demo")
        )

        # Array fields
        vp_field = fields.get("visible_points", {})
        vp_values = vp_field.get("values", vp_field.get("value", []))
        seg["visible_points"] = [
            normalize_terms(str(v.get("value", v) if isinstance(v, dict) else v))
            for v in (vp_values if isinstance(vp_values, list) else [])
        ][:6]

        ost_field = fields.get("on_screen_text", {})
        ost_values = ost_field.get("values", ost_field.get("value", []))
        seg["on_screen_text"] = [
            normalize_terms(str(v.get("value", v) if isinstance(v, dict) else v))
            for v in (ost_values if isinstance(ost_values, list) else [])
        ][:8]

        # Keyframe paths (URLs or local paths from CU response)
        keyframes = item.get("keyFrames", item.get("keyframes", []))
        seg["keyframe_paths"] = [
            str(kf.get("url", kf.get("path", ""))) for kf in keyframes if isinstance(kf, dict)
        ]

        # Transcript snippet
        seg["transcript"] = str(item.get("transcript", ""))

        if seg["duration"] > 0.05:
            segments.append(seg)

    return segments


def _parse_markdown_segments(
    markdown: str,
    *,
    video_duration: float,
) -> list[dict[str, Any]]:
    """Parse segment info from CU markdown output format.

    CU prebuilt analysers produce markdown like:

        # Video: 00:00.000 => 00:06.000
        A lively room...
    """
    segments: list[dict[str, Any]] = []
    pattern = re.compile(
        r"#+ Video:\s*(\d{2}:\d{2}\.\d{3})\s*=>\s*(\d{2}:\d{2}\.\d{3})\s*\n(.*?)(?=\n#+ Video:|\Z)",
        re.S,
    )
    for match in pattern.finditer(markdown):
        start = _parse_mm_ss(match.group(1))
        end = _parse_mm_ss(match.group(2))
        body = match.group(3).strip()
        segments.append({
            "start": start,
            "end": end,
            "duration": round(end - start, 3),
            "title": "",
            "description": body[:200],
            "narration_zh": "",
            "video_type_hint": "mixed_visual_demo",
            "visible_points": [],
            "on_screen_text": [],
            "keyframe_paths": [],
            "transcript": "",
        })
    return segments


def _parse_mm_ss(value: str) -> float:
    """Parse ``MM:SS.mmm`` → seconds."""
    match = re.fullmatch(r"(\d{2}):(\d{2})\.(\d{3})", value)
    if not match:
        return 0.0
    minutes, seconds, ms = int(match.group(1)), int(match.group(2)), int(match.group(3))
    return round(minutes * 60 + seconds + ms / 1000.0, 3)


# ---------------------------------------------------------------------------
# Conversion to pipeline SegmentState objects
# ---------------------------------------------------------------------------

def content_understanding_to_segments(
    cu_result: dict[str, Any],
    *,
    video_duration: float,
    min_segment: float = 3.0,
    max_segment: float = 12.0,
) -> list[SegmentState]:
    """Convert CU analysis results into pipeline SegmentState objects.

    This function:
    1. Parses CU segments
    2. Applies the existing merge/split logic (respecting min/max segment)
    3. Enriches each SegmentState with CU metadata (title, visible_points, etc.)
    4. Sets status to FRAMES_EXTRACTED (since CU already provides keyframes)

    Returns a list ready to be used in a ``Manifest``.
    """
    cu_segments = parse_cu_segments(
        cu_result, video_duration=video_duration, min_segment=min_segment, max_segment=max_segment
    )

    if not cu_segments:
        return []

    # Extract cut points from CU segments for merge/split
    cuts = sorted({seg["start"] for seg in cu_segments if seg["start"] > 0.05})
    built = build_segments(video_duration, cuts, min_segment=min_segment, max_segment=max_segment)

    # Build a lookup from CU segments by time overlap
    states: list[SegmentState] = []
    for segment in built:
        # Find the best-matching CU segment by overlap
        best_cu = _find_best_cu_segment(segment, cu_segments)

        state = SegmentState.from_segment(
            seg_id=segment.id,
            start=segment.start,
            end=segment.end,
            duration=segment.duration,
        )

        if best_cu:
            state.title = best_cu.get("title", "")
            state.visible_points = best_cu.get("visible_points", [])
            state.on_screen_text = best_cu.get("on_screen_text", [])
            state.vision_result = {
                "narration_zh": best_cu.get("narration_zh", ""),
                "title": best_cu.get("title", ""),
                "visible_points": best_cu.get("visible_points", []),
                "on_screen_text": best_cu.get("on_screen_text", []),
                "description": best_cu.get("description", ""),
                "video_type_hint": best_cu.get("video_type_hint", ""),
                "transcript": best_cu.get("transcript", ""),
                "source": "content_understanding",
            }
            state.frame_paths = best_cu.get("keyframe_paths", [])
            # CU provides vision + frames, so skip ahead
            if state.frame_paths and best_cu.get("narration_zh"):
                state.status = SegmentStatus.FRAMES_EXTRACTED
            elif state.frame_paths:
                state.status = SegmentStatus.FRAMES_EXTRACTED

        states.append(state)

    return states


def _find_best_cu_segment(
    segment: Segment, cu_segments: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find the CU segment with the largest time overlap."""
    best: dict[str, Any] | None = None
    best_overlap = 0.0
    for cu_seg in cu_segments:
        overlap_start = max(segment.start, cu_seg["start"])
        overlap_end = min(segment.end, cu_seg["end"])
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = cu_seg
    return best


# ---------------------------------------------------------------------------
# Shot boundary extraction
# ---------------------------------------------------------------------------

def extract_shot_boundaries(cu_result: dict[str, Any]) -> list[float]:
    """Extract shot boundary timestamps (in seconds) from CU result details."""
    details = cu_result.get("details", cu_result.get("analyzeResult", {}))
    shot_times_ms = details.get("cameraShotTimesMs", [])
    return [_parse_timestamp_ms(ms) for ms in shot_times_ms]


def infer_video_type_from_cu(cu_result: dict[str, Any]) -> str:
    """Infer the dominant video type from CU segment hints.

    Counts ``video_type_hint`` across all segments and returns the most common.
    """
    cu_segments = cu_result.get("contents", cu_result.get("content", []))
    if not isinstance(cu_segments, list):
        return "mixed_visual_demo"

    type_counts: dict[str, int] = {}
    for item in cu_segments:
        fields = item.get("fields", {})
        hint = str(fields.get("video_type_hint", {}).get("value", "mixed_visual_demo"))
        type_counts[hint] = type_counts.get(hint, 0) + 1

    if not type_counts:
        return "mixed_visual_demo"

    return max(type_counts, key=type_counts.get)  # type: ignore[arg-type]
