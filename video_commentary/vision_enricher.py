"""Multi-model vision enrichment pipeline.

This module implements Phase 4 of the architecture redesign:

1. **Azure Document Intelligence** — precise OCR for UI screenshots (portal
   walkthroughs, dashboards) where exact text labels matter.

2. **LLM-based Video Profiler** — replaces filename-based heuristics with
   GPT-4o vision analysis of sampled frames to infer video type and policy.

3. **Vision Enricher** — combines Content Understanding output, GPT-4o vision,
   and Document Intelligence into a single enriched metadata dict per segment.
"""

from __future__ import annotations

import base64
import json
import os
import textwrap
from pathlib import Path
from typing import Any

import requests

from .core import normalize_terms
from .planner import (
    ALLOWED_VIDEO_TYPES,
    DEFAULT_VIDEO_TYPE,
    VideoProfile,
    default_profile_for_type,
    normalize_video_profile,
)
from .state import SegmentState


# ---------------------------------------------------------------------------
# Azure Document Intelligence — OCR for UI screenshots
# ---------------------------------------------------------------------------

def analyze_document_ocr(
    image_path: Path,
    *,
    model_id: str = "prebuilt-read",
) -> dict[str, Any]:
    """Extract text from an image using Azure Document Intelligence.

    Requires environment variables:
    - ``AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT``
    - ``AZURE_DOCUMENT_INTELLIGENCE_KEY``

    Parameters
    ----------
    image_path:
        Path to the image file (JPEG or PNG).
    model_id:
        Document Intelligence model.  ``prebuilt-read`` is best for UI text.

    Returns
    -------
    dict with keys:
        ``text`` — full extracted text,
        ``lines`` — list of dicts with ``text`` and ``confidence``,
        ``words`` — list of dicts with ``text`` and ``confidence``.
    """
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")

    if not endpoint or not api_key:
        return {"text": "", "lines": [], "words": [], "error": "credentials not configured"}

    mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"

    url = f"{endpoint}/documentintelligence/documentModels/{model_id}:analyze?api-version=2024-11-30"

    response = requests.post(
        url,
        headers={
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": mime,
        },
        data=image_path.read_bytes(),
        timeout=60,
    )

    if response.status_code == 202:
        # Async operation — poll for result
        import time

        operation_url = response.headers.get("Operation-Location", "")
        for _ in range(30):
            time.sleep(2)
            poll = requests.get(
                operation_url,
                headers={"Ocp-Apim-Subscription-Key": api_key},
                timeout=30,
            )
            poll.raise_for_status()
            result = poll.json()
            if result.get("status") in ("succeeded", "Succeeded"):
                return _parse_doc_intel_result(result.get("analyzeResult", {}))
            if result.get("status") in ("failed", "Failed"):
                return {"text": "", "lines": [], "words": [], "error": str(result)}
        return {"text": "", "lines": [], "words": [], "error": "timeout polling"}

    response.raise_for_status()
    return _parse_doc_intel_result(response.json().get("analyzeResult", {}))


def _parse_doc_intel_result(analyze_result: dict[str, Any]) -> dict[str, Any]:
    """Parse Document Intelligence analyzeResult into a simplified structure."""
    full_text = analyze_result.get("content", "")
    lines = []
    words = []

    for page in analyze_result.get("pages", []):
        for line in page.get("lines", []):
            lines.append({
                "text": line.get("content", ""),
                "confidence": float(line.get("confidence", 0.0)),
            })
        for word in page.get("words", []):
            words.append({
                "text": word.get("content", ""),
                "confidence": float(word.get("confidence", 0.0)),
            })

    return {"text": full_text, "lines": lines, "words": words}


# ---------------------------------------------------------------------------
# LLM-based Video Profiler
# ---------------------------------------------------------------------------

def profile_video_with_llm(
    frame_paths: list[Path],
    *,
    requested_scene_threshold: float = 0.32,
    requested_min_segment: float = 3.0,
    requested_max_segment: float = 12.0,
    requested_base_rate: str = "+0%",
    requested_azure_style: str = "professional",
) -> VideoProfile:
    """Infer video type and policies from sampled frames using GPT-4o vision.

    Sends 5-8 evenly sampled frames to the LLM and asks it to classify
    the video type and suggest segmentation/style parameters.

    Falls back to heuristic profiling if the LLM call fails.

    Parameters
    ----------
    frame_paths:
        Paths to sample frames from the video.
    requested_*:
        CLI-provided parameter defaults.

    Returns
    -------
    VideoProfile
        Profile with video_type, confidence, policies, and rationale.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not all([endpoint, api_key, deployment]) or not frame_paths:
        return _fallback_profile(
            requested_scene_threshold=requested_scene_threshold,
            requested_min_segment=requested_min_segment,
            requested_max_segment=requested_max_segment,
            requested_base_rate=requested_base_rate,
            requested_azure_style=requested_azure_style,
        )

    user_prompt = textwrap.dedent("""\
        Analyze these video frames and determine the video type.

        Classify into exactly one of:
        - deck_recording (slides, presentations, PPT)
        - product_demo (software product walkthrough)
        - portal_walkthrough (cloud console, admin panel, Azure portal)
        - dashboard_demo (analytics dashboard, Grafana, Power BI)
        - mixed_visual_demo (combination or unclear)

        Return valid JSON:
        {
          "video_type": "one of the above types",
          "confidence": 0.0-1.0,
          "rationale": ["reason1", "reason2"],
          "suggested_scene_threshold": 0.20-0.50,
          "suggested_min_segment": 2.0-6.0,
          "suggested_max_segment": 6.0-16.0,
          "suggested_narration_density": "concise|balanced|explanatory",
          "suggested_narration_focus": "screen_change|operation_step|summary_first"
        }
    """)

    # Build vision request with frames
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for path in frame_paths[:8]:
        if not path.exists():
            continue
        mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}", "detail": "low"},
        })

    payload = {
        "messages": [
            {"role": "system", "content": "You are a video analysis expert. Return only valid JSON."},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    try:
        response = requests.post(
            url,
            headers={"api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]

        import re
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.S)
            if not match:
                raise RuntimeError(f"LLM did not return JSON: {raw_text[:500]}")
            parsed = json.loads(match.group(0))

        video_type = parsed.get("video_type", DEFAULT_VIDEO_TYPE)
        if video_type not in ALLOWED_VIDEO_TYPES:
            video_type = DEFAULT_VIDEO_TYPE

        profile = VideoProfile(
            video_type=video_type,
            confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.6)))),
            segmentation_policy={
                "scene_threshold": float(parsed.get("suggested_scene_threshold", requested_scene_threshold)),
                "min_segment": float(parsed.get("suggested_min_segment", requested_min_segment)),
                "max_segment": float(parsed.get("suggested_max_segment", requested_max_segment)),
            },
            style_policy={
                "narration_density": parsed.get("suggested_narration_density", "balanced"),
                "narration_focus": parsed.get("suggested_narration_focus", "screen_change"),
                "azure_style": requested_azure_style,
                "base_rate": requested_base_rate,
            },
            rationale=[
                f"llm profiler inferred video_type={video_type}",
                *[str(r) for r in parsed.get("rationale", [])],
            ],
        )

        return normalize_video_profile(
            profile,
            requested_scene_threshold=requested_scene_threshold,
            requested_min_segment=requested_min_segment,
            requested_max_segment=requested_max_segment,
            requested_base_rate=requested_base_rate,
            requested_azure_style=requested_azure_style,
        )

    except Exception as exc:
        profile = _fallback_profile(
            requested_scene_threshold=requested_scene_threshold,
            requested_min_segment=requested_min_segment,
            requested_max_segment=requested_max_segment,
            requested_base_rate=requested_base_rate,
            requested_azure_style=requested_azure_style,
        )
        profile.rationale.append(f"llm profiler fallback due to error: {exc}")
        return profile


def _fallback_profile(
    *,
    requested_scene_threshold: float,
    requested_min_segment: float,
    requested_max_segment: float,
    requested_base_rate: str,
    requested_azure_style: str,
) -> VideoProfile:
    """Create a fallback profile using heuristic defaults."""
    return normalize_video_profile(
        None,
        requested_scene_threshold=requested_scene_threshold,
        requested_min_segment=requested_min_segment,
        requested_max_segment=requested_max_segment,
        requested_base_rate=requested_base_rate,
        requested_azure_style=requested_azure_style,
    )


# ---------------------------------------------------------------------------
# Multi-model Vision Enricher
# ---------------------------------------------------------------------------

def enrich_segment_vision(
    segment: SegmentState,
    *,
    cu_data: dict[str, Any] | None = None,
    use_doc_intel: bool = False,
    use_gpt4o_vision: bool = True,
    previous_narration: str = "",
    style_policy: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Enrich a segment's visual understanding by fusing multiple model outputs.

    The enrichment strategy is:

    1. **Primary**: Content Understanding data (if available) — provides
       structured fields, keyframes, and descriptions.
    2. **Secondary**: GPT-4o vision (if enabled) — provides narrative-quality
       understanding focused on what changed and what story to tell.
    3. **Tertiary**: Document Intelligence OCR (if enabled) — provides precise
       UI text for portal/dashboard videos.

    The outputs are merged into a single dict that downstream narration and
    QA nodes can consume.

    Parameters
    ----------
    segment:
        The segment to enrich.
    cu_data:
        Pre-existing Content Understanding data for this segment.
    use_doc_intel:
        Whether to run Document Intelligence OCR on keyframes.
    use_gpt4o_vision:
        Whether to run GPT-4o vision analysis.
    previous_narration:
        Previous segment's narration for context.
    style_policy:
        Narration density and focus settings.

    Returns
    -------
    dict
        Merged vision result with keys: ``title``, ``visible_points``,
        ``on_screen_text``, ``narration_zh``, ``description``,
        ``ocr_text``, ``sources``.
    """
    result: dict[str, Any] = {
        "title": segment.title or "",
        "visible_points": list(segment.visible_points),
        "on_screen_text": list(segment.on_screen_text),
        "narration_zh": "",
        "description": "",
        "ocr_text": "",
        "sources": [],
    }

    # Layer 1: Content Understanding data
    if cu_data:
        result["title"] = result["title"] or cu_data.get("title", "")
        result["description"] = cu_data.get("description", "")
        result["narration_zh"] = cu_data.get("narration_zh", "")

        cu_vp = cu_data.get("visible_points", [])
        if cu_vp and not result["visible_points"]:
            result["visible_points"] = cu_vp

        cu_ost = cu_data.get("on_screen_text", [])
        if cu_ost and not result["on_screen_text"]:
            result["on_screen_text"] = cu_ost

        result["sources"].append("content_understanding")

    # Layer 2: GPT-4o vision
    if use_gpt4o_vision and segment.frame_paths:
        try:
            from .pipeline import call_azure_openai_vision

            vision = call_azure_openai_vision(
                frame_paths=[Path(p) for p in segment.frame_paths],
                segment=segment,
                previous_narration=previous_narration,
                style_policy=style_policy,
            )

            # GPT-4o provides narration-quality understanding
            if not result["narration_zh"]:
                result["narration_zh"] = vision.get("narration_zh", "")
            if not result["title"]:
                result["title"] = vision.get("title", "")

            # Merge visible points (deduplicate)
            gpt_vp = vision.get("visible_points", [])
            existing_vp = set(result["visible_points"])
            for vp in gpt_vp:
                if vp not in existing_vp:
                    result["visible_points"].append(vp)
                    existing_vp.add(vp)
            result["visible_points"] = result["visible_points"][:8]

            # Merge on-screen text
            gpt_ost = vision.get("on_screen_text", [])
            existing_ost = set(result["on_screen_text"])
            for ost in gpt_ost:
                if ost not in existing_ost:
                    result["on_screen_text"].append(ost)
                    existing_ost.add(ost)
            result["on_screen_text"] = result["on_screen_text"][:10]

            result["sources"].append("gpt4o_vision")
        except Exception:
            pass  # Graceful degradation

    # Layer 3: Document Intelligence OCR
    if use_doc_intel and segment.frame_paths:
        ocr_texts: list[str] = []
        for frame_path_str in segment.frame_paths[:2]:
            frame_path = Path(frame_path_str)
            if not frame_path.exists():
                continue
            try:
                ocr_result = analyze_document_ocr(frame_path)
                if ocr_result.get("text"):
                    ocr_texts.append(ocr_result["text"])
            except Exception:
                pass

        if ocr_texts:
            result["ocr_text"] = "\n---\n".join(ocr_texts)
            # Add high-confidence OCR lines to on_screen_text
            for ocr in ocr_texts:
                for line in ocr.split("\n"):
                    line = line.strip()
                    if len(line) >= 3 and line not in result["on_screen_text"]:
                        result["on_screen_text"].append(normalize_terms(line))
            result["on_screen_text"] = result["on_screen_text"][:12]
            result["sources"].append("document_intelligence")

    return result
