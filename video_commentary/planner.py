"""Lightweight pre-run planner for video profiling and policy defaults."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_VIDEO_TYPES = {
    "deck_recording",
    "product_demo",
    "portal_walkthrough",
    "dashboard_demo",
    "mixed_visual_demo",
}
ALLOWED_NARRATION_DENSITIES = {"concise", "balanced", "explanatory"}
ALLOWED_NARRATION_FOCUS = {"screen_change", "operation_step", "summary_first"}
DEFAULT_VIDEO_TYPE = "mixed_visual_demo"


@dataclass
class VideoProfile:
    video_type: str = DEFAULT_VIDEO_TYPE
    confidence: float = 0.0
    segmentation_policy: dict[str, Any] = field(default_factory=dict)
    style_policy: dict[str, Any] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoProfile":
        return cls(
            video_type=str(data.get("video_type", DEFAULT_VIDEO_TYPE)),
            confidence=float(data.get("confidence", 0.0)),
            segmentation_policy=dict(data.get("segmentation_policy", {})),
            style_policy=dict(data.get("style_policy", {})),
            rationale=[str(item) for item in data.get("rationale", [])],
        )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def clamp_rate(value: str, *, fallback: str) -> str:
    text = str(value or "").strip()
    match = re.fullmatch(r"([+-]?)(\d{1,2})%", text)
    if not match:
        return fallback
    sign = "-" if match.group(1) == "-" else "+"
    number = clamp(float(match.group(2)), 0.0, 25.0)
    return f"{sign}{int(number)}%"


def infer_video_type(input_video: Path) -> str:
    name = input_video.stem.lower()
    if any(token in name for token in {"deck", "slide", "slides", "ppt", "presentation"}):
        return "deck_recording"
    if any(token in name for token in {"portal", "azure", "console", "admin"}):
        return "portal_walkthrough"
    if any(token in name for token in {"dashboard", "grafana", "powerbi", "metric", "analytics", "bi"}):
        return "dashboard_demo"
    if any(token in name for token in {"demo", "product", "walkthrough", "onboarding", "app"}):
        return "product_demo"
    return DEFAULT_VIDEO_TYPE


def default_profile_for_type(
    video_type: str,
    *,
    requested_scene_threshold: float,
    requested_min_segment: float,
    requested_max_segment: float,
    requested_base_rate: str,
    requested_azure_style: str,
) -> VideoProfile:
    profile = VideoProfile(
        video_type=video_type,
        confidence=0.55 if video_type != DEFAULT_VIDEO_TYPE else 0.0,
        segmentation_policy={
            "scene_threshold": requested_scene_threshold,
            "min_segment": requested_min_segment,
            "max_segment": requested_max_segment,
        },
        style_policy={
            "narration_density": "balanced",
            "narration_focus": "screen_change",
            "azure_style": requested_azure_style,
            "base_rate": requested_base_rate,
        },
        rationale=[f"default policy seeded for video_type={video_type}"],
    )

    if video_type == "deck_recording":
        profile.segmentation_policy.update(
            {
                "scene_threshold": requested_scene_threshold - 0.04,
                "min_segment": requested_min_segment + 1.0,
                "max_segment": requested_max_segment + 2.0,
            }
        )
        profile.style_policy.update(
            {
                "narration_density": "balanced",
                "narration_focus": "summary_first",
                "base_rate": "+0%",
            }
        )
        profile.rationale.append("slides usually benefit from slightly longer, summary-oriented segments")
    elif video_type == "portal_walkthrough":
        profile.segmentation_policy.update(
            {
                "scene_threshold": requested_scene_threshold + 0.04,
                "min_segment": requested_min_segment - 0.5,
                "max_segment": requested_max_segment - 2.0,
            }
        )
        profile.style_policy.update(
            {
                "narration_density": "balanced",
                "narration_focus": "operation_step",
                "base_rate": "+2%",
            }
        )
        profile.rationale.append("portal workflows are action-driven and benefit from slightly tighter segments")
    elif video_type == "dashboard_demo":
        profile.segmentation_policy.update(
            {
                "scene_threshold": requested_scene_threshold + 0.01,
                "min_segment": requested_min_segment,
                "max_segment": requested_max_segment - 1.0,
            }
        )
        profile.style_policy.update(
            {
                "narration_density": "concise",
                "narration_focus": "screen_change",
            }
        )
        profile.rationale.append("dashboard screens change less often, so narration should stay compact")
    elif video_type == "product_demo":
        profile.segmentation_policy.update(
            {
                "scene_threshold": requested_scene_threshold + 0.02,
                "min_segment": requested_min_segment - 0.25,
                "max_segment": requested_max_segment - 1.0,
            }
        )
        profile.style_policy.update(
            {
                "narration_density": "balanced",
                "narration_focus": "operation_step",
                "base_rate": "+1%",
            }
        )
        profile.rationale.append("product demos usually need action-level commentary without overexplaining")
    else:
        profile.rationale.append("planner fell back to mixed visual demo defaults")

    return profile


def normalize_video_profile(
    profile: VideoProfile | dict[str, Any] | None,
    *,
    requested_scene_threshold: float,
    requested_min_segment: float,
    requested_max_segment: float,
    requested_base_rate: str,
    requested_azure_style: str,
) -> VideoProfile:
    if profile is None:
        profile_obj = default_profile_for_type(
            DEFAULT_VIDEO_TYPE,
            requested_scene_threshold=requested_scene_threshold,
            requested_min_segment=requested_min_segment,
            requested_max_segment=requested_max_segment,
            requested_base_rate=requested_base_rate,
            requested_azure_style=requested_azure_style,
        )
    elif isinstance(profile, VideoProfile):
        profile_obj = profile
    else:
        profile_obj = VideoProfile.from_dict(profile)

    video_type = profile_obj.video_type if profile_obj.video_type in ALLOWED_VIDEO_TYPES else DEFAULT_VIDEO_TYPE
    defaulted = default_profile_for_type(
        video_type,
        requested_scene_threshold=requested_scene_threshold,
        requested_min_segment=requested_min_segment,
        requested_max_segment=requested_max_segment,
        requested_base_rate=requested_base_rate,
        requested_azure_style=requested_azure_style,
    )

    raw_seg = dict(defaulted.segmentation_policy)
    raw_seg.update(profile_obj.segmentation_policy)
    min_segment = clamp(float(raw_seg.get("min_segment", requested_min_segment)), 1.0, 15.0)
    max_segment = clamp(float(raw_seg.get("max_segment", requested_max_segment)), 3.0, 20.0)
    if max_segment < min_segment:
        max_segment = min(20.0, max(min_segment, min_segment + 1.0))
    scene_threshold = clamp(float(raw_seg.get("scene_threshold", requested_scene_threshold)), 0.20, 0.50)

    raw_style = dict(defaulted.style_policy)
    raw_style.update(profile_obj.style_policy)
    density = raw_style.get("narration_density", "balanced")
    if density not in ALLOWED_NARRATION_DENSITIES:
        density = defaulted.style_policy["narration_density"]
    focus = raw_style.get("narration_focus", "screen_change")
    if focus not in ALLOWED_NARRATION_FOCUS:
        focus = defaulted.style_policy["narration_focus"]

    azure_style = str(raw_style.get("azure_style") or requested_azure_style or defaulted.style_policy["azure_style"])
    base_rate = clamp_rate(str(raw_style.get("base_rate", requested_base_rate)), fallback=requested_base_rate)

    rationale = [str(item) for item in profile_obj.rationale] or list(defaulted.rationale)
    if video_type != profile_obj.video_type:
        rationale.append(f"video_type normalized to {video_type}")

    return VideoProfile(
        video_type=video_type,
        confidence=clamp(float(profile_obj.confidence), 0.0, 1.0),
        segmentation_policy={
            "scene_threshold": round(scene_threshold, 3),
            "min_segment": round(min_segment, 3),
            "max_segment": round(max_segment, 3),
        },
        style_policy={
            "narration_density": density,
            "narration_focus": focus,
            "azure_style": azure_style,
            "base_rate": base_rate,
        },
        rationale=rationale,
    )


def plan_video_profile(
    *,
    input_video: Path,
    requested_scene_threshold: float,
    requested_min_segment: float,
    requested_max_segment: float,
    requested_base_rate: str,
    requested_azure_style: str,
) -> VideoProfile:
    inferred_type = infer_video_type(input_video)
    profile = default_profile_for_type(
        inferred_type,
        requested_scene_threshold=requested_scene_threshold,
        requested_min_segment=requested_min_segment,
        requested_max_segment=requested_max_segment,
        requested_base_rate=requested_base_rate,
        requested_azure_style=requested_azure_style,
    )
    profile.rationale.insert(0, f"heuristic planner inferred video_type={inferred_type} from input name")
    return normalize_video_profile(
        profile,
        requested_scene_threshold=requested_scene_threshold,
        requested_min_segment=requested_min_segment,
        requested_max_segment=requested_max_segment,
        requested_base_rate=requested_base_rate,
        requested_azure_style=requested_azure_style,
    )
