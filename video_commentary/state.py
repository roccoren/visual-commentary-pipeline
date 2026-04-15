"""State and manifest models for the visual commentary pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .planner import VideoProfile


class SegmentStatus(str, Enum):
    PENDING = "pending"
    FRAMES_EXTRACTED = "frames_extracted"
    CONTENT_GENERATED = "content_generated"
    TTS_GENERATED = "tts_generated"
    ACCEPTED = "accepted"
    SKIPPED = "skipped"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    FAILED = "failed"


class Decision(str, Enum):
    ACCEPT = "accept"
    RETRY_NARRATION = "retry_narration"
    RETRY_TTS = "retry_tts"
    SPLIT_SEGMENT = "split_segment"
    MERGE_WITH_PREVIOUS = "merge_with_previous"
    SKIP_SEGMENT = "skip_segment"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


@dataclass
class RetryEntry:
    action: Decision
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentState:
    id: int
    start: float
    end: float
    duration: float
    status: SegmentStatus = SegmentStatus.PENDING
    frame_paths: list[str] = field(default_factory=list)
    title: str = ""
    semantic_group: str = ""
    narrative_role: str = ""
    visible_points: list[str] = field(default_factory=list)
    on_screen_text: list[str] = field(default_factory=list)
    vision_result: dict[str, Any] = field(default_factory=dict)
    draft_candidates: list[str] = field(default_factory=list)
    selected_draft: str = ""
    original_draft: str = ""
    rewritten_draft: str = ""
    rewrite_attempt_count: int = 0
    auto_retry_attempted: bool = False
    critic_feedback: list[str] = field(default_factory=list)
    raw_audio_path: str = ""
    raw_audio_duration: float = 0.0
    fitted_audio_path: str = ""
    fitted_audio_duration: float = 0.0
    duration_budget: float = 0.0
    duration_gap_ms: int = 0
    decision: Decision | None = None
    final_decision: Decision | None = None
    decision_reason: str = ""
    retry_history: list[RetryEntry] = field(default_factory=list)
    human_review_status: str = ""
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_segment(cls, *, seg_id: int, start: float, end: float, duration: float) -> "SegmentState":
        return cls(id=seg_id, start=start, end=end, duration=duration)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["decision"] = self.decision.value if self.decision else None
        data["final_decision"] = self.final_decision.value if self.final_decision else None
        data["retry_history"] = [
            {
                "action": item.action.value,
                "reason": item.reason,
                "details": item.details,
            }
            for item in self.retry_history
        ]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentState":
        retry_history = [
            RetryEntry(
                action=Decision(item["action"]),
                reason=item["reason"],
                details=item.get("details", {}),
            )
            for item in data.get("retry_history", [])
        ]
        decision = data.get("decision")
        final_decision = data.get("final_decision")
        return cls(
            id=int(data["id"]),
            start=float(data["start"]),
            end=float(data["end"]),
            duration=float(data["duration"]),
            status=SegmentStatus(data.get("status", SegmentStatus.PENDING.value)),
            frame_paths=list(data.get("frame_paths", [])),
            title=str(data.get("title", "")),
            semantic_group=str(data.get("semantic_group", "")),
            narrative_role=str(data.get("narrative_role", "")),
            visible_points=list(data.get("visible_points", [])),
            on_screen_text=list(data.get("on_screen_text", [])),
            vision_result=dict(data.get("vision_result", {})),
            draft_candidates=list(data.get("draft_candidates", [])),
            selected_draft=str(data.get("selected_draft", "")),
            original_draft=str(data.get("original_draft", "")),
            rewritten_draft=str(data.get("rewritten_draft", "")),
            rewrite_attempt_count=int(data.get("rewrite_attempt_count", 0)),
            auto_retry_attempted=bool(data.get("auto_retry_attempted", False)),
            critic_feedback=list(data.get("critic_feedback", [])),
            raw_audio_path=str(data.get("raw_audio_path", "")),
            raw_audio_duration=float(data.get("raw_audio_duration", 0.0)),
            fitted_audio_path=str(data.get("fitted_audio_path", "")),
            fitted_audio_duration=float(data.get("fitted_audio_duration", 0.0)),
            duration_budget=float(data.get("duration_budget", 0.0)),
            duration_gap_ms=int(data.get("duration_gap_ms", 0)),
            decision=Decision(decision) if decision else None,
            final_decision=Decision(final_decision) if final_decision else None,
            decision_reason=str(data.get("decision_reason", "")),
            retry_history=retry_history,
            human_review_status=str(data.get("human_review_status", "")),
            errors=list(data.get("errors", [])),
        )

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any]) -> "SegmentState":
        selected_draft = str(data.get("narration_zh", ""))
        raw_audio_path = str(data.get("audio_path", ""))
        fitted_audio_path = str(data.get("fitted_audio_path", ""))
        if fitted_audio_path and selected_draft:
            status = SegmentStatus.ACCEPTED
        elif raw_audio_path and selected_draft:
            status = SegmentStatus.TTS_GENERATED
        elif selected_draft:
            status = SegmentStatus.CONTENT_GENERATED
        elif data.get("frame_paths"):
            status = SegmentStatus.FRAMES_EXTRACTED
        else:
            status = SegmentStatus.PENDING

        return cls(
            id=int(data["id"]),
            start=float(data["start"]),
            end=float(data["end"]),
            duration=float(data["duration"]),
            status=status,
            frame_paths=list(data.get("frame_paths", [])),
            title=str(data.get("title", "")),
            semantic_group=str(data.get("semantic_group", "")),
            narrative_role=str(data.get("narrative_role", "")),
            visible_points=list(data.get("visible_points", [])),
            on_screen_text=list(data.get("on_screen_text", [])),
            selected_draft=selected_draft,
            original_draft=selected_draft,
            raw_audio_path=raw_audio_path,
            raw_audio_duration=float(data.get("audio_duration", 0.0)),
            fitted_audio_path=fitted_audio_path,
            fitted_audio_duration=float(data.get("fitted_audio_duration", 0.0)),
            decision=Decision.ACCEPT if status == SegmentStatus.ACCEPTED else None,
            final_decision=Decision.ACCEPT if status == SegmentStatus.ACCEPTED else None,
        )


@dataclass
class Manifest:
    version: str
    input_video: str
    output_video: str
    workdir: str
    scene_threshold: float
    min_segment: float
    max_segment: float
    segment_buffer: float
    base_rate: str
    azure_style: str
    duration: float
    video_profile: VideoProfile | None = None
    narrative_outline: str = ""
    status: str = "initialized"
    artifacts: dict[str, str] = field(default_factory=dict)
    segments: list[SegmentState] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "input_video": self.input_video,
            "output_video": self.output_video,
            "workdir": self.workdir,
            "scene_threshold": self.scene_threshold,
            "min_segment": self.min_segment,
            "max_segment": self.max_segment,
            "segment_buffer": self.segment_buffer,
            "base_rate": self.base_rate,
            "azure_style": self.azure_style,
            "duration": self.duration,
            "video_profile": self.video_profile.to_dict() if self.video_profile else None,
            "narrative_outline": self.narrative_outline,
            "status": self.status,
            "artifacts": self.artifacts,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        return cls(
            version=str(data["version"]),
            input_video=str(data["input_video"]),
            output_video=str(data["output_video"]),
            workdir=str(data["workdir"]),
            scene_threshold=float(data["scene_threshold"]),
            min_segment=float(data["min_segment"]),
            max_segment=float(data["max_segment"]),
            segment_buffer=float(data["segment_buffer"]),
            base_rate=str(data["base_rate"]),
            azure_style=str(data["azure_style"]),
            duration=float(data["duration"]),
            video_profile=VideoProfile.from_dict(data["video_profile"]) if data.get("video_profile") else None,
            narrative_outline=str(data.get("narrative_outline", "")),
            status=str(data.get("status", "initialized")),
            artifacts=dict(data.get("artifacts", {})),
            segments=[SegmentState.from_dict(item) for item in data.get("segments", [])],
        )

    @classmethod
    def from_legacy_segments(
        cls,
        segments: list[dict[str, Any]],
        *,
        source_path: Path | None = None,
    ) -> "Manifest":
        workdir = source_path.parent if source_path else Path(".")
        duration = max((float(item.get("end", 0.0)) for item in segments), default=0.0)
        return cls(
            version="1.0-legacy",
            input_video="",
            output_video="",
            workdir=str(workdir),
            scene_threshold=0.32,
            min_segment=3.0,
            max_segment=12.0,
            segment_buffer=0.35,
            base_rate="+0%",
            azure_style="professional",
            duration=duration,
            status="legacy-imported",
            artifacts={"manifest": str(source_path)} if source_path else {},
            segments=[SegmentState.from_legacy_dict(item) for item in segments],
        )

    @classmethod
    def from_json_data(cls, data: Any, *, source_path: Path | None = None) -> "Manifest":
        if isinstance(data, dict):
            return cls.from_dict(data)
        if isinstance(data, list):
            return cls.from_legacy_segments(data, source_path=source_path)
        raise TypeError(f"Unsupported manifest payload type: {type(data).__name__}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls.from_json_data(json.loads(path.read_text(encoding="utf-8")), source_path=path)

    def get_segment(self, segment_id: int) -> SegmentState:
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        raise KeyError(f"Unknown segment id: {segment_id}")
