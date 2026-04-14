"""Core helpers for the visual-first video commentary pipeline."""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

DEFAULT_TERM_MAP = {
    "ashure": "Azure",
    "azri": "Azure",
    "asheri": "Azure",
    "openaai": "OpenAI",
    "coher": "Cohere",
    "mistol": "Mistral",
    "deep seek": "DeepSeek",
    "copilot studio": "Copilot Studio",
    "ai air.com": "ai.azure.com",
    "ai. air.com": "ai.azure.com",
}


@dataclass
class Segment:
    id: int
    start: float
    end: float
    duration: float


@dataclass
class SegmentNarration:
    id: int
    start: float
    end: float
    duration: float
    title: str
    visible_points: List[str]
    on_screen_text: List[str]
    narration_zh: str
    frame_paths: List[str]
    audio_path: str = ""
    audio_duration: float = 0.0
    fitted_audio_path: str = ""
    fitted_audio_duration: float = 0.0


def normalize_terms(text: str, term_map: Mapping[str, str] | None = None) -> str:
    out = str(text)
    for bad, good in (term_map or DEFAULT_TERM_MAP).items():
        out = re.sub(re.escape(bad), good, out, flags=re.I)
    out = re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def build_segments(
    duration: float,
    cuts: Sequence[float],
    *,
    min_segment: float,
    max_segment: float,
) -> List[Segment]:
    points = [0.0] + [c for c in cuts if 0.05 <= c <= duration - 0.05] + [duration]
    raw: List[tuple[float, float]] = []
    for start, end in zip(points, points[1:]):
        if end - start <= 0.05:
            continue
        raw.append((start, end))

    merged: List[tuple[float, float]] = []
    carry_start: float | None = None
    for start, end in raw:
        span = end - start
        if carry_start is not None:
            start = carry_start
            span = end - start
            carry_start = None
        if not merged and span < min_segment:
            carry_start = start
            continue
        if merged and span < min_segment:
            prev_start, _prev_end = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    if carry_start is not None and merged:
        prev_start, _prev_end = merged[-1]
        merged[-1] = (prev_start, duration)

    final: List[Segment] = []
    seg_id = 1
    for start, end in merged:
        span = end - start
        if span <= max_segment:
            final.append(Segment(seg_id, round(start, 3), round(end, 3), round(span, 3)))
            seg_id += 1
            continue

        pieces = max(2, math.ceil(span / max_segment))
        step = span / pieces
        cursor = start
        for idx in range(pieces):
            piece_end = end if idx == pieces - 1 else cursor + step
            final.append(
                Segment(
                    id=seg_id,
                    start=round(cursor, 3),
                    end=round(piece_end, 3),
                    duration=round(piece_end - cursor, 3),
                )
            )
            seg_id += 1
            cursor = piece_end
    return final


def sample_times(segment: Segment) -> List[float]:
    if segment.duration <= 4:
        return [round(segment.start + segment.duration / 2, 3)]
    a = segment.start + segment.duration * 0.12
    b = segment.start + segment.duration * 0.50
    c = segment.end - segment.duration * 0.12
    return sorted(
        {
            round(max(segment.start + 0.05, min(point, segment.end - 0.05)), 3)
            for point in (a, b, c)
        }
    )


def atempo_chain(speedup: float) -> str:
    remaining = speedup
    filters: List[str] = []
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - round(remaining)) < 1e-9:
        filters.append(f"atempo={remaining:.1f}")
    else:
        filters.append(f"atempo={remaining:.5f}")
    return ",".join(filters)


def format_srt_timestamp(ts: float) -> str:
    milliseconds = int(round((ts - int(ts)) * 1000))
    total_seconds = int(ts)
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def build_srt_text(narrations: Iterable[Mapping[str, object] | SegmentNarration]) -> str:
    lines: List[str] = []
    for index, item in enumerate(narrations, start=1):
        record = asdict(item) if isinstance(item, SegmentNarration) else dict(item)
        lines.extend(
            [
                str(index),
                f"{format_srt_timestamp(float(record['start']))} --> {format_srt_timestamp(float(record['end']))}",
                str(record["narration_zh"]),
                "",
            ]
        )
    return "\n".join(lines)


def write_srt_file(narrations: Iterable[Mapping[str, object] | SegmentNarration], out_path: Path) -> None:
    out_path.write_text(build_srt_text(narrations), encoding="utf-8")


def build_azure_tts_ssml(
    *,
    text: str,
    voice: str,
    rate: str = "+0%",
    target_duration_ms: int | None = None,
    style: str | None = None,
) -> str:
    duration_node = (
        f'        <mstts:audioduration value="{int(target_duration_ms)}ms"/>\n'
        if target_duration_ms
        else ""
    )
    escaped_text = html.escape(text)
    prosody = f'<prosody rate="{rate}">{escaped_text}</prosody>'
    if style:
        content = f'<mstts:express-as style="{html.escape(style)}">{prosody}</mstts:express-as>'
    else:
        content = prosody
    return (
        "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" "
        "xmlns:mstts=\"https://www.w3.org/2001/mstts\" xml:lang=\"zh-CN\">\n"
        f"  <voice name=\"{html.escape(voice)}\">\n"
        f"{duration_node}"
        f"    {content}\n"
        "  </voice>\n"
        "</speak>"
    )


def serialize_manifest(narrations: Sequence[SegmentNarration]) -> str:
    return json.dumps([asdict(item) for item in narrations], ensure_ascii=False, indent=2)

