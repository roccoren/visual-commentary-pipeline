"""End-to-end visual-first video commentary pipeline for Azure OpenAI + Azure Speech."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, List

import requests

try:
    import imageio_ffmpeg
except Exception as exc:  # pragma: no cover
    imageio_ffmpeg = None
    _IMAGEIO_IMPORT_ERROR = exc
else:
    _IMAGEIO_IMPORT_ERROR = None

from .core import (
    Segment,
    SegmentNarration,
    atempo_chain,
    build_azure_tts_ssml,
    build_segments,
    normalize_terms,
    sample_times,
    write_srt_file,
)
from .duration_policy import decide_duration_action
from .qa_gate import evaluate_narration_quality
from .segment_policy import decide_segment_action
from .state import Decision, Manifest, RetryEntry, SegmentState, SegmentStatus

DEFAULT_AZURE_OPENAI_API_VERSION = "2024-10-21"
DEFAULT_AZURE_SPEECH_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_AZURE_SPEECH_STYLE = "professional"
DEFAULT_OUTPUT_AUDIO_FORMAT = "audio-24khz-96kbitrate-mono-mp3"
MANIFEST_VERSION = "2.0"


def env_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def find_ffmpeg() -> str:
    if os.getenv("FFMPEG_BIN"):
        return os.environ["FFMPEG_BIN"]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()
    raise SystemExit(
        "ffmpeg not found. Install ffmpeg or imageio-ffmpeg. "
        f"imageio-ffmpeg import error: {_IMAGEIO_IMPORT_ERROR!r}"
    )


def find_ffprobe(ffmpeg_bin: str) -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe
    candidate = str(Path(ffmpeg_bin).with_name("ffprobe"))
    if Path(candidate).exists():
        return candidate
    raise SystemExit("ffprobe not found. Install ffprobe or make it adjacent to ffmpeg.")


def run(cmd: List[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def ffprobe_duration(ffprobe_bin: str, media_path: Path) -> float:
    proc = run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
    )
    return float(proc.stdout.strip())


def detect_scene_cuts(ffmpeg_bin: str, video_path: Path, threshold: float) -> List[float]:
    proc = run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-i",
            str(video_path),
            "-filter:v",
            f"select='gt(scene,{threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    matches = re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", text)
    return sorted({round(float(item), 3) for item in matches if float(item) > 0})


def extract_frame(ffmpeg_bin: str, video_path: Path, timestamp: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(out_path),
        ]
    )


def image_to_data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def azure_openai_uses_responses_api(api_version: str) -> bool:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", api_version)
    if not match:
        return False
    return match.group(1) >= "2025-04-01"


def build_azure_openai_vision_request(
    *,
    endpoint: str,
    deployment: str,
    api_version: str,
    user_prompt: str,
    frame_paths: List[Path],
) -> tuple[str, dict[str, Any], str]:
    if azure_openai_uses_responses_api(api_version):
        content = [{"type": "input_text", "text": user_prompt}]
        for path in frame_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(path),
                    "detail": "low",
                }
            )

        payload = {
            "model": deployment,
            "instructions": "你是擅长做产品演示和 slide 视频重述的中文讲解编辑。输出必须是有效 JSON。",
            "input": [{"type": "message", "role": "user", "content": content}],
            "temperature": 0.2,
            "max_output_tokens": 900,
            "text": {"format": {"type": "json_object"}},
        }
        return f"{endpoint}/openai/responses?api-version={api_version}", payload, "responses"

    content = [{"type": "text", "text": user_prompt}]
    for path in frame_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(path), "detail": "low"},
            }
        )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "你是擅长做产品演示和 slide 视频重述的中文讲解编辑。输出必须是有效 JSON。",
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
        "max_tokens": 900,
    }
    return (
        f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}",
        payload,
        "chat_completions",
    )


def extract_azure_openai_text(response_body: dict[str, Any], api_kind: str) -> str:
    if api_kind == "responses":
        output_text = response_body.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        for item in response_body.get("output", []):
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            parts = []
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    parts.append(content["text"])
            if parts:
                return "".join(parts)
        raise RuntimeError(f"Responses API did not return assistant text: {json.dumps(response_body, ensure_ascii=False)[:1000]}")

    return response_body["choices"][0]["message"]["content"]


def raise_for_status_with_context(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = (response.text or "").strip()
        suffix = f" Response: {detail[:2000]}" if detail else ""
        raise RuntimeError(
            f"Azure OpenAI request failed with status {response.status_code} for {response.url}.{suffix}"
        ) from exc


def call_azure_openai_vision(*, frame_paths: List[Path], segment: SegmentState, previous_narration: str) -> dict:
    endpoint = env_required("AZURE_OPENAI_ENDPOINT").rstrip("/")
    api_key = env_required("AZURE_OPENAI_API_KEY")
    deployment = env_required("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_OPENAI_API_VERSION)

    max_chars = max(18, min(90, int(segment.duration * 8)))
    user_prompt = textwrap.dedent(
        f"""
        你是视频讲解编辑。请仅根据提供的关键帧，为这一段视频生成“基于画面的中文讲解”。

        目标：
        1. 不要逐字朗读所有屏幕文字。
        2. 优先讲当前画面在展示什么、焦点是什么、和上一段相比新增了什么。
        3. 口吻要像自然中文口播，不要像 OCR 或图像描述。
        4. 讲解必须能在大约 {segment.duration:.1f} 秒内说完，尽量控制在 {max_chars} 个中文字符以内。
        5. 如画面信息很少，就简短说明“这里继续展示/演示……”。
        6. 专名保持英文，例如 Azure AI Foundry、OpenAI、Mistral、Cohere、DeepSeek、GitHub、Copilot Studio。

        时间窗：{segment.start:.3f}s - {segment.end:.3f}s
        上一段讲解：{previous_narration or '无'}

        只返回 JSON，不要 markdown，不要额外解释。格式：
        {{
          "title": "一句话标题",
          "visible_points": ["要点1", "要点2"],
          "on_screen_text": ["识别到的关键屏幕文本1", "关键文本2"],
          "narration_zh": "1到3句自然中文讲解"
        }}
        """
    ).strip()

    url, payload, api_kind = build_azure_openai_vision_request(
        endpoint=endpoint,
        deployment=deployment,
        api_version=api_version,
        user_prompt=user_prompt,
        frame_paths=frame_paths,
    )
    response = requests.post(
        url,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    raise_for_status_with_context(response)
    raw = extract_azure_openai_text(response.json(), api_kind)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise RuntimeError(f"Model did not return JSON: {raw[:500]}")
        parsed = json.loads(match.group(0))

    parsed.setdefault("title", f"segment_{segment.id}")
    parsed.setdefault("visible_points", [])
    parsed.setdefault("on_screen_text", [])
    parsed.setdefault("narration_zh", "这里展示了当前画面的核心内容。")
    parsed["title"] = normalize_terms(str(parsed["title"]))
    parsed["visible_points"] = [normalize_terms(str(item)) for item in parsed["visible_points"]][:6]
    parsed["on_screen_text"] = [normalize_terms(str(item)) for item in parsed["on_screen_text"]][:8]
    parsed["narration_zh"] = normalize_terms(str(parsed["narration_zh"]))
    return parsed


def azure_speech_token() -> str:
    key = env_required("AZURE_SPEECH_KEY")
    region = env_required("AZURE_SPEECH_REGION")
    response = requests.post(
        f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
        headers={"Ocp-Apim-Subscription-Key": key},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def synthesize_azure_tts(
    text: str,
    out_path: Path,
    *,
    voice: str,
    rate: str = "+0%",
    target_duration_ms: int | None = None,
    style: str | None = DEFAULT_AZURE_SPEECH_STYLE,
) -> None:
    token = azure_speech_token()
    region = env_required("AZURE_SPEECH_REGION")
    ssml = build_azure_tts_ssml(
        text=text,
        voice=voice,
        rate=rate,
        target_duration_ms=target_duration_ms,
        style=style,
    )
    response = requests.post(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": DEFAULT_OUTPUT_AUDIO_FORMAT,
            "User-Agent": "visual-commentary-pipeline",
        },
        data=ssml.encode("utf-8"),
        timeout=120,
    )
    response.raise_for_status()
    out_path.write_bytes(response.content)


def fit_audio_to_budget(
    ffmpeg_bin: str,
    ffprobe_bin: str,
    src_audio: Path,
    dst_audio: Path,
    *,
    budget_seconds: float,
) -> float:
    src_duration = ffprobe_duration(ffprobe_bin, src_audio)
    if src_duration <= budget_seconds * 1.02:
        shutil.copyfile(src_audio, dst_audio)
        return src_duration
    speedup = max(1.0, src_duration / max(0.6, budget_seconds))
    run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src_audio),
            "-filter:a",
            atempo_chain(speedup),
            "-ar",
            "24000",
            "-ac",
            "1",
            str(dst_audio),
        ]
    )
    return ffprobe_duration(ffprobe_bin, dst_audio)


def compose_commentary_track(
    ffmpeg_bin: str,
    narrations: List[SegmentNarration],
    *,
    full_duration: float,
    out_audio: Path,
    temp_dir: Path,
) -> None:
    filter_file = temp_dir / "amix_filter.txt"
    inputs = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-t",
        f"{full_duration:.3f}",
        "-i",
        "anullsrc=r=24000:cl=mono",
    ]
    chains: List[str] = []
    mix_inputs = ["[0:a]"]
    for index, item in enumerate(narrations, start=1):
        inputs.extend(["-i", item.fitted_audio_path])
        delay_ms = max(0, int(round(item.start * 1000)))
        chains.append(f"[{index}:a]adelay={delay_ms}|{delay_ms},volume=1.25[a{index}]")
        mix_inputs.append(f"[a{index}]")
    chains.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:duration=longest:normalize=0,"
        "loudnorm=I=-16:TP=-1.5:LRA=11[mix]"
    )
    filter_file.write_text(";\n".join(chains), encoding="utf-8")
    run(
        inputs
        + [
            "-filter_complex_script",
            str(filter_file),
            "-map",
            "[mix]",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(out_audio),
        ]
    )


def mux_video(ffmpeg_bin: str, video_path: Path, commentary_audio: Path, srt_path: Path, output_path: Path) -> None:
    run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(commentary_audio),
            "-i",
            str(srt_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=zho",
            "-shortest",
            str(output_path),
        ]
    )


def segment_state_to_narration(segment: SegmentState) -> SegmentNarration:
    return SegmentNarration(
        id=segment.id,
        start=segment.start,
        end=segment.end,
        duration=segment.duration,
        title=segment.title,
        visible_points=segment.visible_points,
        on_screen_text=segment.on_screen_text,
        narration_zh=segment.selected_draft,
        frame_paths=segment.frame_paths,
        audio_path=segment.raw_audio_path,
        audio_duration=round(segment.raw_audio_duration, 3),
        fitted_audio_path=segment.fitted_audio_path,
        fitted_audio_duration=round(segment.fitted_audio_duration, 3),
    )


def build_manifest(
    args: argparse.Namespace,
    *,
    input_video: Path,
    output_video: Path,
    workdir: Path,
    duration: float,
    segments: List[Segment],
) -> Manifest:
    return Manifest(
        version=MANIFEST_VERSION,
        input_video=str(input_video),
        output_video=str(output_video),
        workdir=str(workdir),
        scene_threshold=args.scene_threshold,
        min_segment=args.min_segment,
        max_segment=args.max_segment,
        segment_buffer=args.segment_buffer,
        base_rate=args.base_rate,
        azure_style=args.azure_style,
        duration=round(duration, 3),
        status="planned",
        artifacts={},
        segments=[
            SegmentState.from_segment(
                seg_id=segment.id,
                start=segment.start,
                end=segment.end,
                duration=segment.duration,
            )
            for segment in segments
        ],
    )


def accepted_segments_from_manifest(manifest: Manifest) -> list[SegmentState]:
    return [
        segment
        for segment in manifest.segments
        if segment.status == SegmentStatus.ACCEPTED and segment.fitted_audio_path and segment.selected_draft
    ]


def accepted_narrations_from_manifest(manifest: Manifest) -> list[SegmentNarration]:
    return [segment_state_to_narration(segment) for segment in accepted_segments_from_manifest(manifest)]


def get_previous_segment(manifest: Manifest, segment_id: int) -> SegmentState | None:
    previous: SegmentState | None = None
    for segment in manifest.segments:
        if segment.id == segment_id:
            return previous
        previous = segment
    return previous


def get_previous_accepted_narration(manifest: Manifest, segment_id: int) -> str:
    previous_text = ""
    for segment in manifest.segments:
        if segment.id == segment_id:
            return previous_text
        if segment.status == SegmentStatus.ACCEPTED and segment.selected_draft:
            previous_text = segment.selected_draft
    return previous_text


def should_process_segment(segment: SegmentState, *, redo: str | None, target_segment_id: int | None) -> bool:
    if target_segment_id is not None and segment.id != target_segment_id:
        return False
    if redo:
        return True
    return segment.status not in {SegmentStatus.ACCEPTED, SegmentStatus.SKIPPED}


def clear_segment_outputs(segment: SegmentState) -> None:
    segment.raw_audio_path = ""
    segment.raw_audio_duration = 0.0
    segment.fitted_audio_path = ""
    segment.fitted_audio_duration = 0.0
    segment.duration_gap_ms = 0
    segment.decision = None
    segment.final_decision = None
    segment.decision_reason = ""
    segment.errors = []
    segment.human_review_status = ""


def reset_segment_for_redo(segment: SegmentState, redo: str) -> None:
    if redo == "vision":
        segment.status = SegmentStatus.PENDING
        segment.frame_paths = []
        segment.title = ""
        segment.visible_points = []
        segment.on_screen_text = []
        segment.vision_result = {}
        segment.draft_candidates = []
        segment.selected_draft = ""
        segment.critic_feedback = []
        clear_segment_outputs(segment)
    elif redo == "narration":
        segment.status = SegmentStatus.FRAMES_EXTRACTED
        segment.title = ""
        segment.visible_points = []
        segment.on_screen_text = []
        segment.vision_result = {}
        segment.draft_candidates = []
        segment.selected_draft = ""
        segment.original_draft = ""
        segment.rewritten_draft = ""
        segment.rewrite_attempt_count = 0
        segment.auto_retry_attempted = False
        segment.critic_feedback = []
        clear_segment_outputs(segment)
    elif redo == "tts":
        segment.status = SegmentStatus.CONTENT_GENERATED
        clear_segment_outputs(segment)


def note_segment_decision(
    segment: SegmentState,
    *,
    decision: Decision,
    reason: str,
    details: dict[str, Any] | None = None,
) -> None:
    segment.decision = decision
    segment.decision_reason = reason
    segment.retry_history.append(
        RetryEntry(
            action=decision,
            reason=reason,
            details=details or {},
        )
    )


def run_boundary_step(manifest: Manifest, segment: SegmentState, *, manifest_path: Path) -> bool:
    boundary = decide_segment_action(segment, previous_segment=get_previous_segment(manifest, segment.id))
    if boundary.decision == Decision.SKIP_SEGMENT:
        segment.status = SegmentStatus.SKIPPED
        note_segment_decision(segment, decision=boundary.decision, reason=boundary.reason)
        manifest.save(manifest_path)
        return False

    if boundary.decision == Decision.MERGE_WITH_PREVIOUS:
        segment.status = SegmentStatus.NEEDS_HUMAN_REVIEW
        segment.human_review_status = "pending-merge-review"
        note_segment_decision(segment, decision=boundary.decision, reason=boundary.reason)
        manifest.save(manifest_path)
        return False

    return True


def run_frame_extraction_step(
    segment: SegmentState,
    *,
    input_video: Path,
    frames_dir: Path,
    ffmpeg_bin: str,
) -> None:
    seg_dir = frames_dir / f"seg_{segment.id:03d}"
    seg_dir.mkdir(exist_ok=True)
    frame_paths: List[str] = []
    samples = sample_times(Segment(id=segment.id, start=segment.start, end=segment.end, duration=segment.duration))
    for index, timestamp in enumerate(samples, start=1):
        frame_path = seg_dir / f"frame_{index:02d}.jpg"
        extract_frame(ffmpeg_bin, input_video, timestamp, frame_path)
        frame_paths.append(str(frame_path))
    segment.frame_paths = frame_paths
    segment.status = SegmentStatus.FRAMES_EXTRACTED


def run_vision_step(manifest: Manifest, segment: SegmentState) -> None:
    vision = call_azure_openai_vision(
        frame_paths=[Path(path) for path in segment.frame_paths],
        segment=segment,
        previous_narration=get_previous_accepted_narration(manifest, segment.id),
    )
    segment.vision_result = vision
    segment.title = vision["title"]
    segment.visible_points = vision["visible_points"]
    segment.on_screen_text = vision["on_screen_text"]


def run_narration_step(segment: SegmentState) -> None:
    narration = normalize_terms(str(segment.vision_result.get("narration_zh", "这里展示了当前画面的核心内容。")))
    segment.draft_candidates = [narration]
    segment.selected_draft = narration
    segment.original_draft = narration
    segment.status = SegmentStatus.CONTENT_GENERATED


def rewrite_narration_once(
    *,
    narration: str,
    critic_feedback: list[str],
    decision_reason: str,
    duration_seconds: float,
    previous_narration: str,
) -> str:
    rewritten = narration.strip()
    if "empty narration" in critic_feedback or not rewritten:
        rewritten = "这里继续展示当前步骤的关键内容。"
    elif "repetitive narration" in critic_feedback:
        previous_clean = previous_narration.strip()
        if previous_clean and rewritten == previous_clean:
            rewritten = "这里进一步展示了与上一段不同的当前操作细节。"
        else:
            rewritten = f"这里进一步说明当前画面的新增内容：{rewritten}".strip()
    elif "too long or too dense narration" in critic_feedback:
        max_chars = max(18, min(90, int(duration_seconds * 8)))
        rewritten = rewritten[:max_chars].rstrip("，、；： ")
        if not rewritten:
            rewritten = "这里继续展示当前步骤。"

    rewritten = normalize_terms(rewritten)
    if previous_narration and rewritten.strip() == previous_narration.strip():
        rewritten = normalize_terms("这里继续展示当前步骤的新增内容。")
    return rewritten


def run_qa_gate_step(manifest: Manifest, segment: SegmentState) -> None:
    previous_narration = get_previous_accepted_narration(manifest, segment.id)
    qa_result = evaluate_narration_quality(
        narration=segment.selected_draft,
        previous_narration=previous_narration,
        duration_seconds=segment.duration,
    )
    segment.critic_feedback = qa_result.feedback
    if qa_result.passed:
        segment.final_decision = Decision.ACCEPT
        return

    note_segment_decision(
        segment,
        decision=qa_result.decision,
        reason=qa_result.reason,
        details=qa_result.details,
    )

    if qa_result.decision == Decision.RETRY_NARRATION and not segment.auto_retry_attempted and segment.rewrite_attempt_count < 1:
        rewritten = rewrite_narration_once(
            narration=segment.selected_draft,
            critic_feedback=segment.critic_feedback,
            decision_reason=qa_result.reason,
            duration_seconds=segment.duration,
            previous_narration=previous_narration,
        )
        segment.rewritten_draft = rewritten
        segment.selected_draft = rewritten
        segment.draft_candidates.append(rewritten)
        segment.rewrite_attempt_count += 1
        segment.auto_retry_attempted = True

        second_qa = evaluate_narration_quality(
            narration=segment.selected_draft,
            previous_narration=previous_narration,
            duration_seconds=segment.duration,
        )
        segment.critic_feedback = second_qa.feedback
        note_segment_decision(
            segment,
            decision=second_qa.decision,
            reason=f"post-rewrite qa: {second_qa.reason}",
            details={"phase": "post-rewrite", **second_qa.details},
        )
        if second_qa.passed:
            segment.decision = Decision.ACCEPT
            segment.final_decision = Decision.ACCEPT
            segment.decision_reason = second_qa.reason
            segment.status = SegmentStatus.CONTENT_GENERATED
            segment.human_review_status = ""
            return

        segment.decision = Decision.NEEDS_HUMAN_REVIEW
        segment.final_decision = Decision.NEEDS_HUMAN_REVIEW
        segment.decision_reason = second_qa.reason
        segment.status = SegmentStatus.NEEDS_HUMAN_REVIEW
        segment.human_review_status = "qa-rewrite-failed"
        return

    segment.final_decision = qa_result.decision
    if qa_result.decision == Decision.RETRY_NARRATION:
        segment.status = SegmentStatus.NEEDS_HUMAN_REVIEW
        segment.human_review_status = "qa-retry-narration"
    else:
        segment.status = SegmentStatus.NEEDS_HUMAN_REVIEW
        segment.human_review_status = "qa-human-review"


def run_tts_step(
    segment: SegmentState,
    *,
    raw_audio_dir: Path,
    fit_audio_dir: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    base_rate: str,
    azure_style: str,
    segment_buffer: float,
) -> None:
    raw_audio = raw_audio_dir / f"seg_{segment.id:03d}.mp3"
    fitted_audio = fit_audio_dir / f"seg_{segment.id:03d}.mp3"
    target_duration_ms = int(max(800, round((segment.duration - segment_buffer) * 1000)))
    synthesize_azure_tts(
        segment.selected_draft,
        raw_audio,
        voice=os.getenv("AZURE_SPEECH_VOICE", DEFAULT_AZURE_SPEECH_VOICE),
        rate=base_rate,
        target_duration_ms=target_duration_ms,
        style=azure_style,
    )
    raw_duration = ffprobe_duration(ffprobe_bin, raw_audio)
    budget_seconds = max(0.8, segment.duration - segment_buffer)
    fitted_duration = fit_audio_to_budget(
        ffmpeg_bin,
        ffprobe_bin,
        raw_audio,
        fitted_audio,
        budget_seconds=budget_seconds,
    )
    segment.raw_audio_path = str(raw_audio)
    segment.raw_audio_duration = round(raw_duration, 3)
    segment.fitted_audio_path = str(fitted_audio)
    segment.fitted_audio_duration = round(fitted_duration, 3)
    segment.duration_budget = round(budget_seconds, 3)
    segment.status = SegmentStatus.TTS_GENERATED


def run_duration_gate_step(segment: SegmentState) -> None:
    duration_decision = decide_duration_action(
        budget_seconds=segment.duration_budget,
        fitted_duration_seconds=segment.fitted_audio_duration,
    )
    segment.duration_gap_ms = duration_decision.gap_ms
    note_segment_decision(
        segment,
        decision=duration_decision.decision,
        reason=duration_decision.reason,
        details={
            "budget_seconds": segment.duration_budget,
            "fitted_duration_seconds": segment.fitted_audio_duration,
            "gap_ms": duration_decision.gap_ms,
        },
    )
    if duration_decision.decision == Decision.ACCEPT:
        segment.status = SegmentStatus.ACCEPTED
    else:
        segment.status = SegmentStatus.NEEDS_HUMAN_REVIEW
        segment.human_review_status = "timing-adjustment-needed"


def process_segment(
    manifest: Manifest,
    segment: SegmentState,
    *,
    input_video: Path,
    frames_dir: Path,
    raw_audio_dir: Path,
    fit_audio_dir: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    args: argparse.Namespace,
    manifest_path: Path,
) -> None:
    if not run_boundary_step(manifest, segment, manifest_path=manifest_path):
        return

    try:
        if segment.status == SegmentStatus.PENDING:
            run_frame_extraction_step(
                segment,
                input_video=input_video,
                frames_dir=frames_dir,
                ffmpeg_bin=ffmpeg_bin,
            )
            manifest.save(manifest_path)

        if segment.status == SegmentStatus.FRAMES_EXTRACTED:
            run_vision_step(manifest, segment)
            manifest.save(manifest_path)
            run_narration_step(segment)
            manifest.save(manifest_path)
            run_qa_gate_step(manifest, segment)
            manifest.save(manifest_path)

        if segment.status == SegmentStatus.CONTENT_GENERATED:
            run_tts_step(
                segment,
                raw_audio_dir=raw_audio_dir,
                fit_audio_dir=fit_audio_dir,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                base_rate=args.base_rate,
                azure_style=args.azure_style,
                segment_buffer=args.segment_buffer,
            )
            manifest.save(manifest_path)

        if segment.status == SegmentStatus.TTS_GENERATED:
            run_duration_gate_step(segment)
            manifest.save(manifest_path)

    except Exception as exc:
        segment.status = SegmentStatus.FAILED
        segment.errors.append(str(exc))
        manifest.save(manifest_path)
        raise


def finalize_outputs(manifest: Manifest, *, ffmpeg_bin: str, output_video: Path, workdir: Path) -> dict[str, Any]:
    narrations = accepted_narrations_from_manifest(manifest)

    srt_path = workdir / "commentary_zh.srt"
    audio_mix_path = workdir / "commentary_zh.m4a"
    manifest_path = workdir / "commentary_manifest.json"

    write_srt_file(narrations, srt_path)
    compose_commentary_track(
        ffmpeg_bin,
        narrations,
        full_duration=manifest.duration,
        out_audio=audio_mix_path,
        temp_dir=workdir,
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)
    mux_video(ffmpeg_bin, Path(manifest.input_video), audio_mix_path, srt_path, output_video)

    manifest.artifacts.update(
        {
            "manifest": str(manifest_path),
            "srt": str(srt_path),
            "audio": str(audio_mix_path),
            "output_video": str(output_video),
        }
    )
    manifest.status = "completed"
    manifest.save(manifest_path)
    return {
        "input_video": manifest.input_video,
        "output_video": str(output_video),
        "manifest": str(manifest_path),
        "srt": str(srt_path),
        "audio": str(audio_mix_path),
        "segments": len(narrations),
        "duration": round(manifest.duration, 3),
        "needs_review": len(
            [segment for segment in manifest.segments if segment.status == SegmentStatus.NEEDS_HUMAN_REVIEW]
        ),
    }


def load_or_create_manifest(
    args: argparse.Namespace,
    *,
    input_video: Path,
    output_video: Path,
    workdir: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    manifest_path: Path,
) -> Manifest:
    if args.resume_from_manifest:
        return Manifest.load(Path(args.resume_from_manifest).expanduser().resolve())
    if manifest_path.exists() and not args.force_replan:
        return Manifest.load(manifest_path)

    duration = ffprobe_duration(ffprobe_bin, input_video)
    cuts = detect_scene_cuts(ffmpeg_bin, input_video, args.scene_threshold)
    segments = build_segments(
        duration,
        cuts,
        min_segment=args.min_segment,
        max_segment=args.max_segment,
    )
    manifest = build_manifest(
        args,
        input_video=input_video,
        output_video=output_video,
        workdir=workdir,
        duration=duration,
        segments=segments,
    )
    manifest.save(manifest_path)
    return manifest


def maybe_apply_redo(manifest: Manifest, args: argparse.Namespace, *, manifest_path: Path) -> None:
    if args.segment_id is None or args.redo is None:
        return

    target = manifest.get_segment(args.segment_id)
    reset_segment_for_redo(target, args.redo)
    note_segment_decision(
        target,
        decision=Decision.RETRY_NARRATION if args.redo in {"vision", "narration"} else Decision.RETRY_TTS,
        reason=f"manual redo requested: {args.redo}",
    )
    manifest.save(manifest_path)


def narrate_video(args: argparse.Namespace) -> dict:
    ffmpeg_bin = find_ffmpeg()
    ffprobe_bin = find_ffprobe(ffmpeg_bin)

    input_video = Path(args.input).expanduser().resolve()
    output_video = Path(args.output).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    frames_dir = workdir / "frames"
    raw_audio_dir = workdir / "tts_raw"
    fit_audio_dir = workdir / "tts_fit"
    workdir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    raw_audio_dir.mkdir(exist_ok=True)
    fit_audio_dir.mkdir(exist_ok=True)
    manifest_path = workdir / "commentary_manifest.json"

    manifest = load_or_create_manifest(
        args,
        input_video=input_video,
        output_video=output_video,
        workdir=workdir,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        manifest_path=manifest_path,
    )
    maybe_apply_redo(manifest, args, manifest_path=manifest_path)

    manifest.status = "running"
    manifest.save(manifest_path)

    for segment in manifest.segments:
        if not should_process_segment(segment, redo=args.redo, target_segment_id=args.segment_id):
            continue

        process_segment(
            manifest,
            segment,
            input_video=input_video,
            frames_dir=frames_dir,
            raw_audio_dir=raw_audio_dir,
            fit_audio_dir=fit_audio_dir,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            args=args,
            manifest_path=manifest_path,
        )

        print(
            f"[segment {segment.id:03d}] {segment.start:.2f}-{segment.end:.2f}s | "
            f"status={segment.status.value} decision={(segment.decision.value if segment.decision else 'none')}"
        )

    return finalize_outputs(manifest, ffmpeg_bin=ffmpeg_bin, output_video=output_video, workdir=workdir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Chinese commentary video from visual slide/demo content."
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--workdir", required=True, help="Working directory")
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.32,
        help="ffmpeg scene cut threshold (0.2~0.5 typical)",
    )
    parser.add_argument("--min-segment", type=float, default=3.0, help="Merge segments shorter than this")
    parser.add_argument("--max-segment", type=float, default=12.0, help="Split segments longer than this")
    parser.add_argument(
        "--segment-buffer",
        type=float,
        default=0.35,
        help="Reserve silence inside each segment",
    )
    parser.add_argument(
        "--base-rate",
        default="+0%",
        help="Base Azure Speech SSML prosody rate, e.g. +5%%",
    )
    parser.add_argument(
        "--azure-style",
        default=DEFAULT_AZURE_SPEECH_STYLE,
        help="Azure Speech mstts:express-as style, e.g. professional / calm / cheerful",
    )
    parser.add_argument(
        "--resume-from-manifest",
        help="Resume from an existing manifest JSON",
    )
    parser.add_argument(
        "--segment-id",
        type=int,
        help="Restrict processing to one segment id",
    )
    parser.add_argument(
        "--redo",
        choices=["vision", "narration", "tts"],
        help="Redo a specific stage for the selected segment",
    )
    parser.add_argument(
        "--force-replan",
        action="store_true",
        help="Ignore existing manifest in workdir and rebuild segment plan",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    print(json.dumps(narrate_video(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
