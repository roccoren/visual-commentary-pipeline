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
    serialize_manifest,
    write_srt_file,
)

DEFAULT_AZURE_OPENAI_API_VERSION = "2024-10-21"
DEFAULT_AZURE_SPEECH_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_AZURE_SPEECH_STYLE = "professional"
DEFAULT_OUTPUT_AUDIO_FORMAT = "audio-24khz-96kbitrate-mono-mp3"


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


def call_azure_openai_vision(*, frame_paths: List[Path], segment: Segment, previous_narration: str) -> dict:
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

    duration = ffprobe_duration(ffprobe_bin, input_video)
    cuts = detect_scene_cuts(ffmpeg_bin, input_video, args.scene_threshold)
    segments = build_segments(
        duration,
        cuts,
        min_segment=args.min_segment,
        max_segment=args.max_segment,
    )

    narrations: List[SegmentNarration] = []
    previous_narration = ""
    for segment in segments:
        seg_dir = frames_dir / f"seg_{segment.id:03d}"
        seg_dir.mkdir(exist_ok=True)
        frame_paths: List[Path] = []
        for index, timestamp in enumerate(sample_times(segment), start=1):
            frame_path = seg_dir / f"frame_{index:02d}.jpg"
            extract_frame(ffmpeg_bin, input_video, timestamp, frame_path)
            frame_paths.append(frame_path)

        vision = call_azure_openai_vision(
            frame_paths=frame_paths,
            segment=segment,
            previous_narration=previous_narration,
        )
        narration = SegmentNarration(
            id=segment.id,
            start=segment.start,
            end=segment.end,
            duration=segment.duration,
            title=vision["title"],
            visible_points=vision["visible_points"],
            on_screen_text=vision["on_screen_text"],
            narration_zh=vision["narration_zh"],
            frame_paths=[str(path) for path in frame_paths],
        )

        raw_audio = raw_audio_dir / f"seg_{segment.id:03d}.mp3"
        fitted_audio = fit_audio_dir / f"seg_{segment.id:03d}.mp3"
        target_duration_ms = int(max(800, round((segment.duration - args.segment_buffer) * 1000)))
        synthesize_azure_tts(
            narration.narration_zh,
            raw_audio,
            voice=os.getenv("AZURE_SPEECH_VOICE", DEFAULT_AZURE_SPEECH_VOICE),
            rate=args.base_rate,
            target_duration_ms=target_duration_ms,
            style=args.azure_style,
        )
        raw_duration = ffprobe_duration(ffprobe_bin, raw_audio)
        fitted_duration = fit_audio_to_budget(
            ffmpeg_bin,
            ffprobe_bin,
            raw_audio,
            fitted_audio,
            budget_seconds=max(0.8, segment.duration - args.segment_buffer),
        )
        narration.audio_path = str(raw_audio)
        narration.audio_duration = round(raw_duration, 3)
        narration.fitted_audio_path = str(fitted_audio)
        narration.fitted_audio_duration = round(fitted_duration, 3)
        narrations.append(narration)
        previous_narration = narration.narration_zh

        print(
            f"[segment {segment.id:03d}] {segment.start:.2f}-{segment.end:.2f}s | {narration.narration_zh}"
        )

    srt_path = workdir / "commentary_zh.srt"
    manifest_path = workdir / "commentary_manifest.json"
    audio_mix_path = workdir / "commentary_zh.m4a"

    write_srt_file(narrations, srt_path)
    manifest_path.write_text(serialize_manifest(narrations), encoding="utf-8")
    compose_commentary_track(
        ffmpeg_bin,
        narrations,
        full_duration=duration,
        out_audio=audio_mix_path,
        temp_dir=workdir,
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)
    mux_video(ffmpeg_bin, input_video, audio_mix_path, srt_path, output_video)
    return {
        "input_video": str(input_video),
        "output_video": str(output_video),
        "manifest": str(manifest_path),
        "srt": str(srt_path),
        "audio": str(audio_mix_path),
        "segments": len(narrations),
        "duration": round(duration, 3),
    }


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
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    print(json.dumps(narrate_video(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

