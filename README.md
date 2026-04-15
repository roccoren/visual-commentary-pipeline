# Visual Commentary Pipeline

Standalone Python project for generating a **基于视频画面的中文讲解版** from a deck recording, UI walkthrough, or product demo video.

## Documentation

- [Solution principles and architecture diagram](docs/solution-principles-zh.md) (Chinese)
- [How to use / 使用说明](docs/how-to-use-zh.md) (Chinese)
- [Parameter tuning guide](docs/parameter-tuning-guide-zh.md) (Chinese)
- [P3-mini Video Planner design draft](docs/p3-mini-video-planner-design-zh.md) (Chinese)

## What it does

1. Detect visual segments from the input video via ffmpeg scene detection.
2. Extract representative keyframes for each segment.
3. Ask an Azure OpenAI vision-capable deployment to summarize what changed on screen.
4. Generate concise Chinese narration for each segment.
5. Synthesize speech with Azure Speech using SSML timing controls.
6. Fit per-segment clips into the slide time budget.
7. Export a dubbed video, mixed audio track, subtitle file, and manifest JSON.

## Why this project exists

This repo is intentionally **visual-first**, not a speech translation pipeline. It is designed for:

- slide deck recordings
- Azure / portal walkthroughs
- product demos
- dashboard demos
- architecture walkthrough videos

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Required environment variables

### Azure OpenAI

Used for vision understanding and narration generation (always required).

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="<vision-capable deployment>"
# optional
export AZURE_OPENAI_API_VERSION="2024-10-21"
```

### Azure Speech

Used for TTS synthesis (always required).

```bash
export AZURE_SPEECH_KEY="..."
export AZURE_SPEECH_REGION="eastus"
# optional
export AZURE_SPEECH_VOICE="zh-CN-XiaoxiaoNeural"
```

### Azure Content Understanding (optional — `--use-content-understanding`)

Replaces ffmpeg scene detection with model-driven video analysis.

```bash
export AZURE_CONTENT_UNDERSTANDING_ENDPOINT="https://<resource>.cognitiveservices.azure.com"
export AZURE_CONTENT_UNDERSTANDING_KEY="..."
# optional — defaults to 2025-11-01
export AZURE_CU_API_VERSION="2025-11-01"
# optional — model deployment for custom analyzer field extraction
export AZURE_CU_MODEL_DEPLOYMENT="gpt-4o"
```

### Azure Document Intelligence (optional — `--use-doc-intel`)

Provides precise OCR for UI screenshots in portal/dashboard videos.

```bash
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://<resource>.cognitiveservices.azure.com"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="..."
```

> **Note:** The LLM critic (`--use-llm-critic`) and LLM profiler (`--use-llm-profiler`) reuse the Azure OpenAI credentials above — no additional environment variables are needed.

## Run

### Standard pipeline (ffmpeg-based)

```bash
visual-commentary \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work
```

### Enhanced pipeline (LangGraph + Azure AI services)

```bash
visual-commentary \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work \
  --graph \
  --use-content-understanding \
  --use-llm-critic \
  --use-doc-intel \
  --use-llm-profiler
```

| Flag | What it does | Extra env vars needed |
|------|-------------|----------------------|
| `--graph` | Use LangGraph StateGraph orchestration | None |
| `--use-content-understanding` | Azure Content Understanding for video analysis | `AZURE_CONTENT_UNDERSTANDING_*` |
| `--use-llm-critic` | Two-layer QA: rules → LLM critic with rewrite loops | None (reuses Azure OpenAI) |
| `--use-doc-intel` | Document Intelligence OCR for UI text | `AZURE_DOCUMENT_INTELLIGENCE_*` |
| `--use-llm-profiler` | LLM vision-based video type profiling | None (reuses Azure OpenAI) |

All enhanced flags are opt-in and default to off. Each degrades gracefully if its Azure credentials are missing.

## Workflow explanation

Treat this project as a **manifest-driven workflow**, not a one-shot script.

Recommended usage pattern:

1. Run the full pipeline once to generate the initial manifest, subtitles, mixed audio, and commentary video.
2. Inspect `workdir/commentary_manifest.json` before judging the final video. Check `status`, `decision`, `final_decision`, `decision_reason`, `critic_feedback`, `retry_history`, and draft fields.
3. If only one segment is wrong, use `--resume-from-manifest` + `--segment-id` + `--redo narration|tts|vision` instead of rerunning the whole video.
4. Only use `--force-replan` when the segment boundaries themselves are wrong.

This keeps iteration cheap and makes the workflow debuggable.

Resume or redo a single segment:

```bash
visual-commentary \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work \
  --resume-from-manifest out/work/commentary_manifest.json \
  --segment-id 7 \
  --redo narration
```

Redo multiple segments in one sequential run:

```bash
visual-commentary \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work \
  --resume-from-manifest out/work/commentary_manifest.json \
  --segment-ids 2,4,17,24,52,64 \
  --redo narration
```

`--segment-ids` runs the selected segments sequentially inside one process and then rebuilds the final outputs once. It is intended as the safe alternative to launching multiple processes against the same manifest.

Redo semantics are explicit:
- `--redo vision` → reset the selected segment's derived frame/vision/narration/audio state and rerun that segment from visual understanding onward
- `--redo narration` → reuse extracted frames, regenerate narration + QA + rewrite + TTS for that segment
- `--redo tts` → reuse existing narration, regenerate only speech timing/audio for that segment

Accepted segments, narration rebuild, and final audio composition are all reconstructed from the manifest rather than transient in-memory lists.

The manifest is the single source of truth for:
- segment status progression
- decision / final_decision
- critic feedback and retry history
- original vs rewritten drafts
- whether automatic rewrite retry was attempted

## Timing control strategy

The pipeline combines three layers of timing control:

1. **Prompt-side length control**: the narration is generated to fit the current segment budget.
2. **Azure Speech SSML control**: `prosody rate` and `mstts:express-as` tune delivery.
3. **Azure `mstts:audioduration`**: requests a target clip duration before audio post-processing.

If a clip is still slightly too long, ffmpeg `atempo` is used as a fallback only for the final fit.

## Project layout

- `video_commentary/core.py` — reusable timing / SRT / SSML helpers
- `video_commentary/state.py` — segment state, manifest, and decision enums
- `video_commentary/segment_policy.py` — basic skip / merge boundary policy
- `video_commentary/duration_policy.py` — duration-fit decision policy
- `video_commentary/planner.py` — lightweight video type inference and policy defaults
- `video_commentary/pipeline.py` — Azure + ffmpeg orchestrator loop with resume/regenerate
- `video_commentary/graph.py` — LangGraph StateGraph orchestration (segment + pipeline graphs)
- `video_commentary/graph_state.py` — TypedDict state definitions for the graph
- `video_commentary/content_understanding.py` — Azure Content Understanding integration
- `video_commentary/llm_critic.py` — LLM-powered QA critic and narration rewriter
- `video_commentary/vision_enricher.py` — multi-model vision enrichment (CU + GPT-4o + Doc Intel)
- `scripts/visual_commentary_pipeline.py` — runnable script entrypoint
- `tests/test_video_commentary.py` — unit coverage for helpers + stateful workflow pieces
- `tests/test_graph.py` — LangGraph orchestration tests
- `tests/test_enhanced_pipeline.py` — tests for Content Understanding, LLM critic, and vision enricher

## Test

```bash
pytest tests/ -q
```
