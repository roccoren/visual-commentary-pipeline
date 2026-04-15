# Visual Commentary Pipeline 使用说明

## 1. 适用场景

这个项目适合把下面这类视频做成 **基于视频画面的中文讲解版**：

- PPT / Deck 录屏
- Azure / Portal / 控制台操作演示
- 产品 walkthrough
- Dashboard / 可视化展示
- 架构方案讲解录像

它不是把原始语音翻译成中文，而是：

1. 看视频画面
2. 按视觉变化切段
3. 为每段生成中文讲解
4. 合成语音并卡进该段时间窗
5. 输出讲解版视频、字幕和 manifest

---

## 2. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

如果你的系统没有全局 `ffmpeg` / `ffprobe`，需要先安装它们；否则场景切段、抽帧和音频拟合都会失败。

---

## 3. 环境变量

### Azure OpenAI

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="***"
export AZURE_OPENAI_DEPLOYMENT="<vision-capable deployment>"
# optional
export AZURE_OPENAI_API_VERSION="2024-10-21"
```

### Azure Speech

```bash
export AZURE_SPEECH_KEY="***"
export AZURE_SPEECH_REGION="eastus"
# optional
export AZURE_SPEECH_VOICE="zh-CN-XiaoxiaoNeural"
```

---

## 4. 最常用的第一次运行

```bash
python scripts/visual_commentary_pipeline.py \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work \
  --scene-threshold 0.32 \
  --min-segment 3 \
  --max-segment 12 \
  --segment-buffer 0.35 \
  --base-rate "+0%" \
  --azure-style professional
```

或者安装 console script 后使用：

```bash
visual-commentary \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work
```

### 这次运行会做什么

第一次运行会按顺序执行：

1. 用 `ffprobe` 获取视频总时长
2. 用 `ffmpeg` 做 scene detection
3. 构建 segment plan
4. 写出 `commentary_manifest.json`
5. 对每个 segment：
   - 抽帧
   - 视觉理解
   - 生成 narration
   - 经过 QA gate
   - 必要时执行一次自动 rewrite retry
   - 做 TTS
   - 做 duration gate
6. 从 manifest 重建 accepted narration
7. 生成：
   - `commentary_zh.srt`
   - `commentary_zh.m4a`
   - `commentary_zh.mp4`

---

## 5. 推荐工作流（workflow explanation）

建议把这个项目理解成一个 **stateful workflow**，而不是“一次性脚本”。

### Step 1：先跑一次完整流程
先拿默认参数跑完，让系统生成：

- manifest
- 字幕
- 混合音轨
- 成片视频

### Step 2：优先检查 manifest
不要一上来只听最终视频。先看 `workdir/commentary_manifest.json`，重点看：

- 哪些 segment 被 `accepted`
- 哪些被 `needs_human_review`
- `decision` / `decision_reason`
- `critic_feedback`
- `retry_history`
- `original_draft` / `rewritten_draft`

这一步能最快告诉你问题是在：
- 切段
- 文案
- QA
- TTS 时长

### Step 3：如果只是单段有问题，用 resume / redo
这个项目已经是 manifest-driven 工作流，不需要每次都整视频重跑。

#### 只重做某一段 narration
```bash
visual-commentary \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work \
  --resume-from-manifest out/work/commentary_manifest.json \
  --segment-id 7 \
  --redo narration
```

#### 一次顺序重做多段 narration
```bash
visual-commentary \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work \
  --resume-from-manifest out/work/commentary_manifest.json \
  --segment-ids 2,4,17,24,52,64 \
  --redo narration
```

`--segment-ids` 会在同一个进程里按顺序处理这些 segment，最后只统一重建一次字幕、音轨和成片。它适合替代“同时起多个进程抢写同一个 manifest”的做法。

#### 只重做某一段 TTS
```bash
visual-commentary \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work \
  --resume-from-manifest out/work/commentary_manifest.json \
  --segment-id 7 \
  --redo tts
```

#### 重新做 vision → narration → tts
```bash
visual-commentary \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work \
  --resume-from-manifest out/work/commentary_manifest.json \
  --segment-id 7 \
  --redo vision
```

### Step 4：只有在切段本身不合理时，才 force replan
如果问题不在单段文案，而在 segment plan 本身，比如：

- 切得太碎
- 明显该拆没拆
- 明显该合没合

再重新规划：

```bash
visual-commentary \
  --input input.mp4 \
  --output out/commentary_zh.mp4 \
  --workdir out/work \
  --force-replan \
  --scene-threshold 0.36 \
  --min-segment 3.5 \
  --max-segment 10
```

---

## 6. redo 语义说明

- `--redo vision`
  - 重置该段的 frame / vision / narration / audio 派生结果
  - 再从视觉理解开始重跑该段后续步骤
- `--redo narration`
  - 保留现有 frames
  - 重新生成 narration、QA、rewrite、TTS、duration gate
- `--redo tts`
  - 保留现有 narration
  - 只重做语音合成和后续时长拟合

---

## 7. manifest 里最值得看的字段

每个 segment 都是一个状态化工作单元。建议重点看这些字段：

- `status`
- `decision`
- `final_decision`
- `decision_reason`
- `critic_feedback`
- `retry_history`
- `selected_draft`
- `original_draft`
- `rewritten_draft`
- `rewrite_attempt_count`
- `auto_retry_attempted`
- `human_review_status`

### 如何解读

#### `decision`
表示当前最近一次 gate 或 policy 给出的动作建议，例如：
- `accept`
- `retry_narration`
- `retry_tts`
- `needs_human_review`

#### `final_decision`
表示该段在当前运行结束后真正落下来的终态决策。

#### `critic_feedback`
表示 QA gate 为什么拦这段，例如：
- `empty narration`
- `repetitive narration`
- `too long or too dense narration`

#### `retry_history`
表示该段已经经历过哪些自动动作与决策。

---

## 8. 输出文件说明

在 `workdir` 下，常见产物包括：

- `commentary_manifest.json`
- `commentary_zh.srt`
- `commentary_zh.m4a`
- `frames/`
- `tts_raw/`
- `tts_fit/`

最终视频输出到你通过 `--output` 指定的位置。

---

## 9. 推荐的实际使用顺序

如果你是第一次上手，建议按这个顺序：

1. 用默认参数跑一次
2. 打开 manifest 看 segment 状态
3. 只对有问题的 segment 做 `--redo narration` 或 `--redo tts`
4. 如果问题普遍出在切段，再做 `--force-replan`
5. 满意后再导出最终成片用于交付

这个顺序比“每次整条视频重跑”更稳，也更省成本。

---

## 10. 什么时候该优先调参数

如果你发现：

- 很多段都 `needs_human_review`
- 很多段都太短、重复、赶
- 很多段都需要 TTS 压缩

那通常应优先看：

- `--scene-threshold`
- `--min-segment`
- `--max-segment`
- `--segment-buffer`

具体调法请配合阅读：
- [参数调优指南](parameter-tuning-guide-zh.md)
