# P3-mini Video Planner 设计草案

## 1. 背景

当前这条分支已经完成并签收：

- P0：stateful manifest / resume / regenerate
- P1-A：manifest single source of truth
- P1-B：explicit step orchestration
- P1-C：minimal QA gate
- P2-mini：single rewrite retry

因此，当前系统的强项已经很明确：

- 它是一个 **segment-centric / stateful / retryable / resumable** 的执行器
- 它已经具备稳定的 segment 级执行闭环
- 它不是一个纯脚本堆砌，而是一个 manifest-driven workflow

但有一块还没有真正落地：

> **Video Planner**
>
> - 识别视频类型
> - 设定 segmentation policy
> - 设定 narration / TTS style policy

也就是说，当前系统更像：

> 先用固定规则切段 + 固定风格生成，再做 segment 级 stateful 执行

而不是：

> 先识别视频类型 → 产出 segmentation/style policy → 再执行

P3-mini 的目的，就是以**最小可控范围**把这块补上，但不扩成重型 planner / multi-agent 系统。

---

## 2. 设计目标

P3-mini 只解决一件事：

> 在正式切段和 narration 之前，增加一个非常轻量的 **pre-run video planner**。

planner 的输出只用于两类决策：

1. **Segmentation profile**
   - 决定切段偏好
   - 影响 `scene-threshold` / `min-segment` / `max-segment`

2. **Style profile**
   - 决定讲解密度与口吻
   - 影响 narration prompt wording / TTS style / base rate

它**不负责执行**任何复杂动作，也**不做多轮 planning**。

---

## 3. 明确不做什么

P3-mini 明确不做以下内容：

- 不做 multi-agent planner
- 不做 planner → critic → planner 的多轮协商
- 不做完整视频语义理解图谱
- 不做显式 split / merge 的自动执行扩张
- 不做 timeline-level global optimization
- 不做复杂 prompt tree
- 不做 UI / dashboard / talking-head 的全量 taxonomy 设计

换句话说，P3-mini 只是一层 **lightweight pre-run profiling**，不是重型调度系统。

---

## 4. 最小功能范围

## 4.1 新增入口：`plan_video_profile(...)`

新增一个最小 planner 函数，例如：

```python
def plan_video_profile(
    *,
    input_video: Path,
    ffmpeg_bin: str,
    sample_frame_dir: Path,
    requested_scene_threshold: float,
    requested_min_segment: float,
    requested_max_segment: float,
    requested_base_rate: str,
    requested_azure_style: str,
) -> VideoProfile:
    ...
```

它做的事：

1. 从视频头部 + 少量全局位置抽几张 sample frames
2. 用一个轻量 vision prompt 判断视频类型
3. 输出一个 `VideoProfile`

---

## 4.2 输出 schema：`VideoProfile`

建议增加结构：

```python
@dataclass
class VideoProfile:
    video_type: str
    confidence: float
    segmentation_policy: dict[str, Any]
    style_policy: dict[str, Any]
    rationale: list[str]
```

### 推荐的最小 `video_type`

P3-mini 不要设计太多类型，先只保留这 5 类：

- `deck_recording`
- `product_demo`
- `portal_walkthrough`
- `dashboard_demo`
- `mixed_visual_demo`

如果模型判断不稳，就落到：

- `mixed_visual_demo`

这比引入大量细分类更稳。

---

## 4.3 planner 输出影响哪些参数

### Segmentation policy

最小只调三项：

- `scene_threshold`
- `min_segment`
- `max_segment`

例如：

- `deck_recording`
  - scene_threshold 可以稍低
  - min_segment 可稍长
  - max_segment 可稍长
- `portal_walkthrough`
  - scene_threshold 稍高
  - min_segment 稍短
  - max_segment 稍短
- `dashboard_demo`
  - 中间值，避免切过碎

### Style policy

最小只调四项：

- `narration_density`: `concise | balanced | explanatory`
- `narration_focus`: `screen_change | operation_step | summary_first`
- `azure_style`
- `base_rate`

注意：
- P3-mini 不直接让 planner 接管整个 narration prompt
- 只通过少量 profile 字段去改变 prompt wording 和 TTS defaults

---

## 5. Manifest 落点

planner 结果必须写入 manifest root，而不是只存在运行时内存里。

建议新增：

```json
{
  "video_profile": {
    "video_type": "portal_walkthrough",
    "confidence": 0.82,
    "segmentation_policy": {
      "scene_threshold": 0.36,
      "min_segment": 2.5,
      "max_segment": 9.0
    },
    "style_policy": {
      "narration_density": "balanced",
      "narration_focus": "operation_step",
      "azure_style": "professional",
      "base_rate": "+2%"
    },
    "rationale": [
      "screen content is mostly portal UI",
      "changes are operation-driven rather than slide-driven"
    ]
  }
}
```

这样做的好处：

- resume 时可复用 planner 结果
- reviewer/human 可以直接看到 planner 判断
- single source of truth 仍然成立
- 后续如果要 redo / replan，边界更清晰

---

## 6. 接入点

P3-mini 只接两处，不多接：

### 接入点 1：切段前

在 `load_or_create_manifest()` 或其前置步骤中：

1. planner 先跑一次
2. planner 输出 profile
3. 用 profile 覆盖默认的 segmentation 参数
4. 再调用：
   - `detect_scene_cuts()`
   - `build_segments()`

### 接入点 2：narration / TTS 默认值

在 `call_azure_openai_vision()` 与 `run_tts_step()` 中，读取 `manifest.video_profile.style_policy`：

- 轻量调整 prompt wording
- 调整 `azure_style`
- 调整 `base_rate`

注意：
- 不要在太多函数间传递大量新参数
- 优先从 manifest 读取，保持 single source of truth

---

## 7. 推荐实现切法

为了控制风险，建议分成两步，但在同一个 P3-mini 里实现最小闭环。

### Step A：planner 产出 profile

新增文件例如：

- `video_commentary/planner.py`

包含：

- `VideoProfile`
- `sample_planner_frames(...)`
- `plan_video_profile(...)`
- `apply_video_profile_defaults(...)`

### Step B：pipeline 接入 profile

最小改动：

- `state.py`
  - Manifest 加 `video_profile`
- `pipeline.py`
  - 创建 manifest 前先 plan
  - narration / TTS 读取 profile
- `tests/test_video_commentary.py`
  - planner schema
  - planner default application
  - manifest serialization / roundtrip
  - pipeline 读取 profile 行为

---

## 8. 分工安排

遵循已经验证有效的协作方式：

### Claw（implementer）
负责：

- 写 `planner.py`
- 改 `state.py` / `pipeline.py`
- 补测试
- 本地跑 `pytest -q`
- 提交并 push 到同一 GitHub 分支

### Hermes（reviewer / architect）
负责：

- 审设计是否过度扩张
- 审 manifest schema 是否清晰
- 审 planner 接入点是否保持最小侵入
- 审测试是否覆盖核心边界
- 不与 Claw 并行写同一实现面

### Rocco（product / final direction）
负责：

- 确认 planner 分类粒度是否够用
- 确认文档是否清晰可用
- 决定 P3-mini 之后是否继续做更强 planner

---

## 9. 验收标准

P3-mini 完成的标准应当非常具体。

### 必须有

1. 存在显式 `plan_video_profile(...)`
2. manifest root 有 `video_profile`
3. planner 结果真的影响：
   - segmentation 参数
   - narration/TTS 默认风格
4. planner 结果可序列化 / 反序列化 / resume
5. 有测试覆盖最小闭环

### 必须没有

1. 不出现 planner 自旋
2. 不出现 multi-agent expansion
3. 不引入复杂 orchestration graph
4. 不把 split/merge 自动执行面继续扩张

---

## 10. 推荐测试清单

至少覆盖：

1. `VideoProfile` roundtrip
2. planner 默认 fallback 到 `mixed_visual_demo`
3. profile 正确覆盖 segmentation 参数
4. profile 正确影响 style defaults
5. resume 时复用 manifest 中已有 planner 结果，而不是重复 plan

---

## 11. 一句话结论

P3-mini 的正确切法不是“做一个很强的 planner”，而是：

> **给当前已经很稳的 stateful executor，补一层最小的、可解释的 pre-run video profiling。**

这会让系统从：

- 固定规则执行器

进化到：

- 带轻量前置策略选择的 stateful workflow

但仍然保持工程上可控。
