"""LLM-powered QA critic and narration rewriter.

This module provides two capabilities layered on top of the existing rule-based
``qa_gate.py``:

1. **LLM Critic** — evaluates narration accuracy, naturalness, and flow against
   the segment's visual context.  Returns structured feedback with specific
   issues and actionable suggestions.

2. **LLM Rewriter** — rewrites narration guided by critic feedback, preserving
   segment context and duration constraints.

Both use ``langchain-core`` for prompt management and structured output parsing
via Pydantic models so they can work with any LangChain-compatible LLM backend.
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any

from pydantic import BaseModel, Field

from .azure_auth import azure_openai_auth_headers
from .core import normalize_terms
from .qa_gate import QAConfig
from .state import Decision, SegmentState


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class CriticIssue(BaseModel):
    """A single issue identified by the LLM critic."""

    category: str = Field(
        description="Issue category: accuracy | naturalness | flow | length | repetition | terminology"
    )
    description: str = Field(description="What is wrong")
    suggestion: str = Field(description="How to fix it")
    severity: str = Field(description="low | medium | high")


class CriticResult(BaseModel):
    """Structured output from the LLM critic."""

    passed: bool = Field(description="Whether the narration is acceptable")
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
    issues: list[CriticIssue] = Field(default_factory=list, description="List of identified issues")
    overall_feedback: str = Field(default="", description="Brief overall assessment")


class RewriteResult(BaseModel):
    """Structured output from the LLM rewriter."""

    narration_zh: str = Field(description="The rewritten narration in Chinese")
    changes_made: list[str] = Field(
        default_factory=list,
        description="Brief descriptions of what was changed and why",
    )
    confidence: float = Field(description="Confidence the rewrite is better 0.0-1.0", ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = textwrap.dedent("""\
    你是一位专业的视频讲解质量评审专家。你的任务是评估一段中文视频旁白是否
    准确、自然、信息密度适当。

    评估维度：
    1. **准确性 (accuracy)** — 旁白是否准确描述了画面内容？是否存在虚构信息？
    2. **自然度 (naturalness)** — 旁白读起来是否像自然口播？是否过于书面化或像 OCR？
    3. **流畅度 (flow)** — 与前一段衔接是否自然？是否有生硬的过渡词？
    4. **长度 (length)** — 在给定时长内能否自然读完？
    5. **重复 (repetition)** — 是否与前一段重复？
    6. **术语 (terminology)** — 英文专名（如 Azure、OpenAI、Copilot Studio）是否保留英文？

    你必须返回有效 JSON，格式严格遵循以下 schema：
    {
      "passed": true/false,
      "confidence": 0.0-1.0,
      "issues": [
        {"category": "...", "description": "...", "suggestion": "...", "severity": "low|medium|high"}
      ],
      "overall_feedback": "简短总结"
    }
""")

REWRITER_SYSTEM_PROMPT = textwrap.dedent("""\
    你是一位专业的视频讲解文案编辑。根据评审反馈，改写旁白以解决指出的问题。

    规则：
    1. 保持中文口播风格，不要像 OCR 或图像描述。
    2. 保留英文专名不翻译（Azure、OpenAI、Copilot Studio 等）。
    3. 改写后的文案必须能在指定时长内自然读完。
    4. 只修复评审指出的问题，不要做不必要的改动。
    5. 如果画面信息很少，简短说明即可。

    你必须返回有效 JSON：
    {
      "narration_zh": "改写后的中文旁白",
      "changes_made": ["改了什么1", "改了什么2"],
      "confidence": 0.0-1.0
    }
""")


# ---------------------------------------------------------------------------
# LLM interaction (uses langchain-core / Azure OpenAI)
# ---------------------------------------------------------------------------

def _call_llm_json(
    *,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Call Azure OpenAI and parse JSON response.

    Uses the same Azure OpenAI credentials as the vision pipeline:
    ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``,
    ``AZURE_OPENAI_DEPLOYMENT``, ``AZURE_OPENAI_API_VERSION``.
    """
    import requests as _requests

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or None
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not all([endpoint, deployment]):
        raise RuntimeError(
            "LLM critic requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT "
            "environment variables (AZURE_OPENAI_API_KEY is optional when using MSI)"
        )

    url = (
        f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={api_version}"
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_completion_tokens": 1200,
        "response_format": {"type": "json_object"},
    }

    response = _requests.post(
        url,
        headers={**azure_openai_auth_headers(api_key), "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]

    import re
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            return json.loads(match.group(0))
        raise RuntimeError(f"LLM did not return valid JSON: {raw[:500]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_narration_llm(
    *,
    narration: str,
    segment: SegmentState,
    previous_narration: str,
    visible_points: list[str] | None = None,
    on_screen_text: list[str] | None = None,
    qa_config: QAConfig | None = None,
) -> CriticResult:
    """Evaluate narration quality using an LLM critic.

    Parameters
    ----------
    narration:
        The narration text to evaluate.
    segment:
        The segment being narrated (provides time boundaries and context).
    previous_narration:
        The accepted narration from the previous segment.
    visible_points:
        Visual elements detected on screen.
    on_screen_text:
        Text detected on screen.
    qa_config:
        Optional QA thresholds (uses defaults when *None*).

    Returns
    -------
    CriticResult
        Structured evaluation with pass/fail, issues, and feedback.
    """
    cfg = qa_config or QAConfig()
    max_chars = cfg.max_chars(segment.duration)
    chars_per_second = len(narration.strip()) / max(segment.duration, 0.1)

    user_prompt = textwrap.dedent(f"""\
        请评估以下视频旁白的质量。

        ## 片段信息
        - 时间窗：{segment.start:.3f}s - {segment.end:.3f}s（时长 {segment.duration:.1f}s）
        - 建议字数上限：{max_chars} 字
        - 当前字数：{len(narration.strip())} 字（{chars_per_second:.1f} 字/秒）

        ## 画面关键信息
        - 可见要素：{json.dumps(visible_points or segment.visible_points, ensure_ascii=False)}
        - 屏幕文字：{json.dumps(on_screen_text or segment.on_screen_text, ensure_ascii=False)}

        ## 上一段旁白
        {previous_narration or '（无）'}

        ## 待评估旁白
        {narration}
    """)

    try:
        raw = _call_llm_json(system_prompt=CRITIC_SYSTEM_PROMPT, user_prompt=user_prompt)
        return CriticResult(
            passed=bool(raw.get("passed", False)),
            confidence=float(raw.get("confidence", 0.5)),
            issues=[CriticIssue(**issue) for issue in raw.get("issues", [])],
            overall_feedback=str(raw.get("overall_feedback", "")),
        )
    except Exception as exc:
        # Graceful degradation — if LLM fails, let rule-based gate decide
        return CriticResult(
            passed=True,
            confidence=0.0,
            issues=[],
            overall_feedback=f"LLM critic unavailable: {exc}",
        )


def rewrite_narration_llm(
    *,
    narration: str,
    segment: SegmentState,
    critic_result: CriticResult,
    previous_narration: str,
    visible_points: list[str] | None = None,
    on_screen_text: list[str] | None = None,
    qa_config: QAConfig | None = None,
) -> RewriteResult:
    """Rewrite narration using an LLM guided by critic feedback.

    Parameters
    ----------
    narration:
        The current narration to rewrite.
    segment:
        The segment being narrated.
    critic_result:
        The critic's evaluation with specific issues to address.
    previous_narration:
        The accepted narration from the previous segment.
    visible_points:
        Visual elements detected on screen.
    on_screen_text:
        Text detected on screen.
    qa_config:
        Optional QA thresholds (uses defaults when *None*).

    Returns
    -------
    RewriteResult
        The rewritten narration with change descriptions.
    """
    cfg = qa_config or QAConfig()
    max_chars = cfg.max_chars(segment.duration)

    issues_text = "\n".join(
        f"  - [{issue.severity}] {issue.category}: {issue.description} → 建议: {issue.suggestion}"
        for issue in critic_result.issues
    ) or "  （无具体问题）"

    user_prompt = textwrap.dedent(f"""\
        请根据评审反馈改写以下视频旁白。

        ## 片段信息
        - 时间窗：{segment.start:.3f}s - {segment.end:.3f}s（时长 {segment.duration:.1f}s）
        - 字数上限：{max_chars} 字

        ## 画面关键信息
        - 可见要素：{json.dumps(visible_points or segment.visible_points, ensure_ascii=False)}
        - 屏幕文字：{json.dumps(on_screen_text or segment.on_screen_text, ensure_ascii=False)}

        ## 上一段旁白（避免重复）
        {previous_narration or '（无）'}

        ## 评审反馈
        {critic_result.overall_feedback}

        ## 具体问题
        {issues_text}

        ## 当前旁白（需改写）
        {narration}
    """)

    try:
        raw = _call_llm_json(system_prompt=REWRITER_SYSTEM_PROMPT, user_prompt=user_prompt)
        rewritten = normalize_terms(str(raw.get("narration_zh", narration)))
        return RewriteResult(
            narration_zh=rewritten,
            changes_made=[str(c) for c in raw.get("changes_made", [])],
            confidence=float(raw.get("confidence", 0.5)),
        )
    except Exception:
        # Graceful degradation — return the original narration
        return RewriteResult(
            narration_zh=narration,
            changes_made=["LLM rewriter unavailable, returning original"],
            confidence=0.0,
        )


# ---------------------------------------------------------------------------
# Combined two-layer QA evaluation
# ---------------------------------------------------------------------------

def evaluate_narration_two_layer(
    *,
    narration: str,
    segment: SegmentState,
    previous_narration: str,
    use_llm: bool = True,
    qa_config: QAConfig | None = None,
) -> tuple[bool, Decision, str, list[str], CriticResult | None]:
    """Two-layer QA evaluation: rule-based first, then LLM if rules pass.

    Returns
    -------
    tuple of (passed, decision, reason, feedback_list, critic_result_or_None)
    """
    from .qa_gate import evaluate_narration_quality

    cfg = qa_config or QAConfig()

    # Layer 1: rule-based (fast, free)
    rule_result = evaluate_narration_quality(
        narration=narration,
        previous_narration=previous_narration,
        duration_seconds=segment.duration,
        qa_config=cfg,
    )

    if not rule_result.passed:
        return (
            False,
            rule_result.decision,
            f"rule-based: {rule_result.reason}",
            rule_result.feedback,
            None,
        )

    if not use_llm:
        return (True, Decision.ACCEPT, rule_result.reason, [], None)

    # Layer 2: LLM critic (only if rules pass)
    critic = evaluate_narration_llm(
        narration=narration,
        segment=segment,
        previous_narration=previous_narration,
        qa_config=cfg,
    )

    if critic.passed or critic.confidence == 0.0:
        # LLM passed or was unavailable
        return (True, Decision.ACCEPT, "two-layer qa passed", [], critic)

    # LLM found issues
    high_severity = any(i.severity == "high" for i in critic.issues)
    feedback = [f"llm_critic: {i.category} - {i.description}" for i in critic.issues]
    if cfg.critic_lenient or not high_severity:
        decision = Decision.RETRY_NARRATION
    else:
        decision = Decision.NEEDS_HUMAN_REVIEW
    return (
        False,
        decision,
        f"llm critic: {critic.overall_feedback}",
        feedback,
        critic,
    )
