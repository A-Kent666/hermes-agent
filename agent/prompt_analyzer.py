"""Pre-turn prompt analysis — classify each user prompt before the agent loop starts.

Runs a single, cheap auxiliary LLM call that returns a structured
:class:`PromptAnalysis` describing:

* **task_type** — the broad category of work requested
  (``"conversation"``, ``"coding"``, ``"research"``, ``"file_ops"``,
  ``"reasoning"``, ``"creative"``, ``"unknown"``).
* **needs_tools** — whether the request is likely to require tool calls at all.
  A ``False`` classification lets the agent skip heavy toolset setup for
  conversational turns where the overhead is pure waste.
* **use_compression** — whether the history should be trimmed/compressed before
  sending (``True`` when the prompt is short but references a long prior context
  that the current task doesn't need; ``False`` when the task needs full history).
* **strategy_hints** — an ordered list of free-form strategy tags the loop can
  act on.  Defined tags understood by :mod:`agent.conversation_loop`:

    ``"no_tools"``         — strongly signal tool-calls are not needed this turn.
    ``"compact_history"``  — prefer a trimmed transcript (Copilot ACP, budget mode).
    ``"needs_context"``    — request has a high dependency on prior conversation state.
    ``"fast_response"``    — user expects a low-latency conversational reply.
    ``"heavy_compute"``    — task will use many tool iterations; raise iteration budget.

* **context_budget_hint** — rough estimate of how many context tokens the task
  needs: ``"small"`` (< 8k), ``"medium"`` (8k–32k), or ``"large"`` (> 32k).

The analysis is **best-effort**:

- When the auxiliary call fails for any reason the caller receives a safe
  :data:`NULL_ANALYSIS` (tools assumed, no hints, medium budget) so the main
  path is unaffected.
- The analysis is always skipped when ``enabled`` is ``False`` in
  ``auxiliary.prompt_analysis`` config (the default), so existing deployments
  are unchanged until opt-in.
- The analysis uses :func:`agent.auxiliary_client.call_llm` with
  ``task="prompt_analysis"`` so it inherits the full provider/model/timeout
  config chain.

Usage::

    from agent.prompt_analyzer import analyze_prompt, NULL_ANALYSIS

    analysis = analyze_prompt(user_message, conversation_history, model=agent.model)
    if "fast_response" in analysis.strategy_hints:
        ...
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import so the module loads fast and the symbol is patchable in tests.
# ``from agent.auxiliary_client import call_llm`` at module top would trigger
# a heavyweight SDK import chain on every agent startup.  The pattern here
# mirrors ``agent.auxiliary_client``'s own lazy OpenAI SDK loader.
def _load_call_llm():
    from agent.auxiliary_client import call_llm as _fn  # noqa: PLC0415
    return _fn


def call_llm(*args, **kwargs):  # type: ignore[misc]  # noqa: F811
    """Thin forwarder — replaced by ``patch('agent.prompt_analyzer.call_llm')`` in tests."""
    return _load_call_llm()(*args, **kwargs)

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

_VALID_TASK_TYPES = frozenset(
    {"conversation", "coding", "research", "file_ops", "reasoning", "creative", "unknown"}
)
_VALID_STRATEGY_HINTS = frozenset(
    {"no_tools", "compact_history", "needs_context", "fast_response", "heavy_compute"}
)
_VALID_BUDGET_HINTS = frozenset({"small", "medium", "large"})


@dataclass(frozen=True)
class PromptAnalysis:
    """Structured result of a pre-turn prompt classification."""

    # Broad category of the requested work.
    task_type: str = "unknown"
    # Whether the request is expected to require tool calls.
    needs_tools: bool = True
    # Whether the history should be trimmed before the main call.
    use_compression: bool = False
    # Ordered list of actionable strategy tags (see module docstring).
    strategy_hints: List[str] = field(default_factory=list)
    # Rough context-token budget the task is expected to consume.
    context_budget_hint: str = "medium"


# Returned on any analysis failure — safe defaults that preserve existing behaviour.
NULL_ANALYSIS = PromptAnalysis(
    task_type="unknown",
    needs_tools=True,
    use_compression=False,
    strategy_hints=[],
    context_budget_hint="medium",
)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_ANALYSIS_ENABLED_CACHE: Optional[bool] = None
_ANALYSIS_ENABLED_CACHE_LOCK = None  # set lazily to avoid import-time threading cost


def _is_analysis_enabled() -> bool:
    """Return True when ``auxiliary.prompt_analysis.enabled`` is truthy in config.

    Result is cached for the lifetime of the process (config is static after
    startup).  Returns ``False`` on any error so analysis silently stays off
    by default — users must explicitly opt in.
    """
    global _ANALYSIS_ENABLED_CACHE, _ANALYSIS_ENABLED_CACHE_LOCK
    import threading

    if _ANALYSIS_ENABLED_CACHE_LOCK is None:
        _ANALYSIS_ENABLED_CACHE_LOCK = threading.Lock()

    if _ANALYSIS_ENABLED_CACHE is not None:
        return _ANALYSIS_ENABLED_CACHE

    with _ANALYSIS_ENABLED_CACHE_LOCK:
        if _ANALYSIS_ENABLED_CACHE is not None:
            return _ANALYSIS_ENABLED_CACHE
        try:
            from hermes_cli.config import load_config

            config = load_config() or {}
            aux = config.get("auxiliary") or {}
            pa = aux.get("prompt_analysis") or {}
            enabled = bool(pa.get("enabled", False))
        except Exception:
            enabled = False
        _ANALYSIS_ENABLED_CACHE = enabled
        return enabled


def _clear_analysis_enabled_cache() -> None:
    """Reset the enabled-state cache (test helper)."""
    global _ANALYSIS_ENABLED_CACHE
    _ANALYSIS_ENABLED_CACHE = None


# ---------------------------------------------------------------------------
# Prompt / system message construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a task-routing assistant. "
    "Analyze the user message (and brief history context, if provided) and "
    "return a JSON object with EXACTLY these keys:\n"
    "\n"
    '  "task_type"          : one of "conversation", "coding", "research", '
    '"file_ops", "reasoning", "creative", "unknown"\n'
    '  "needs_tools"        : true | false — will this request likely require '
    "tool calls (terminal, file read/write, web search, etc.)?\n"
    '  "use_compression"    : true | false — is the prior history irrelevant '
    "to this request so it can safely be trimmed?\n"
    '  "strategy_hints"     : array (may be empty) of zero or more of: '
    '"no_tools", "compact_history", "needs_context", "fast_response", "heavy_compute"\n'
    '  "context_budget_hint": one of "small" (< 8 k tokens), '
    '"medium" (8 k–32 k), "large" (> 32 k)\n'
    "\n"
    "Return ONLY the JSON object, no explanation, no markdown fences."
)


def _build_analysis_messages(
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Build the messages list for the auxiliary classification call."""
    # Include a brief history snippet (last 3 turns, truncated) so the
    # classifier can see whether the request is conversational follow-up
    # vs. a fresh task — without sending the full expensive history.
    history_snippet = ""
    if conversation_history:
        recent = [
            m for m in conversation_history
            if m.get("role") in ("user", "assistant")
        ][-3:]
        parts = []
        for m in recent:
            role = m.get("role", "")
            raw = m.get("content") or ""
            text = raw if isinstance(raw, str) else json.dumps(raw)
            snippet = (text[:200] + "…") if len(text) > 200 else text
            parts.append(f"{role.title()}: {snippet}")
        if parts:
            history_snippet = "\n\n[Recent history]\n" + "\n".join(parts) + "\n"

    prompt_text = (
        "[User message]\n"
        + (user_message[:500] if len(user_message) > 500 else user_message)
        + history_snippet
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def _parse_analysis_response(raw: str) -> PromptAnalysis:
    """Parse the raw JSON string from the classifier into a PromptAnalysis.

    Applies field-level validation with safe fallbacks so a partially correct
    response still yields a useful result.
    """
    try:
        # Strip markdown fences if the model added them despite instructions.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.debug("prompt_analyzer: could not parse JSON response: %r", raw[:200])
        return NULL_ANALYSIS

    if not isinstance(data, dict):
        return NULL_ANALYSIS

    task_type = str(data.get("task_type") or "unknown").strip().lower()
    if task_type not in _VALID_TASK_TYPES:
        task_type = "unknown"

    needs_tools = bool(data.get("needs_tools", True))
    use_compression = bool(data.get("use_compression", False))

    raw_hints = data.get("strategy_hints") or []
    if not isinstance(raw_hints, list):
        raw_hints = []
    strategy_hints = [
        h for h in (str(x).strip().lower() for x in raw_hints if x)
        if h in _VALID_STRATEGY_HINTS
    ]

    context_budget_hint = str(data.get("context_budget_hint") or "medium").strip().lower()
    if context_budget_hint not in _VALID_BUDGET_HINTS:
        context_budget_hint = "medium"

    return PromptAnalysis(
        task_type=task_type,
        needs_tools=needs_tools,
        use_compression=use_compression,
        strategy_hints=strategy_hints,
        context_budget_hint=context_budget_hint,
    )


def analyze_prompt(
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    *,
    model: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> PromptAnalysis:
    """Classify a user prompt and return a :class:`PromptAnalysis`.

    Returns :data:`NULL_ANALYSIS` when:

    * ``auxiliary.prompt_analysis.enabled`` is falsy (default — opt-in only).
    * The auxiliary LLM call fails for any reason.
    * The response cannot be parsed as valid JSON.

    Args:
        user_message: The raw user message for this turn.
        conversation_history: Working message list (may be ``None`` or empty
            for first turns).  Only the last three user/assistant pairs are
            included in the classifier prompt.
        model: The main agent model name (passed as context hint to
            ``call_llm`` via ``main_runtime``; does not override the
            auxiliary task's configured model).
        main_runtime: Optional dict forwarded to ``call_llm`` so the
            classifier can prefer the same provider as the main agent when
            ``auxiliary.prompt_analysis.provider`` is ``"auto"``.
    """
    if not _is_analysis_enabled():
        return NULL_ANALYSIS

    if not isinstance(user_message, str) or not user_message.strip():
        return NULL_ANALYSIS

    messages = _build_analysis_messages(user_message, conversation_history)

    _runtime: Dict[str, Any] = dict(main_runtime or {})
    if model and "model" not in _runtime:
        _runtime["model"] = model

    try:
        response = call_llm(
            task="prompt_analysis",
            messages=messages,
            max_tokens=256,
            temperature=0.0,
            main_runtime=_runtime or None,
        )
        raw_content = (response.choices[0].message.content or "").strip()
        analysis = _parse_analysis_response(raw_content)
        logger.debug(
            "prompt_analyzer: task_type=%s needs_tools=%s hints=%s budget=%s",
            analysis.task_type,
            analysis.needs_tools,
            analysis.strategy_hints,
            analysis.context_budget_hint,
        )
        return analysis
    except Exception as exc:
        logger.debug("prompt_analyzer: auxiliary call failed (%s) — using null analysis", exc)
        return NULL_ANALYSIS
