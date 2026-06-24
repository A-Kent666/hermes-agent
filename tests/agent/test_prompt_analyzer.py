"""Tests for agent.prompt_analyzer — pre-turn prompt classification."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.prompt_analyzer import (
    NULL_ANALYSIS,
    PromptAnalysis,
    _build_analysis_messages,
    _clear_analysis_enabled_cache,
    _is_analysis_enabled,
    _parse_analysis_response,
    analyze_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _valid_json(
    task_type: str = "conversation",
    needs_tools: bool = False,
    use_compression: bool = False,
    strategy_hints: list | None = None,
    context_budget_hint: str = "small",
) -> str:
    return json.dumps(
        {
            "task_type": task_type,
            "needs_tools": needs_tools,
            "use_compression": use_compression,
            "strategy_hints": strategy_hints or [],
            "context_budget_hint": context_budget_hint,
        }
    )


# ---------------------------------------------------------------------------
# _is_analysis_enabled
# ---------------------------------------------------------------------------

class TestIsAnalysisEnabled:
    def setup_method(self):
        _clear_analysis_enabled_cache()

    def teardown_method(self):
        _clear_analysis_enabled_cache()

    def test_disabled_by_default(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert _is_analysis_enabled() is False

    def test_enabled_when_config_says_so(self):
        cfg = {"auxiliary": {"prompt_analysis": {"enabled": True}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _is_analysis_enabled() is True

    def test_returns_false_on_config_error(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("bad")):
            assert _is_analysis_enabled() is False

    def test_result_is_cached(self):
        cfg = {"auxiliary": {"prompt_analysis": {"enabled": True}}}
        with patch("hermes_cli.config.load_config", return_value=cfg) as mock_load:
            _is_analysis_enabled()
            _is_analysis_enabled()
        # Second call must hit the cache, not re-invoke load_config.
        assert mock_load.call_count == 1


# ---------------------------------------------------------------------------
# _build_analysis_messages
# ---------------------------------------------------------------------------

class TestBuildAnalysisMessages:
    def test_returns_two_messages(self):
        msgs = _build_analysis_messages("hello", None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_is_truncated_at_500(self):
        long = "x" * 600
        msgs = _build_analysis_messages(long, None)
        assert len(msgs[1]["content"]) < 600

    def test_history_snippet_appears(self):
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        msgs = _build_analysis_messages("follow-up", history)
        user_content = msgs[1]["content"]
        assert "first question" in user_content or "first answer" in user_content

    def test_history_capped_at_three_turns(self):
        history = [
            {"role": "user", "content": f"msg {i}"}
            for i in range(10)
        ]
        msgs = _build_analysis_messages("now", history)
        user_content = msgs[1]["content"]
        # Only last 3 user/assistant turns should appear
        # Earlier messages ("msg 0" through "msg 6") should be absent.
        assert "msg 0" not in user_content


# ---------------------------------------------------------------------------
# _parse_analysis_response
# ---------------------------------------------------------------------------

class TestParseAnalysisResponse:
    def test_parses_valid_json(self):
        raw = _valid_json("coding", True, False, ["heavy_compute"], "large")
        result = _parse_analysis_response(raw)
        assert result.task_type == "coding"
        assert result.needs_tools is True
        assert result.use_compression is False
        assert result.strategy_hints == ["heavy_compute"]
        assert result.context_budget_hint == "large"

    def test_returns_null_on_invalid_json(self):
        assert _parse_analysis_response("not json") is NULL_ANALYSIS

    def test_returns_null_on_non_dict(self):
        assert _parse_analysis_response("[1, 2, 3]") is NULL_ANALYSIS

    def test_strips_markdown_fences(self):
        raw = "```json\n" + _valid_json() + "\n```"
        result = _parse_analysis_response(raw)
        assert result.task_type == "conversation"

    def test_unknown_task_type_becomes_unknown(self):
        raw = _valid_json("alien_task_type")
        result = _parse_analysis_response(raw)
        assert result.task_type == "unknown"

    def test_unknown_budget_hint_becomes_medium(self):
        raw = json.dumps(
            {
                "task_type": "conversation",
                "needs_tools": False,
                "use_compression": False,
                "strategy_hints": [],
                "context_budget_hint": "ridiculous",
            }
        )
        result = _parse_analysis_response(raw)
        assert result.context_budget_hint == "medium"

    def test_unknown_strategy_hints_are_filtered(self):
        raw = json.dumps(
            {
                "task_type": "coding",
                "needs_tools": True,
                "use_compression": False,
                "strategy_hints": ["no_tools", "unknown_hint_xyz"],
                "context_budget_hint": "small",
            }
        )
        result = _parse_analysis_response(raw)
        assert result.strategy_hints == ["no_tools"]

    def test_missing_optional_fields_use_defaults(self):
        raw = json.dumps({"task_type": "research"})
        result = _parse_analysis_response(raw)
        assert result.needs_tools is True
        assert result.use_compression is False
        assert result.strategy_hints == []
        assert result.context_budget_hint == "medium"

    def test_all_valid_task_types_are_accepted(self):
        for tt in ("conversation", "coding", "research", "file_ops", "reasoning", "creative", "unknown"):
            raw = _valid_json(tt)
            assert _parse_analysis_response(raw).task_type == tt

    def test_all_valid_strategy_hints_are_kept(self):
        hints = ["no_tools", "compact_history", "needs_context", "fast_response", "heavy_compute"]
        raw = json.dumps(
            {
                "task_type": "conversation",
                "needs_tools": False,
                "use_compression": False,
                "strategy_hints": hints,
                "context_budget_hint": "small",
            }
        )
        result = _parse_analysis_response(raw)
        assert set(result.strategy_hints) == set(hints)


# ---------------------------------------------------------------------------
# analyze_prompt (integration with disabled/enabled gate)
# ---------------------------------------------------------------------------

class TestAnalyzePrompt:
    def setup_method(self):
        _clear_analysis_enabled_cache()

    def teardown_method(self):
        _clear_analysis_enabled_cache()

    def test_returns_null_when_disabled(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            result = analyze_prompt("hello world")
        assert result is NULL_ANALYSIS

    def test_returns_null_for_empty_message(self):
        cfg = {"auxiliary": {"prompt_analysis": {"enabled": True}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            result = analyze_prompt("   ")
        assert result is NULL_ANALYSIS

    def test_returns_parsed_result_when_enabled(self):
        cfg = {"auxiliary": {"prompt_analysis": {"enabled": True}}}
        payload = _valid_json("coding", True, False, ["heavy_compute"], "large")
        mock_resp = _make_llm_response(payload)

        with (
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("agent.auxiliary_client.call_llm", return_value=mock_resp),
        ):
            result = analyze_prompt("write a fibonacci function")

        assert result.task_type == "coding"
        assert result.needs_tools is True
        assert "heavy_compute" in result.strategy_hints

    def test_returns_null_on_llm_failure(self):
        cfg = {"auxiliary": {"prompt_analysis": {"enabled": True}}}
        with (
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("network error")),
        ):
            result = analyze_prompt("what time is it?")
        assert result is NULL_ANALYSIS

    def test_passes_main_runtime_to_call_llm(self):
        cfg = {"auxiliary": {"prompt_analysis": {"enabled": True}}}
        mock_resp = _make_llm_response(_valid_json())
        with (
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("agent.auxiliary_client.call_llm", return_value=mock_resp) as mock_llm,
        ):
            analyze_prompt(
                "hello",
                model="gpt-4o",
                main_runtime={"provider": "openrouter"},
            )
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs.get("task") == "prompt_analysis"
        assert call_kwargs.get("max_tokens") == 256
        assert call_kwargs.get("temperature") == 0.0

    def test_does_not_raise_on_none_history(self):
        cfg = {"auxiliary": {"prompt_analysis": {"enabled": True}}}
        mock_resp = _make_llm_response(_valid_json())
        with (
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("agent.auxiliary_client.call_llm", return_value=mock_resp),
        ):
            result = analyze_prompt("hello", None)
        assert result is not NULL_ANALYSIS
