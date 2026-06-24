---
name: prompt-strategy
description: "Analyse prompt, pick the best token strategy, then execute."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [token-optimization, strategy, routing, copilot, connectors, performance, efficiency]
    category: software-development
    related_skills: [simplify-code, plan, systematic-debugging]
    config:
      auxiliary.prompt_analysis.enabled: true
      auxiliary.prompt_analysis.model: ""
---

# Prompt Strategy Skill

Classify an incoming request before doing any work, choose the optimal
token-use strategy for it, apply that strategy to the agent runtime, then
execute the task.  Eliminates wasted tokens on conversational turns that don't
need tools, compresses history when the prior context is irrelevant, and
allocates extra iterations only when the task genuinely needs them.

## When to Use

Load this skill when the user asks any of:

- "optimize token use" / "reduce my token spend" / "speed up your responses"
- "analyze this request before acting" / "pick the best strategy"
- "use compact mode" / "don't waste tokens"
- Working on long-running Copilot ACP or relay-connector sessions where
  every API call is billed or rate-limited and wasted context hurts.

Also apply this skill automatically when the session has accumulated a long
history (> 20 turns) and you notice the same conversation topic shifting to
something unrelated — the prior context is likely irrelevant.

## Prerequisites

- `auxiliary.prompt_analysis.enabled: true` must be set in `config.yaml`
  (or set it for this session; see Quick Reference).
- A fast, cheap model is strongly recommended for the classifier:
  e.g. `gemini-flash`, `claude-haiku`, or any sub-10 B parameter local model.
  The full main model is overkill — the classifier emits only ~20 tokens.

## How to Run

Tell the user you are applying the prompt-strategy workflow, then follow the
procedure below.

## Quick Reference

```yaml
# config.yaml — enable pre-turn prompt classification
auxiliary:
  prompt_analysis:
    enabled: true
    model: ""          # leave blank to use main model, or set a fast model
    timeout: 10        # seconds; keep short — runs in the turn prologue
```

Check/set it in one command:

```bash
hermes config set auxiliary.prompt_analysis.enabled true
hermes config set auxiliary.prompt_analysis.model "google/gemini-3-flash-preview"
```

Verify the active config:

```bash
hermes config get auxiliary.prompt_analysis
```

## Procedure

### Step 1 — Classify the prompt

Before doing any work, call `analyze_prompt` (exposed via
`agent.prompt_analyzer`) to classify the incoming request.  When
`auxiliary.prompt_analysis.enabled` is `true` this runs automatically in the
turn prologue; when it is `false` you can classify manually:

```python
from agent.prompt_analyzer import analyze_prompt, NULL_ANALYSIS

analysis = analyze_prompt(
    user_message,          # the raw request text
    conversation_history,  # current working message list
    model=agent.model,
)
```

The result is a `PromptAnalysis` with these fields:

| Field | Values | Meaning |
|---|---|---|
| `task_type` | `conversation` `coding` `research` `file_ops` `reasoning` `creative` `unknown` | Broad category of work |
| `needs_tools` | `true` / `false` | Will this turn likely call tools? |
| `use_compression` | `true` / `false` | Is prior history irrelevant to this request? |
| `strategy_hints` | list of tags (see below) | Actionable optimizations |
| `context_budget_hint` | `small` `medium` `large` | Expected token footprint |

**Strategy hint tags:**

| Tag | Applied effect |
|---|---|
| `no_tools` | Tool schema omitted from the Copilot ACP prompt; no tool-use enforcement injected |
| `compact_history` | Only the system message + last 4 turns included in Copilot ACP transcript |
| `needs_context` | Full history preserved; compression skipped this turn |
| `fast_response` | No extra-iteration budget; answer directly |
| `heavy_compute` | Iteration budget raised by 33 % to accommodate many tool calls |

### Step 2 — Choose and apply the strategy

Map the classification to actions:

| Scenario | Actions |
|---|---|
| `task_type == "conversation"` and `needs_tools == false` | Set `no_tools` + `fast_response` hints; skip tool-schema injection; answer directly |
| `task_type == "coding"` or `task_type == "file_ops"` | Standard tool path; set `heavy_compute` if the diff/codebase is large |
| `task_type == "research"` | Enable `web_search` toolset; set `context_budget_hint = large` |
| `use_compression == true` | Trigger preflight compression now — don't wait for the threshold |
| `needs_context == true` in hints | Disable preflight compression this turn; keep full history |
| Copilot ACP backend (`provider == "copilot-acp"`) | Always propagate `compact_history` and `no_tools` flags to `client.strategy` |
| Relay/connector session | Apply the same flags; additionally gate heavy toolsets behind `CapabilityDescriptor.pii_safe` check |

When running with the built-in `prompt_analyzer` wired into `turn_context.py`,
these flags are already applied automatically.  This procedure is for manual
override or for sessions where the auto-classifier is disabled.

### Step 3 — Set the flags on the active client (Copilot ACP)

When the active backend is Copilot ACP, propagate the strategy after
classification:

```python
from agent.copilot_acp_client import CopilotACPClient

client = getattr(agent, "_client", None)
if isinstance(client, CopilotACPClient):
    client.strategy = {
        "compact_history": "compact_history" in analysis.strategy_hints,
        "no_tools": "no_tools" in analysis.strategy_hints,
    }
```

`_format_messages_as_prompt` reads `client.strategy` on every call, so the
strategy takes effect on the very next Copilot ACP request.

### Step 4 — Execute the task

Proceed with the task using the strategy you selected.  Announce the strategy
briefly so the user understands what you did:

> "Classified as **conversation / no tools needed**.  Using compact history and
> skipping tool schemas — this saves ~40 % of the prompt tokens for this turn."

or:

> "Classified as **coding / heavy compute**.  Full history retained, iteration
> budget raised to 120.  Proceeding with implementation."

### Step 5 — Report token impact (optional)

After the turn completes, if the user asked for optimization reporting, use
`/insights` or the `InsightsEngine` to compare the turn's token use against
the session baseline:

```bash
/insights --last 1
```

Key metrics to compare:
- `prompt_tokens` this turn vs. session average
- `cache_read_tokens` (a high ratio means the system prompt cache stayed warm)
- `completion_tokens` (should be proportional to the response length)

## Pitfalls

- **Never disable `needs_context` turns prematurely.**  If `needs_context` is
  in `strategy_hints`, the model determined the prior context is load-bearing.
  Compressing it will produce an incorrect or incomplete answer.
- **Do not set `no_tools` when `task_type` is `file_ops` or `coding`.**  These
  tasks reliably need `terminal`, `read_file`, or `patch`; skipping the tool
  schema will cause the model to attempt inline text edits instead.
- **Copilot ACP `compact_history` drops tool result messages outside the recent
  window.**  If the task is a continuation that depends on an earlier tool
  result (e.g. "now apply that same patch to the other file"), keep
  `compact_history = False` or widen the recent-turns window.
- **The classifier itself uses tokens.**  On a very cheap model (gemini-flash)
  the classification call costs ~150 input + ~25 output tokens.  At > 100
  turns per session the savings outweigh this overhead by ~10×; at < 5 turns
  the overhead dominates.  Consider enabling only for long-running sessions.
- **Cache invalidation:** changing `client.strategy` between turns is safe —
  the strategy dict is read per-request.  Changing toolsets or the system
  prompt mid-conversation still breaks the prompt-cache prefix.  The
  prompt-strategy optimization works precisely because it does NOT touch the
  system prompt.

## Verification

After applying the strategy, confirm it worked:

```python
# 1. Check the analysis was produced
assert agent._prompt_analysis_compact_history in (True, False)
assert agent._prompt_analysis_no_tools in (True, False)

# 2. For Copilot ACP — confirm strategy dict is set
from agent.copilot_acp_client import CopilotACPClient
client = getattr(agent, "_client", None)
if isinstance(client, CopilotACPClient):
    assert isinstance(client.strategy, dict)

# 3. Check the turn context carried the analysis
# (available as _ctx.prompt_analysis in conversation_loop.py)
```

For a full integration check, run:

```bash
scripts/run_tests.sh tests/agent/test_prompt_analyzer.py -q
```
