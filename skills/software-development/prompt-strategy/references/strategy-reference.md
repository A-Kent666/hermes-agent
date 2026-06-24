# Prompt Strategy — Reference

## PromptAnalysis Contract

`agent/prompt_analyzer.PromptAnalysis` is a frozen dataclass returned by
`analyze_prompt()`.  All fields have safe defaults so callers never need to
guard against `None`.

```
PromptAnalysis(
    task_type          : str   = "unknown"   # see Task Types below
    needs_tools        : bool  = True        # conservative default
    use_compression    : bool  = False       # don't compress unless asked
    strategy_hints     : list  = []          # no hints = standard path
    context_budget_hint: str   = "medium"    # see Budget Hints below
)
```

`NULL_ANALYSIS` is the module-level sentinel returned on any failure or when
analysis is disabled.  It equals the zero-configuration defaults above and is
safe to pass to all consumers.

---

## Task Types

| Value | Typical requests |
|---|---|
| `conversation` | "what is X?", "explain Y", greetings, follow-up questions with no file/code involvement |
| `coding` | "write a function", "fix this bug", "add a test", "refactor this" |
| `research` | "find papers on X", "summarize what Y does", "compare A and B" |
| `file_ops` | "rename all files", "copy these to", "show me the diff", "compress this directory" |
| `reasoning` | "solve this math problem", "find the flaw in this argument", "reason through X" |
| `creative` | "write a poem", "generate an image prompt", "come up with names for" |
| `unknown` | Anything the classifier could not confidently categorize |

---

## Strategy Hints

Strategy hints are the classifier's actionable output.  The loop applies them
in `agent/conversation_loop.py` immediately after unpacking `TurnContext`.

### `no_tools`
**Applied when:** `task_type == "conversation"` and the request has no
plausible tool-use path.

**Effect:**
- `agent._prompt_analysis_no_tools = True`
- Copilot ACP: `_format_messages_as_prompt` omits the tool-schema block and
  tool-choice hint entirely — saving the full tool-schema token cost per turn.
- Main OpenAI path: flag is visible on `agent` for future optimizations but
  does not yet suppress tools (would require mid-conversation toolset swap
  which breaks prompt caching).

**Do NOT set when:** `task_type` is `coding`, `file_ops`, or `reasoning` —
these tasks reliably require at least one tool call.

### `compact_history`
**Applied when:** the current request is a fresh topic that does not depend on
earlier turns (e.g. first turn of a new topic in a long session, or a
self-contained question in a multi-topic chat).

**Effect:**
- `agent._prompt_analysis_compact_history = True`
- Copilot ACP: `_format_messages_as_prompt` keeps only the system message and
  the most recent 4 user/assistant turns.  Tool result messages outside that
  window are dropped.

**Do NOT set when:** `needs_context` is also in hints, or `task_type` is
`coding` / `file_ops` referencing earlier turns.

### `needs_context`
**Applied when:** the request explicitly references prior turns or depends on
the accumulated conversation state ("as we discussed", "now do the same for
the next file", continuation tasks).

**Effect:**
- Suppresses `compact_history` — full history is preserved.
- Signals the compression system to skip preflight compression this turn
  (even if the token estimate is above the threshold).

### `fast_response`
**Applied when:** the task is a simple conversational reply where latency
matters more than thoroughness.

**Effect:**
- No extra iteration budget is granted.
- (Future) May signal the agent to prefer streaming and skip post-response
  enrichment steps.

### `heavy_compute`
**Applied when:** the task is expected to require many sequential tool calls
(large codebases, multi-file refactors, long research chains).

**Effect:**
- `agent.iteration_budget` is rebuilt with `max_iterations + max_iterations // 3`
  (a 33 % increase over the configured ceiling).
- The increase is capped per-turn and resets on the next turn.

---

## Context Budget Hints

| Value | Token range | Implication |
|---|---|---|
| `small` | < 8 k tokens | Simple request; current context is unlikely to need compression |
| `medium` | 8 k–32 k tokens | Standard path; compression threshold applies normally |
| `large` | > 32 k tokens | Heavy context; consider triggering preflight compression proactively |

---

## Configuration Reference

All settings live under `auxiliary.prompt_analysis` in `config.yaml`:

```yaml
auxiliary:
  prompt_analysis:
    enabled: false     # true to activate pre-turn classification
    provider: auto     # provider for the classifier call
    model: ""          # empty = use main model; set a fast/cheap model here
    base_url: ""       # override endpoint (for local classifiers)
    api_key: ""        # override key
    timeout: 10        # seconds — keep short; runs in turn prologue
    extra_body: {}     # provider-specific request fields
```

**Recommended model for classifier:** any model that reliably follows JSON
instructions.  Suggestions by cost tier:

| Tier | Model |
|---|---|
| Free / very cheap | `google/gemini-3-flash-preview` via OpenRouter |
| Mid-tier | `anthropic/claude-haiku-4-5` |
| Local (ollama) | `qwen2.5:3b` or `phi3:mini` |
| Main model (default) | Leave `model: ""` — uses whatever the main session uses |

---

## Copilot ACP Integration

`CopilotACPClient.strategy` is a plain `dict` read by `_create_chat_completion`
on every call.  Write it from outside without subclassing:

```python
from agent.copilot_acp_client import CopilotACPClient

client = getattr(agent, "_client", None)
if isinstance(client, CopilotACPClient):
    client.strategy = {
        "compact_history": True,   # trim history to last 4 turns
        "no_tools": False,         # keep tool schemas
    }
```

Both keys default to `False` when absent.  The dict is read-only from the
client's perspective; the loop resets it at the start of each turn via the
`conversation_loop.py` hint-application block.

---

## Relay Connector Integration

The relay `CapabilityDescriptor` carries a `pii_safe` flag.  When
`pii_safe == True` the connector guarantees that no personally identifiable
data flows through the channel.  Future work: expose `pii_safe` as an
additional strategy input so the classifier can suggest lighter tool schemas
for privacy-constrained channels.

Until then, the `compact_history` hint is the primary token optimization for
relay sessions — it prevents the agent from retransmitting a long conversation
thread on every assistant response.
