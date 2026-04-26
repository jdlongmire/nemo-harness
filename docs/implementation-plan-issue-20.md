# Implementation Plan: Issue #20 - Dynamic Guide Refresh

## Problem Statement

The system prompt is monolithic: every behavioral guide and every tool definition is sent to the model on every turn, regardless of what the user is actually talking about. For a 4K context model (Nemotron), this is expensive. A coding question doesn't need the git workflow guide, and a planning discussion doesn't need the file-editing tools. Sending everything unconditionally wastes context budget and dilutes the model's attention.

---

## Architecture: Tier 1 Intent Classification with Hysteresis

A deterministic keyword-based classifier detects conversation topic shifts and dynamically adjusts which behavioral guides and tools are sent to the model each turn. No LLM call is needed for classification -- compiled regex patterns run in sub-millisecond time.

```
User message(s)
      |
      v
[classify_recent()]  -- aggregate last 3 user messages
      |
      v
[ContextState]       -- hysteresis: 2 consecutive signals required to switch
      |
      v
[get_relevant_guides(intent)]  -- guide key subset
[get_relevant_tools(intent)]   -- tool name subset (or None = all)
      |
      v
[get_effective_system_prompt(guide_keys)]  -- compose prompt from fragments
[TOOL_REGISTRY.openai_tools(only=...)]     -- filtered tool definitions
      |
      v
[LLM request with trimmed context]
```

### Intent Categories

| Intent | Description | Fallback? |
|--------|-------------|-----------|
| `coding` | Code writing, debugging, file editing | No |
| `document` | Writing, formatting, markdown | No |
| `research` | Searching, analyzing, comparing | No |
| `planning` | Task management, architecture, design | No |
| `git` | Version control, commits, branches | No |
| `conversation` | General chat, unclear intent | Yes (default) |

When intent is `conversation` (fallback), all guides and all tools are sent -- identical to pre-feature behavior.

---

## Sub-task 1: Context Sensor Module

**Scope: Medium**

### What was built

New module `tools/context_sensor.py` containing:

- **`classify_message(text)`** - Classify a single message against compiled regex keyword patterns. Returns intent string and per-category scores.
- **`classify_recent(messages, window=3)`** - Classify the last N user messages with aggregate scoring across the window. Smooths out single-message noise.
- **`ContextState` dataclass** - Tracks current intent with hysteresis (threshold=2 consecutive matching signals required before switching). Prevents thrashing on ambiguous messages.
- **`INTENT_TOOLS` dict** - Maps each intent to the set of tool names relevant to that topic. Core tools (`read_file`, `list_dir`) appear in every intent.
- **`INTENT_GUIDES` dict** - Maps each intent to the list of guide block keys relevant to that topic.
- **`get_relevant_tools(intent)`** - Returns a tool name set for the given intent, or `None` (meaning "all tools") for the `conversation` fallback.
- **`get_relevant_guides(intent)`** - Returns a guide key list for the given intent.

### Design decisions

- **Compiled regex** over raw string matching: patterns are compiled once at import time, classification is a set of `re.search()` calls. Sub-millisecond per message.
- **Aggregate scoring** over single-message classification: "search" matches both research and coding, but looking at 3 messages disambiguates.
- **Hysteresis threshold of 2**: prevents a single off-topic message from triggering a full context switch. Intentionally conservative.

---

## Sub-task 2: Tool Filtering in Registry

**Scope: Small**

### What was changed

- **`tools/registry.py`** - `openai_tools()` method updated with an optional `only: set[str] | None` parameter:
  ```python
  def openai_tools(self, only: set[str] | None = None) -> list[dict]:
  ```
  When `only` is `None`, all registered tools are returned (current behavior). When a set is provided, only tools whose names appear in the set are included in the response.

---

## Sub-task 3: Guide Decomposition

**Scope: Medium**

### What was changed

The monolithic mode prompts in `server.py` were refactored into indexed fragments:

- **`GUIDE_BLOCKS` dict** - Each behavioral guide (memory behavior, tool usage, safety, etc.) is stored as a keyed fragment.
- **`_MODE_HEADERS` dict** - Per-mode header text (mode identity, role description).
- **`_MODE_FOOTERS` dict** - Per-mode footer text (closing instructions).
- **`_build_mode_prompt()` function** - Assembles a mode prompt from header + selected guide blocks + footer.
- **`get_effective_system_prompt(guide_keys)`** - Updated to accept an optional `guide_keys` list. When provided, only the specified guide blocks are included. When `None`, all guide blocks are included (current behavior).

### Design note

The decomposition preserves the exact text of the original mode prompts when all guides are included. No behavioral change when dynamic guides are disabled.

---

## Sub-task 4: Server Wiring

**Scope: Medium**

### What was changed

- **`server.py`** - Added import of `classify_recent`, `get_relevant_tools`, `get_relevant_guides`, and `ContextState` from `tools.context_sensor`.
- **`ENABLE_DYNAMIC_GUIDES`** - New env var (`NEMO_ENABLE_DYNAMIC_GUIDES`, default `true`). Controls whether the feature is active.
- **`_context_state`** - Module-level `ContextState` instance for tracking intent across turns.
- **`handle_chat()`** - Before building messages, classifies intent via `classify_recent()`, updates `_context_state`, then passes `guide_keys` and `tool_subset` downstream.
- **`_process_chat_job()`** - Accepts new `tool_subset` parameter, passes it to `TOOL_REGISTRY.openai_tools(only=tool_subset)`.
- **`handle_clear()`** - Resets `_context_state` when conversation is cleared.
- **Config endpoint** - Exposes `dynamic_guides` (enabled/disabled) and `current_intent` in the config API response.

### Control flow per request

1. `handle_chat()` receives user message
2. If `ENABLE_DYNAMIC_GUIDES` is true:
   - Call `classify_recent(conversation)` to get intent
   - Update `_context_state` (hysteresis check)
   - `guide_keys = get_relevant_guides(current_intent)`
   - `tool_subset = get_relevant_tools(current_intent)`
3. If disabled or intent is `conversation`:
   - `guide_keys = None` (all guides)
   - `tool_subset = None` (all tools)
4. Build system prompt via `get_effective_system_prompt(guide_keys)`
5. Pass `tool_subset` to `_process_chat_job()`

---

## Sub-task 5: State Management

**Scope: Small**

### What was built

- `ContextState` reset on `/clear` command (via `handle_clear()`)
- Current intent exposed in config API for debugging/observability
- `ENABLE_DYNAMIC_GUIDES` toggle allows disabling the feature entirely without code changes

---

## Summary: Files and Functions Modified

| File | Function/Location | Change | Scope |
|------|-------------------|--------|-------|
| `tools/context_sensor.py` | (new file) | Intent classifier, hysteresis state, tool/guide mappings | Medium |
| `tools/registry.py` | `openai_tools()` | Added `only: set[str] \| None` filter parameter | Small |
| `server.py` | imports | Added context_sensor imports | Small |
| `server.py` | `GUIDE_BLOCKS` | Decomposed monolithic mode prompts into indexed fragments | Medium |
| `server.py` | `_MODE_HEADERS`, `_MODE_FOOTERS` | Per-mode header/footer text | Small |
| `server.py` | `_build_mode_prompt()` | New function: assemble mode prompt from fragments | Small |
| `server.py` | `get_effective_system_prompt()` | Accepts optional `guide_keys` for dynamic composition | Medium |
| `server.py` | `handle_chat()` | Classifies intent, passes guide_keys and tool_subset | Medium |
| `server.py` | `_process_chat_job()` | Accepts `tool_subset`, passes to `openai_tools(only=...)` | Small |
| `server.py` | `handle_clear()` | Resets `_context_state` | Small |
| `server.py` | config endpoint | Exposes `dynamic_guides` and `current_intent` | Small |
| `server.py` | `ENABLE_DYNAMIC_GUIDES`, `_context_state` | New env var and module-level state | Small |

**Total scope: Medium** (12 touch points across 3 files, no new dependencies)

---

## Risks and Edge Cases

1. **Keyword ambiguity** - "search" matches both research and coding intents. Mitigated by aggregate scoring across last 3 messages; the dominant intent across the window wins.
2. **Hysteresis delay** - Takes 2 consecutive signals to switch intent. This is intentional to prevent thrashing, but means the first message in a new topic still uses the previous topic's context. Acceptable trade-off.
3. **Tool pruning hides needed tools** - A user in "research" mode might need to edit a file. Mitigated by always including core tools (`read_file`, `list_dir`) in every intent, and by the `conversation` fallback sending everything.
4. **Guide pruning loses relevant instructions** - If a guide block relevant to the user's actual need is excluded, the model may behave incorrectly. Mitigated by the conservative hysteresis and by the fallback behavior.
5. **Global state** - `_context_state` is per-process. Acceptable for single-user Nemo deployment.

---

## Configuration

| Env Var | Default | Effect |
|---------|---------|--------|
| `NEMO_ENABLE_DYNAMIC_GUIDES` | `true` | Toggle feature on/off |

---

## Migration

- **Zero-migration**: env var defaults to enabled, fallback intent (`conversation`) sends all guides and all tools (identical to pre-feature behavior).
- **To disable**: set `NEMO_ENABLE_DYNAMIC_GUIDES=false` in the environment or `.env` file.
- **No schema changes, no new dependencies, no data migration required.**

---

## Implementation Order

1. Sub-task 1 (context sensor module) - no dependencies
2. Sub-task 2 (tool filtering in registry) - no dependencies
3. Sub-task 3 (guide decomposition) - no dependencies
4. Sub-task 4 (server wiring) - depends on Sub-tasks 1, 2, 3
5. Sub-task 5 (state management) - depends on Sub-task 4

## Testing Checklist

- [x] `classify_message()` returns correct intents for representative messages
- [x] `classify_recent()` aggregate scoring resolves ambiguous single messages
- [x] `ContextState` hysteresis prevents thrashing (needs 2 consecutive signals)
- [x] `openai_tools(only=...)` returns filtered subset; `openai_tools()` returns all
- [x] `get_effective_system_prompt(guide_keys)` composes correct subset; `None` returns full prompt
- [x] `handle_chat()` classifies intent and passes correct subsets downstream
- [x] `/clear` resets context state
- [x] Config endpoint exposes current intent
- [x] `NEMO_ENABLE_DYNAMIC_GUIDES=false` disables feature, full context sent every turn
- [x] Live chat test confirms topic-appropriate guides and tools per turn
