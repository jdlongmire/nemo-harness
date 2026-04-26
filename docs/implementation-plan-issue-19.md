# Implementation Plan: Issue #19 - Sliding Window + Conversation Summarization

## Problem Statement

Conversation history is a flat list capped at 50 turns via `pop(0)` (`server.py:895-896`). No token counting, no summarization, no observation masking. Old turns silently drop when the list exceeds `CFG['max_history']`. The model (Nemotron, 4K context) loses coherence because:

1. **Turn-based, not token-based** - a tool result can be 3K tokens, "yes" is 2 tokens, but both count as 1 turn
2. **No compression** - old context is discarded, not summarized
3. **Tool output bloat** - full tool results stay verbatim forever, dominating the context window

---

## Architecture: Three-Zone Model

```
[System Prompt + Tools + Memory]  ~1000 tokens (pinned, never evicted)
[Running Summary]                 ~200 tokens  (compressed older context)
[Verbatim Recent Turns]           ~2500 tokens (recent history, untouched)
[Response Budget]                 ~300 tokens  (model output)
                                  ─────────────
                                  ~4000 tokens total (4K context)
```

### Zone Rules

| Zone | Content | Eviction | Budget |
|------|---------|----------|--------|
| Pinned | System prompt, tools, memory block | Never | ~1000 tokens |
| Summary | Compressed older turns | Regenerated on eviction | ~200 tokens |
| Verbatim | Recent N turns | Oldest moves to summary zone | ~2500 tokens |
| Response | Model output space | N/A | ~300 tokens |

---

## Sub-task 1: Token Estimation Utility

**Scope: Small**

### What changes

Add token estimation functions. Nemotron doesn't expose a tokenizer, so use chars/4 heuristic (same as `memory_store.py`).

### Files to modify

- **`server.py`** - Add near top (after imports, ~line 30):
  ```python
  def estimate_tokens(text: str) -> int:
      """Estimate token count using chars/4 heuristic."""
      return len(text) // 4 if text else 0

  def message_tokens(msg: dict) -> int:
      """Estimate tokens for a conversation message."""
      content = msg.get('content', '')
      if isinstance(content, str):
          return estimate_tokens(content) + 4  # role overhead
      # Handle structured content (tool calls, etc.)
      return estimate_tokens(str(content)) + 4
  ```

---

## Sub-task 2: Observation Masking

**Scope: Medium**

### What changes

For tool result messages older than the most recent K tool interactions, replace verbose output with a compact placeholder. This preserves conversation structure without the token cost.

**Key insight from JetBrains research:** Observation masking outperformed summarization in 4/5 settings with 52% cost savings. Full summarization caused agents to run ~15% longer because compressed context obscured stopping signals.

### Files to modify

- **`server.py`** - Add function after `message_tokens()`:
  ```python
  VERBATIM_TOOL_RESULTS = 3  # Keep last N tool results unmasked

  def mask_observations(messages: list[dict], keep_recent: int = VERBATIM_TOOL_RESULTS) -> list[dict]:
      """Replace old tool result content with compact placeholders."""
      # Find indices of tool result messages (role='user' containing tool output markers)
      tool_indices = [
          i for i, m in enumerate(messages)
          if m.get('role') == 'user' and '[Tool:' in m.get('content', '')
      ]
      # Mask all but the most recent `keep_recent`
      to_mask = tool_indices[:-keep_recent] if len(tool_indices) > keep_recent else []
      result = []
      for i, msg in enumerate(messages):
          if i in to_mask:
              # Extract tool name if possible, replace content
              result.append({
                  'role': msg['role'],
                  'content': '[Previous tool results truncated]'
              })
          else:
              result.append(msg)
      return result
  ```

### Design note

The current tool result format uses `[Tool: name]` prefixes in the user message content (see `server.py:869-880`). The masking function keys off this pattern. If the format changes, update the detection logic.

---

## Sub-task 3: Token-Based Sliding Window

**Scope: Medium**

### What changes

Replace the turn-count windowing (`conversation[-50:]` at line 596, `pop(0)` at line 895) with token-budget windowing. Walk backwards from the most recent message, accumulating token cost, stopping when budget is exhausted.

### Files to modify

- **`server.py`** - Add configuration constant:
  ```python
  HISTORY_TOKEN_BUDGET = int(os.environ.get('NEMO_HISTORY_TOKEN_BUDGET', '2500'))
  ```

- **`server.py`** - Add windowing function:
  ```python
  def window_by_tokens(messages: list[dict], budget: int = HISTORY_TOKEN_BUDGET) -> list[dict]:
      """Select recent messages that fit within token budget.

      Walks backwards from most recent. Never splits a tool_call
      from its tool response (consecutive user messages after assistant).
      """
      result = []
      tokens_used = 0
      i = len(messages) - 1
      while i >= 0:
          msg = messages[i]
          cost = message_tokens(msg)
          if tokens_used + cost > budget:
              break
          result.insert(0, msg)
          tokens_used += cost
          i -= 1
      return result
  ```

- **`server.py:594-596`** - Replace history assembly:
  ```python
  # BEFORE:
  messages = [{'role': 'system', 'content': get_effective_system_prompt()}]
  messages.extend(conversation[-CFG['max_history']:])

  # AFTER:
  system_msg = {'role': 'system', 'content': get_effective_system_prompt()}
  masked = mask_observations(list(conversation))
  windowed = window_by_tokens(masked, HISTORY_TOKEN_BUDGET)
  messages = [system_msg] + windowed
  ```

- **`server.py:893-896`** - Simplify storage (remove turn-count cap, let token windowing handle it):
  ```python
  # BEFORE:
  if final_text:
      conversation.append({'role': 'assistant', 'content': final_text})
      while len(conversation) > CFG['max_history']:
          conversation.pop(0)

  # AFTER:
  if final_text:
      conversation.append({'role': 'assistant', 'content': final_text})
      # Keep a reasonable upper bound to prevent unbounded memory growth
      # Token windowing handles context selection at query time
      while len(conversation) > 200:
          conversation.pop(0)
  ```

### Tool-call pair integrity

The `window_by_tokens` function must not split assistant tool-call messages from their corresponding user tool-result messages. The current implementation walks backwards message-by-message, which naturally keeps pairs together since tool results always follow tool calls. If a tool-call message fits but its result doesn't, exclude both.

---

## Sub-task 4: Running Summary for Evicted Turns

**Scope: Medium-Large**

### What changes

When turns are evicted from the verbatim window, generate a brief running summary and prepend it as context. This preserves the gist of older conversation without the token cost.

### Design decision: Heuristic vs LLM summarization

**Option A (recommended):** Heuristic extraction - pull the last user question and first sentence of each assistant response from evicted turns. No API call needed, zero latency overhead.

**Option B:** LLM summarization via the inference endpoint. Higher quality but adds latency, costs tokens, and risks the "obscured stopping signals" problem from JetBrains research.

**Start with Option A. Upgrade to Option B only if heuristic summaries prove insufficient.**

### Files to modify

- **`server.py`** - Add summary builder:
  ```python
  SUMMARY_TOKEN_BUDGET = 200

  def build_running_summary(evicted: list[dict], existing_summary: str = '') -> str:
      """Build a running summary from evicted messages."""
      points = []
      if existing_summary:
          points.append(existing_summary)
      for msg in evicted:
          content = msg.get('content', '')
          if msg['role'] == 'user' and not content.startswith('['):
              # User question - keep first 100 chars
              points.append(f"User asked: {content[:100]}")
          elif msg['role'] == 'assistant':
              # Assistant response - first sentence
              first_sentence = content.split('.')[0][:120]
              points.append(f"Assistant: {first_sentence}")

      summary = '\n'.join(points[-10:])  # Keep last 10 points max
      # Trim to budget
      max_chars = SUMMARY_TOKEN_BUDGET * 4
      if len(summary) > max_chars:
          summary = summary[-max_chars:]
      return summary
  ```

- **`server.py`** - Add module-level state:
  ```python
  _running_summary: str = ''
  ```

- **`server.py:594-596`** - Update history assembly to include summary:
  ```python
  system_msg = {'role': 'system', 'content': get_effective_system_prompt()}
  masked = mask_observations(list(conversation))
  windowed = window_by_tokens(masked, HISTORY_TOKEN_BUDGET)

  # Build summary from turns that didn't make the window
  evicted = [m for m in masked if m not in windowed]
  if evicted:
      global _running_summary
      _running_summary = build_running_summary(evicted, _running_summary)

  # Inject summary as first user-visible context
  messages = [system_msg]
  if _running_summary:
      messages.append({
          'role': 'system',
          'content': f'Previous conversation summary:\n{_running_summary}'
      })
  messages.extend(windowed)
  ```

---

## Summary: Files and Functions to Modify

| File | Function/Location | Lines | Change | Scope |
|------|-------------------|-------|--------|-------|
| `server.py` | (new) `estimate_tokens()` | ~30 | Token estimation heuristic | Small |
| `server.py` | (new) `message_tokens()` | ~35 | Message-level token counting | Small |
| `server.py` | (new) `mask_observations()` | ~42 | Replace old tool outputs with placeholders | Medium |
| `server.py` | (new) `window_by_tokens()` | ~55 | Token-budget sliding window | Medium |
| `server.py` | (new) `build_running_summary()` | ~70 | Heuristic summary from evicted turns | Medium |
| `server.py` | message assembly | 594-596 | Wire up masking + windowing + summary | Small |
| `server.py` | conversation storage | 893-896 | Remove turn-count cap, raise safety limit | Small |
| `server.py` | (new) config constants | ~28 | `HISTORY_TOKEN_BUDGET`, `SUMMARY_TOKEN_BUDGET`, `VERBATIM_TOOL_RESULTS` | Small |

**Total estimated scope: Medium** (8 touch points, all in 1 file, no new dependencies)

---

## Risks and Edge Cases

1. **Token estimation inaccuracy** - chars/4 is approximate. Nemotron's actual tokenizer may differ. Monitor for context overflow via API errors.
2. **Observation masking too aggressive** - If the model needs old tool results for reasoning, masking hurts. The `VERBATIM_TOOL_RESULTS=3` default is conservative; tune if needed.
3. **Running summary drift** - Heuristic summaries may lose important nuance. If users report "the bot forgot what we discussed," consider upgrading to LLM summarization (Option B).
4. **Global state for summary** - `_running_summary` is per-process. In multi-worker deployments, each worker has its own summary. Acceptable for single-user Nemo deployment.
5. **Tool-call splitting** - The window function must handle the edge case where a tool-call fits but its result doesn't. Test with large tool outputs.

## Implementation Order

1. Sub-task 1 (token utils) - no dependencies
2. Sub-task 2 (observation masking) - depends on Sub-task 1
3. Sub-task 3 (token windowing) - depends on Sub-tasks 1, 2
4. Sub-task 4 (running summary) - depends on Sub-task 3

## Testing Checklist

- [ ] Token estimation returns reasonable values for text, tool calls, and tool results
- [ ] Observation masking preserves recent tool results and masks older ones
- [ ] Token windowing respects budget and doesn't split tool-call pairs
- [ ] Running summary captures key conversation points
- [ ] Full pipeline: system prompt + summary + windowed history fits within 4K context
- [ ] Server restart + live chat test confirms coherence improvement
