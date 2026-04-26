# Implementation Plan: Issue #21 - Wire memory.md + DB into prompt context

## Problem Statement

`build_context_block()` exists in `memory_store.py:132` but is never called. Instead, `get_effective_system_prompt()` (`server.py:422`) reads `memory.md` as a flat file (lines 436-443) and injects it raw into the system prompt. Meanwhile, all write operations (`upsert_entry`, `delete_entry`) go through SQLite only. This means:

1. **memory.md and SQLite are out of sync** - writes go to DB, reads come from flat file
2. **No token budget** - the entire memory.md is injected regardless of size
3. **No staleness filtering** - old entries persist forever
4. **No relevance filtering** - everything is injected regardless of conversation context

---

## Architecture Snapshot

| Component | File | Line | Current Role |
|-----------|------|------|-------------|
| `get_effective_system_prompt()` | `server.py` | 422 | Builds system prompt; reads `memory.md` flat file |
| `build_context_block()` | `memory_store.py` | 132 | Builds `<memory>` block from SQLite; **never called** |
| `upsert_entry()` | `memory_store.py` | 58 | Writes to SQLite only |
| `delete_entry()` | `memory_store.py` | 122 | Deletes from SQLite only |
| `process_action_tags()` | `server.py` | 457 | Calls `upsert_entry()` on `[ACTION:remember]` tags |
| `_MEMORY_BEHAVIOR` | `server.py` | 161 | System prompt text telling Nemo to use `memory.md` via `write_file` tool |

---

## Sub-task 1: Wire `build_context_block()` into `get_effective_system_prompt()`

**Scope: Small**

### What changes

Replace the flat-file memory.md read in `get_effective_system_prompt()` with a call to `memory_store.build_context_block()`.

### Files to modify

- **`server.py:435-443`** - Replace the `memory_file` block:
  ```python
  # BEFORE (lines 435-443):
  memory_file = BASE_DIR / 'memory.md'
  if memory_file.exists():
      try:
          mem_content = memory_file.read_text().strip()
          if mem_content:
              base_prompt += f'\n\n<memory>\n{mem_content}\n</memory>'
      except Exception as e:
          logger.warning('Failed to read memory.md: %s', e)

  # AFTER:
  mem_block = memory_store.build_context_block()
  if mem_block:
      base_prompt += f'\n\n<memory>\n{mem_block}\n</memory>'
  ```

### Token budget

- Add a `max_tokens` parameter to `build_context_block()` (default ~500 tokens, roughly 2000 chars).
- **`memory_store.py:132`** - Update signature: `build_context_block(max_tokens: int = 500) -> str`
- Use a conservative 4 chars/token estimate. Truncate the assembled block at `max_tokens * 4` chars, cutting at the last complete entry boundary.

---

## Sub-task 2: SQLite as single source of truth; memory.md as read-only projection

**Scope: Medium**

### What changes

After every write operation (`upsert_entry`, `delete_entry`), regenerate `memory.md` from SQLite. Update `_MEMORY_BEHAVIOR` to stop telling Nemo to write `memory.md` directly.

### Files to modify

- **`memory_store.py`** - Add new function `regenerate_memory_md()`:
  - Location: after `build_context_block()` (~line 145)
  - Reads all entries from SQLite via `get_entries()`
  - Writes `memory.md` in the existing markdown format (organized by type headers: User Profile, Facts & Preferences, Project Context, Session Notes)
  - Called at the end of `upsert_entry()` (line ~86, after `conn.commit()`) and `delete_entry()` (line ~127, after `conn.commit()`)

- **`memory_store.py:58` (`upsert_entry`)** - Add call: `regenerate_memory_md()` after commit, inside the `try` block before `return`

- **`memory_store.py:122` (`delete_entry`)** - Add call: `regenerate_memory_md()` after commit, before `return`

- **`server.py:161-171` (`_MEMORY_BEHAVIOR`)** - Rewrite to:
  - Remove instruction to "UPDATE memory.md using your write_file tool"
  - Instead: "Your persistent memory is stored in a database. The `<memory>` block in your context is auto-generated from it. Use `[ACTION:remember|type|name|description|content]` tags to store new memories. Use `/recall` and `/forget` commands to search and delete."
  - Remove "Write the ENTIRE file contents (it overwrites)" guidance

### Mapping of memory types to markdown sections

| DB type | Markdown header |
|---------|----------------|
| `user` | `## User Profile` |
| `feedback` | `## Facts & Preferences` |
| `project` | `## Project Context` |
| `reference` | `## References` |

---

## Sub-task 3: Relevance filtering interface

**Scope: Small (interface only; full implementation deferred)**

### What changes

Design the filtering interface in `build_context_block()` so it can accept a scoring function in the future, but for now inject all entries (the store is small).

### Files to modify

- **`memory_store.py:132` (`build_context_block`)** - Update signature:
  ```python
  def build_context_block(
      max_tokens: int = 500,
      score_fn: Callable[[dict], float] | None = None,
  ) -> str:
  ```
  - If `score_fn` is provided, call it on each entry, sort descending by score, and greedily pack entries until the token budget is exhausted.
  - If `score_fn` is `None` (default), include all entries ordered by `updated_at DESC` (current behavior), truncating at token budget.
  - The `score_fn` signature: `(entry: dict) -> float` where entry has keys: `id, type, name, description, content, created_at, updated_at`. Higher score = more relevant.

### Future composite scoring (design notes, no implementation now)

The scoring function should eventually combine:
1. **Recency** - days since `updated_at` (exponential decay)
2. **Type priority** - user > feedback > project > reference
3. **Semantic similarity** - cosine similarity of entry embedding vs. current query embedding
4. **Access frequency** - how often the entry has been retrieved (requires adding a `hit_count` column)

For now, the interface just needs to accept an optional callable. No embeddings, no frequency tracking.

---

## Sub-task 4: Staleness TTL by memory type

**Scope: Small**

### What changes

Add TTL filtering so stale entries are excluded from prompt injection (but NOT deleted from the database).

### Files to modify

- **`memory_store.py`** - Add TTL constants (near line 18, after `_VALID_TYPES`):
  ```python
  _TYPE_TTL_DAYS = {
      'user': None,       # infinite - facts about the user don't expire
      'feedback': None,   # infinite - behavioral guidance persists
      'project': 90,      # 90 days - project context goes stale
      'reference': None,  # infinite - external pointers stay valid
  }
  ```

- **`memory_store.py:132` (`build_context_block`)** - Add TTL filtering:
  - Before assembling the block, filter out entries where `type` has a TTL and `updated_at` is older than `now - TTL` days
  - This happens before the `score_fn` sort (if present) so expired entries never compete for token budget

- **`memory_store.py`** - Add new function `get_active_entries()` (~line 107, after `get_entries`):
  - Wraps `get_entries()` but applies TTL filtering
  - Used by `build_context_block()` instead of raw `get_entries()`
  - Also useful for the API layer if we want to expose "active only" views later

### Note on session type

The task mentions "session: 30d" TTL, but the current schema has no `session` type (valid types are: `user`, `feedback`, `project`, `reference`). Two options:
1. Add `session` as a valid type (requires schema migration)
2. Map session-like data to `project` type with 90d TTL

**Recommendation**: Do not add `session` type now. If needed later, it's a one-line addition to `_VALID_TYPES` and `_TYPE_TTL_DAYS`. The 90d TTL on `project` already covers the main staleness concern.

---

## Summary: Files and Functions to Modify

| File | Function/Location | Lines | Change | Scope |
|------|-------------------|-------|--------|-------|
| `server.py` | `get_effective_system_prompt()` | 435-443 | Replace flat-file read with `build_context_block()` call | Small |
| `server.py` | `_MEMORY_BEHAVIOR` | 161-171 | Rewrite to reflect DB-backed memory, remove write_file guidance | Small |
| `memory_store.py` | `build_context_block()` | 132-144 | Add `max_tokens` + `score_fn` params, TTL filtering, token budget | Medium |
| `memory_store.py` | `upsert_entry()` | ~86 | Add `regenerate_memory_md()` call after commit | Small |
| `memory_store.py` | `delete_entry()` | ~127 | Add `regenerate_memory_md()` call after commit | Small |
| `memory_store.py` | (new) `regenerate_memory_md()` | ~145 | New function: write memory.md from SQLite | Small |
| `memory_store.py` | (new) `get_active_entries()` | ~107 | New function: get_entries + TTL filter | Small |
| `memory_store.py` | (new) `_TYPE_TTL_DAYS` | ~18 | TTL constants by type | Small |

**Total estimated scope: Medium** (8 touch points, all in 2 files, no schema changes, no new dependencies)

---

## Risks and Edge Cases

1. **Race condition on regenerate**: If two concurrent requests both call `upsert_entry()`, the second `regenerate_memory_md()` wins. This is fine since the file is a projection and will be correct.
2. **Large memory store**: If entries grow beyond token budget, the greedy packing by recency means older entries silently drop out of context. This is acceptable and matches the design intent.
3. **Existing memory.md content**: On first deploy, any hand-edited content in `memory.md` that isn't in SQLite will be overwritten. Migration: import existing `memory.md` entries into SQLite before enabling regeneration.
4. **Tool-based writes**: The `_MEMORY_BEHAVIOR` text currently tells Nemo to use `write_file` on `memory.md`. After this change, direct file writes would be overwritten on next DB write. The updated prompt must clearly redirect Nemo to use action tags instead.

## Migration Checklist

- [ ] If `memory.md` has content not in SQLite, import it before deploying
- [ ] Verify `build_context_block()` output matches expected `<memory>` format
- [ ] Test that `regenerate_memory_md()` produces valid markdown
- [ ] Confirm `_MEMORY_BEHAVIOR` prompt changes don't break existing memory workflows
