# Nemo Harness -- Consolidated Test Summary (v2)

**Date:** 2026-04-25
**Model:** nvidia/nemotron-3-nano
**Target:** thinxai-workstation (Tailscale)

## Overall Score

**20 / 38 passed (52.6%) -- Grade: F**

## Comparison to Previous Run

| Metric | Previous | Current | Delta |
|--------|----------|---------|-------|
| Passed | 15 / 38 | 20 / 38 | +5 |
| Rate | 39.5% | 52.6% | +13.1pp |

Improvement of **+13 percentage points** over previous run.

## Per-Batch Breakdown

| # | Batch | Prev | Now | Total | % | Grade |
|---|-------|------|----|-------|---|-------|
| 1 | API Endpoints | 8 | 8 | 8 | 100% | A |
| 2 | Identity & Modes | 1 | 1 | 5 | 20% | F |
| 3 | Memory & Conversation | 2 | 4 | 6 | 67% | C |
| 4 | File & Search Tools | 0 | 0 | 5 | 0% | F |
| 5 | Web, Docs, Planning, RAG | 2 | 4 | 9 | 44% | D |
| 6 | Orchestration & Settings | 2 | 3 | 5 | 60% | C |
| | **TOTAL** | **15** | **20** | **38** | **52.6%** | **F** |

### Gains Since Previous Run (+5 tests)
- Memory: 3.3 Memory API add/delete, 2.2 Conversation save/list (+2)
- Advanced Tools: 8.5 SVG creation, 11.1 Create plan (+2)
- Orchestration: 18.2 Code analysis task (+1)

## Remaining Failures (18 tests)

### Batch 2: Identity & Modes (4 failures)
| Test | Name | Details | Notes |
|------|------|---------|-------|
| 1.1 | Identity check | nemo=False, thinx=False | Model not identifying as Nemo/ThinxS |
| 1.2 | Mode switching | technical/creative mismatch | Mode switch not reflected in responses |
| 1.3 | Mode listing | Empty response | Model fails to list available modes via chat |
| 6.1 | Shell command (uname) | tools=1, linux=False | Shell tool called but output not parsed correctly |

### Batch 3: Memory & Conversation (2 failures)
| Test | Name | Details | Notes |
|------|------|---------|-------|
| 2.1 | Multi-turn context | Empty response | Context not maintained across turns |
| 3.1 | Memory store (chat) | Empty response | Memory store via chat prompt not working |

### Batch 4: File & Search Tools (5 failures)
| Test | Name | Details | Notes |
|------|------|---------|-------|
| 4.1 | Read file (server.py) | tools=[], port_found=False | Model not invoking read_file tool |
| 4.2 | Write file | tools=[] | Model not invoking write_file tool |
| 4.4 | List directory | tools=[], has_py=False | Model not invoking list_directory tool |
| 5.1 | Glob search | tools=[], has_py=False | Model not invoking glob_search tool |
| 5.2 | Grep search | tools=[], found=False | Model not invoking grep_search tool |

### Batch 5: Web, Docs, Planning, RAG (5 failures)
| Test | Name | Details | Notes |
|------|------|---------|-------|
| 7.1 | Web fetch | tools=[] | Model not invoking web_fetch tool |
| 7.2 | Web search | tools=[], has_nvidia=False | Model not invoking web_search tool |
| 8.1 | PPTX creation | tools=[], artifacts=0 | Model not invoking create_pptx tool |
| 8.3 | DOCX creation | tools=[], artifacts=0 | Model not invoking create_docx tool |
| 8.4 | XLSX creation | tools=[], artifacts=0 | 120s timeout, no tool call |

### Batch 6: Orchestration & Settings (2 failures)
| Test | Name | Details | Notes |
|------|------|---------|-------|
| 18.1 | Multi-step research | tools=[], artifacts=0 | No tools invoked for research task |
| 18.3 | Clarification behavior | Empty response | Model not asking clarifying questions |

## Key Observations

1. **API layer is solid** -- all 8 endpoint tests pass (100%). Server infrastructure works correctly.
2. **Tool invocation is the biggest gap** -- File tools (0/5), web tools (0/2), and document creation (0/3) all fail because the model never calls available tools. This accounts for 10 of 18 failures.
3. **Identity/system prompt** needs work -- model doesn't identify itself correctly or list modes via natural language.
4. **Memory API works, chat-based memory doesn't** -- direct API calls pass but prompting the model to store memory fails.
5. **New passes in SVG, planning, RAG, and code analysis** show the tool registration improvements are working for some tools.

## Recommended Priority Fixes

1. **Strengthen tool-use system prompt** -- add explicit instructions and few-shot examples for file, web, and document tools.
2. **Fix chat-based memory/context** -- multi-turn context and memory-store-via-chat both return empty.
3. **Enforce identity in system prompt** -- add stronger identity anchoring ("You are Nemo, built by ThinxAI").
4. **Debug shell command output parsing** -- uname tool is called but "linux" not found in response.
5. **Document tool registration** -- PPTX/DOCX/XLSX tools appear unregistered or model doesn't know about them.
