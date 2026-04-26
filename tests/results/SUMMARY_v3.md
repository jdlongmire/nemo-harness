# Nemo Harness Test Summary — v3

**Date:** 2026-04-25
**Overall:** 24/38 passed (63.2%) — Grade: **C**

---

## Per-Batch Scores

| Batch | Passed | Total | Score | Grade |
|-------|--------|-------|-------|-------|
| 1. API Endpoints | 8 | 8 | 100.0% | A |
| 2. Identity & Modes | 1 | 5 | 20.0% | F |
| 3. Memory & Conversation | 4 | 6 | 66.7% | C |
| 4. File & Search Tools | 0 | 5 | 0.0% | F |
| 5. Web, Docs, Planning, RAG | 7 | 9 | 77.8% | B |
| 6. Orchestration & Settings | 4 | 5 | 80.0% | B |
| **Overall** | **24** | **38** | **63.2%** | **C** |

---

## Version Comparison

| Version | Passed | Total | Score | Grade | Delta |
|---------|--------|-------|-------|-------|-------|
| v1 | — | — | 39.0% | F | — |
| v2 | — | — | 52.6% | D | +13.6pp |
| **v3** | **24** | **38** | **63.2%** | **C** | **+10.6pp** |

**Trend:** 39% -> 52.6% -> 63.2% (+24.2pp total improvement)

---

## Remaining Failures (14 tests)

### Batch 2: Identity & Modes (4 failures)
| Test | Name | Notes |
|------|------|-------|
| 1.1 | Identity check | "nemo" not found in response |
| 1.2 | Mode switching | Empty responses for /mode technical and /mode creative |
| 1.3 | Mode listing | Empty response for /modes |
| 6.1 | Shell command (uname) | Tool called but "linux" not detected in response text |

### Batch 3: Memory & Conversation (2 failures)
| Test | Name | Notes |
|------|------|-------|
| 2.1 | Multi-turn context | Empty response — name recall failed |
| 3.1 | Memory store (chat) | Empty response — chat-based memory store not acknowledged |

### Batch 4: File & Search Tools (5 failures)
| Test | Name | Notes |
|------|------|-------|
| 4.1 | Read file (server.py) | No tool calls, no port found |
| 4.2 | Write file | No tool calls |
| 4.4 | List directory | No tool calls |
| 5.1 | Glob search | No tool calls, no .py files found |
| 5.2 | Grep search | No tool calls, no results |

### Batch 5: Web, Docs, Planning, RAG (2 failures)
| Test | Name | Notes |
|------|------|-------|
| 7.1 | Web fetch | No tool calls for web_fetch |
| 7.2 | Web search | No tool calls, timeout at 120s |

### Batch 6: Orchestration & Settings (1 failure)
| Test | Name | Notes |
|------|------|-------|
| 18.3 | Clarification behavior | Empty response — no clarifying question for ambiguous "Deploy the thing" |

---

## Analysis

**Strengths:**
- API endpoints: perfect 8/8 — all REST endpoints working correctly
- Document creation (PPTX, DOCX, XLSX, SVG): all passing
- Planning and RAG tools: fully functional
- Orchestration: multi-step and code analysis tasks working well
- Config get/set and model listing: solid

**Weaknesses:**
- File/search tools (batch 4): complete failure — tools not being called at all via chat
- Identity/modes: model not identifying as "Nemo", mode commands returning empty
- Several tests returning empty responses — possible SSE streaming issue for shorter responses
- Web tools (fetch/search) not triggered via chat prompts

**Root Cause Pattern:** Many failures show `tools=[]` and empty response text, suggesting the chat interface may not be streaming token/tool events for certain prompt types, or the model is not triggering tool use for file/web operations.
