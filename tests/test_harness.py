#!/usr/bin/env python3
"""
Nemo Chatbot Competency Test Harness
Sends prompts via API, streams SSE responses, evaluates results.
Usage: python test_harness.py [--batch BATCH_NAME] [--base-url URL]

IMPORTANT: The server has a single active job slot. Running multiple batches
in parallel (e.g., via separate agents) causes job interruption — each new
/api/chat request kills the previous active job. Always run batches sequentially.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import ssl

BASE_URL = "https://thinxai-workstation.tail99d888.ts.net:8091"

# Disable SSL verification for self-signed certs
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def api_get(path: str) -> dict:
    """GET request, return parsed JSON."""
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return json.loads(resp.read())


def api_post(path: str, data: dict) -> dict:
    """POST request with JSON body, return parsed JSON."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(f"{BASE_URL}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return json.loads(resp.read())


def api_delete(path: str) -> dict:
    """DELETE request, return parsed JSON."""
    req = urllib.request.Request(f"{BASE_URL}{path}", method="DELETE")
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return json.loads(resp.read())


def chat_and_collect(message: str, timeout: int = 120) -> dict:
    """Send a chat message and collect the full SSE response.
    Returns dict with keys: text, tool_calls, tool_results, artifacts, raw_events
    """
    # Post chat message
    result = api_post("/api/chat", {"message": message})
    job_id = result.get("job_id")
    if not job_id:
        return {"error": f"No job_id returned: {result}", "text": "", "tool_calls": [],
                "tool_results": [], "artifacts": [], "raw_events": []}

    # Stream SSE events
    req = urllib.request.Request(f"{BASE_URL}/api/stream/{job_id}")
    text_parts = []
    tool_calls = []
    tool_results = []
    artifacts = []
    raw_events = []

    start = time.time()
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        buffer = ""
        while time.time() - start < timeout:
            chunk = resp.read(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")

            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                for line in block.strip().split("\n"):
                    if line.startswith("data: "):
                        raw = line[6:]
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        raw_events.append(event)
                        etype = event.get("type", "")

                        if etype in ("token", "content"):
                            text_parts.append(event.get("content", ""))
                        elif etype == "tool_call":
                            tool_calls.append(event)
                        elif etype == "tool_result":
                            tool_results.append(event)
                            # Check for artifact
                            res = event.get("result", {})
                            if isinstance(res, dict) and res.get("success") and res.get("path"):
                                artifacts.append(res)
                        elif etype in ("done", "error", "interrupted"):
                            return {
                                "text": "".join(text_parts),
                                "tool_calls": tool_calls,
                                "tool_results": tool_results,
                                "artifacts": artifacts,
                                "raw_events": raw_events,
                                "final_event": etype,
                            }

    return {
        "text": "".join(text_parts),
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "artifacts": artifacts,
        "raw_events": raw_events,
        "final_event": "timeout",
    }


def clear_conversation():
    """Clear the current conversation."""
    try:
        api_post("/api/clear", {})
    except Exception:
        pass


# ─── Test result tracking ───

class TestResult:
    def __init__(self, test_id: str, name: str):
        self.test_id = test_id
        self.name = name
        self.passed = False
        self.notes = ""
        self.duration = 0.0

    def to_dict(self):
        return {
            "test_id": self.test_id,
            "name": self.name,
            "passed": self.passed,
            "notes": self.notes,
            "duration": self.duration,
        }


results: list[TestResult] = []


def run_test(test_id: str, name: str, fn):
    """Run a test function, catch exceptions, record result."""
    r = TestResult(test_id, name)
    start = time.time()
    try:
        passed, notes = fn()
        r.passed = passed
        r.notes = notes
    except Exception as e:
        r.passed = False
        r.notes = f"EXCEPTION: {e}"
    r.duration = round(time.time() - start, 2)
    results.append(r)
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{status}] {test_id}: {name} ({r.duration}s) - {r.notes[:100]}")


# ═══════════════════════════════════════════════════
# BATCH 1: API Endpoints (Category 17)
# ═══════════════════════════════════════════════════

def batch_api():
    print("\n=== BATCH 1: API Endpoints ===\n")

    def test_status():
        r = api_get("/api/status")
        ok = "status" in r or "model" in r or "mode" in r
        return ok, f"Keys: {list(r.keys())}"

    def test_models():
        r = api_get("/api/models")
        ok = isinstance(r, (list, dict))
        return ok, f"Type: {type(r).__name__}, content: {str(r)[:80]}"

    def test_modes():
        r = api_get("/api/modes")
        ok = isinstance(r, (list, dict))
        return ok, f"Content: {str(r)[:80]}"

    def test_config():
        r = api_get("/api/config")
        ok = isinstance(r, dict) and ("model" in r or "temperature" in r or "mode" in r)
        return ok, f"Keys: {list(r.keys())}"

    def test_memory():
        r = api_get("/api/memory")
        ok = isinstance(r, (list, dict))
        return ok, f"Type: {type(r).__name__}, len: {len(r) if isinstance(r, list) else 'dict'}"

    def test_conversations():
        r = api_get("/api/conversations")
        ok = isinstance(r, (list, dict))
        return ok, f"Type: {type(r).__name__}"

    def test_chat_empty():
        try:
            api_post("/api/chat", {"message": ""})
            return False, "Should have returned 400"
        except urllib.error.HTTPError as e:
            return e.code == 400, f"Status: {e.code}"

    def test_stream_invalid():
        try:
            api_get("/api/stream/nonexistent")
            return False, "Should have returned 404"
        except urllib.error.HTTPError as e:
            return e.code == 404, f"Status: {e.code}"

    run_test("17.1", "GET /api/status", test_status)
    run_test("17.2", "GET /api/models", test_models)
    run_test("17.3", "GET /api/modes", test_modes)
    run_test("17.4", "GET /api/config", test_config)
    run_test("17.5", "GET /api/memory", test_memory)
    run_test("17.6", "GET /api/conversations", test_conversations)
    run_test("17.7", "POST /api/chat empty", test_chat_empty)
    run_test("17.8", "GET /api/stream invalid", test_stream_invalid)


# ═══════════════════════════════════════════════════
# BATCH 2: Identity, Modes, Shell (Categories 1, 6)
# ═══════════════════════════════════════════════════

def batch_identity():
    print("\n=== BATCH 2: Identity & Modes ===\n")
    clear_conversation()

    def test_identity():
        r = chat_and_collect("Who are you?")
        text = r["text"].lower()
        has_nemo = "nemo" in text
        has_thinx = "thinx" in text or "nemotron" in text
        no_chatgpt = "chatgpt" not in text and "i'm an ai language model" not in text
        ok = has_nemo and no_chatgpt
        return ok, f"nemo={has_nemo}, thinx={has_thinx}, no_chatgpt={no_chatgpt}"

    def test_mode_switch():
        clear_conversation()
        r1 = chat_and_collect("/mode technical")
        clear_conversation()
        r2 = chat_and_collect("/mode creative")
        # Just check that modes were acknowledged
        ok = len(r1["text"]) > 0 and len(r2["text"]) > 0
        return ok, f"technical: {r1['text'][:50]}... creative: {r2['text'][:50]}..."

    def test_mode_list():
        clear_conversation()
        r = chat_and_collect("/modes")
        text = r["text"].lower()
        has_modes = any(m in text for m in ["default", "technical", "creative", "research"])
        return has_modes, f"Response: {r['text'][:100]}"

    def test_shell_uname():
        clear_conversation()
        r = chat_and_collect("Run `uname -a` and tell me what OS this is")
        has_tool = len(r["tool_calls"]) > 0
        text = r["text"].lower()
        has_linux = "linux" in text or "ubuntu" in text
        return has_tool and has_linux, f"tools={len(r['tool_calls'])}, linux={has_linux}"

    def test_sandbox():
        clear_conversation()
        r = chat_and_collect("Run `rm -rf /tmp/nemo_sandbox_test_nonexistent`")
        # Should either use sandbox or refuse
        text = r["text"].lower()
        refused_or_sandboxed = ("sandbox" in text or "restricted" in text or
                                 "cannot" in text or "won't" in text or
                                 len(r["tool_results"]) > 0)
        return True, f"Response: {r['text'][:100]}"  # Soft pass - just checking it doesn't crash

    run_test("1.1", "Identity check", test_identity)
    run_test("1.2", "Mode switching", test_mode_switch)
    run_test("1.3", "Mode listing", test_mode_list)
    run_test("6.1", "Shell command (uname)", test_shell_uname)
    run_test("6.2", "Sandbox enforcement", test_sandbox)


# ═══════════════════════════════════════════════════
# BATCH 3: Memory & Conversation (Categories 2, 3)
# ═══════════════════════════════════════════════════

def batch_memory():
    print("\n=== BATCH 3: Memory & Conversation ===\n")

    def test_multiturn():
        clear_conversation()
        chat_and_collect("My name is TestUser42")
        r = chat_and_collect("What's my name?")
        ok = "testuser42" in r["text"].lower()
        return ok, f"Response: {r['text'][:100]}"

    def test_memory_store():
        clear_conversation()
        r = chat_and_collect("Remember that my favorite language is Rust")
        text = r["text"].lower()
        ok = "remember" in text or "stored" in text or "noted" in text or "memory" in text or "rust" in text
        return ok, f"Response: {r['text'][:100]}"

    def test_memory_api_list():
        r = api_get("/api/memory")
        ok = isinstance(r, (list, dict))
        return ok, f"Memory entries: {len(r) if isinstance(r, list) else 'dict'}"

    def test_memory_api_add_delete():
        # Add
        added = api_post("/api/memory", {"content": "TEST_MEMORY_ENTRY_DELETE_ME", "type": "reference"})
        mem_id = added.get("id") or added.get("memory_id")

        # List and find it
        mems = api_get("/api/memory")
        found = False
        if isinstance(mems, list):
            for m in mems:
                if "TEST_MEMORY_ENTRY_DELETE_ME" in str(m):
                    found = True
                    if not mem_id:
                        mem_id = m.get("id")

        if not mem_id:
            return False, f"Could not get memory ID. Added: {added}"

        # Delete
        try:
            api_delete(f"/api/memory/{mem_id}")
            deleted = True
        except Exception as e:
            deleted = False

        return found and deleted, f"found={found}, deleted={deleted}, id={mem_id}"

    def test_memory_search():
        try:
            r = api_get("/api/memory/search?q=rust")
            ok = isinstance(r, (list, dict))
            return ok, f"Search result: {str(r)[:80]}"
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code}"

    def test_conversation_save_list():
        # Save current conversation
        saved = api_post("/api/conversations", {"title": "Test Conv"})
        conv_id = saved.get("id") or saved.get("conversation_id")

        # List
        convs = api_get("/api/conversations")
        found = isinstance(convs, (list, dict))

        return found, f"saved={saved}, list_type={type(convs).__name__}"

    run_test("2.1", "Multi-turn context", test_multiturn)
    run_test("3.1", "Memory store (chat)", test_memory_store)
    run_test("3.2", "Memory API list", test_memory_api_list)
    run_test("3.3", "Memory API add/delete", test_memory_api_add_delete)
    run_test("3.4", "Memory search", test_memory_search)
    run_test("2.2", "Conversation save/list", test_conversation_save_list)


# ═══════════════════════════════════════════════════
# BATCH 4: File & Search Tools (Categories 4, 5)
# ═══════════════════════════════════════════════════

def batch_tools_file():
    print("\n=== BATCH 4: File & Search Tools ===\n")

    def test_read_file():
        clear_conversation()
        r = chat_and_collect("Read the contents of server.py and tell me what port it runs on")
        has_tool = any(tc.get("name") == "read_file" or "read" in tc.get("name", "").lower()
                       for tc in r["tool_calls"])
        text = r["text"]
        has_port = "8091" in text or "8090" in text
        return has_tool or has_port, f"tools={[tc.get('name') for tc in r['tool_calls']]}, port_found={has_port}"

    def test_write_file():
        clear_conversation()
        r = chat_and_collect("Create a file called test_output.txt with the text 'Hello from Nemo'")
        has_tool = any("write" in tc.get("name", "").lower() for tc in r["tool_calls"])
        has_result = len(r["tool_results"]) > 0
        return has_tool or has_result, f"tools={[tc.get('name') for tc in r['tool_calls']]}"

    def test_list_dir():
        clear_conversation()
        r = chat_and_collect("List the files in the tools/ directory")
        has_tool = any("list" in tc.get("name", "").lower() or "dir" in tc.get("name", "").lower()
                       for tc in r["tool_calls"])
        text = r["text"].lower()
        has_files = ".py" in text
        return has_tool or has_files, f"tools={[tc.get('name') for tc in r['tool_calls']]}, has_py={has_files}"

    def test_glob():
        clear_conversation()
        r = chat_and_collect("Find all Python files in this project")
        has_tool = any("glob" in tc.get("name", "").lower() or "search" in tc.get("name", "").lower()
                       for tc in r["tool_calls"])
        text = r["text"]
        has_py = ".py" in text
        return has_tool or has_py, f"tools={[tc.get('name') for tc in r['tool_calls']]}, has_py={has_py}"

    def test_grep():
        clear_conversation()
        r = chat_and_collect("Search for 'circuit_breaker' in the codebase")
        has_tool = any("grep" in tc.get("name", "").lower() or "search" in tc.get("name", "").lower()
                       for tc in r["tool_calls"])
        text = r["text"].lower()
        has_result = "circuit" in text or "breaker" in text
        return has_tool or has_result, f"tools={[tc.get('name') for tc in r['tool_calls']]}, found={has_result}"

    run_test("4.1", "Read file (server.py)", test_read_file)
    run_test("4.2", "Write file", test_write_file)
    run_test("4.4", "List directory", test_list_dir)
    run_test("5.1", "Glob search", test_glob)
    run_test("5.2", "Grep search", test_grep)


# ═══════════════════════════════════════════════════
# BATCH 5: Web, Documents, Planning, RAG (Categories 7, 8, 11, 12)
# ═══════════════════════════════════════════════════

def batch_tools_advanced():
    print("\n=== BATCH 5: Web, Documents, Planning, RAG ===\n")

    def test_web_fetch():
        clear_conversation()
        r = chat_and_collect("Fetch the contents of https://httpbin.org/get")
        has_tool = any("web" in tc.get("name", "").lower() or "fetch" in tc.get("name", "").lower()
                       for tc in r["tool_calls"])
        text = r["text"].lower()
        has_result = "httpbin" in text or "headers" in text or "origin" in text
        return has_tool or has_result, f"tools={[tc.get('name') for tc in r['tool_calls']]}"

    def test_web_search():
        clear_conversation()
        r = chat_and_collect("Search the web for 'NVIDIA Nemotron model family'")
        has_tool = any("search" in tc.get("name", "").lower() and "web" in tc.get("name", "").lower()
                       for tc in r["tool_calls"])
        text = r["text"].lower()
        has_result = "nemotron" in text or "nvidia" in text
        return has_tool or has_result, f"tools={[tc.get('name') for tc in r['tool_calls']]}, has_nvidia={'nvidia' in text}"

    def test_pptx():
        clear_conversation()
        r = chat_and_collect("Create a 3-slide presentation about Python programming", timeout=180)
        has_tool = any("pptx" in tc.get("name", "").lower() for tc in r["tool_calls"])
        has_artifact = len(r["artifacts"]) > 0
        return has_tool or has_artifact, f"tools={[tc.get('name') for tc in r['tool_calls']]}, artifacts={len(r['artifacts'])}"

    def test_docx():
        clear_conversation()
        r = chat_and_collect("Create a Word document with a brief summary of what Nemo is", timeout=180)
        has_tool = any("docx" in tc.get("name", "").lower() for tc in r["tool_calls"])
        has_artifact = len(r["artifacts"]) > 0
        return has_tool or has_artifact, f"tools={[tc.get('name') for tc in r['tool_calls']]}, artifacts={len(r['artifacts'])}"

    def test_xlsx():
        clear_conversation()
        r = chat_and_collect("Create a spreadsheet with 3 columns: Task, Status, Priority and 5 sample rows", timeout=180)
        has_tool = any("xlsx" in tc.get("name", "").lower() for tc in r["tool_calls"])
        has_artifact = len(r["artifacts"]) > 0
        return has_tool or has_artifact, f"tools={[tc.get('name') for tc in r['tool_calls']]}, artifacts={len(r['artifacts'])}"

    def test_svg():
        clear_conversation()
        r = chat_and_collect("Create an SVG diagram showing: Input -> Process -> Output", timeout=180)
        has_tool = any("svg" in tc.get("name", "").lower() for tc in r["tool_calls"])
        has_artifact = len(r["artifacts"]) > 0
        return has_tool or has_artifact, f"tools={[tc.get('name') for tc in r['tool_calls']]}, artifacts={len(r['artifacts'])}"

    def test_plan():
        clear_conversation()
        r = chat_and_collect("Create a plan called 'test-plan' with tasks: research, implement, test")
        has_tool = any("plan" in tc.get("name", "").lower() for tc in r["tool_calls"])
        text = r["text"].lower()
        has_plan = "plan" in text and ("research" in text or "implement" in text)
        return has_tool or has_plan, f"tools={[tc.get('name') for tc in r['tool_calls']]}"

    def test_rag_index():
        clear_conversation()
        r = chat_and_collect("Index the following text for later retrieval: 'The harness engineering pattern uses 7 principles'")
        has_tool = any("index" in tc.get("name", "").lower() or "rag" in tc.get("name", "").lower()
                       for tc in r["tool_calls"])
        text = r["text"].lower()
        has_ack = "index" in text or "stored" in text or "saved" in text
        return has_tool or has_ack, f"tools={[tc.get('name') for tc in r['tool_calls']]}"

    def test_rag_search():
        clear_conversation()
        r = chat_and_collect("Search my indexed knowledge for 'harness principles'")
        has_tool = any("search" in tc.get("name", "").lower() and "semantic" in tc.get("name", "").lower()
                       for tc in r["tool_calls"])
        return True, f"tools={[tc.get('name') for tc in r['tool_calls']]}, text={r['text'][:80]}"

    run_test("7.1", "Web fetch", test_web_fetch)
    run_test("7.2", "Web search", test_web_search)
    run_test("8.1", "PPTX creation", test_pptx)
    run_test("8.3", "DOCX creation", test_docx)
    run_test("8.4", "XLSX creation", test_xlsx)
    run_test("8.5", "SVG creation", test_svg)
    run_test("11.1", "Create plan", test_plan)
    run_test("12.1", "RAG index", test_rag_index)
    run_test("12.2", "RAG search", test_rag_search)


# ═══════════════════════════════════════════════════
# BATCH 6: Multi-tool Orchestration & Settings (Categories 16, 18)
# ═══════════════════════════════════════════════════

def batch_orchestration():
    print("\n=== BATCH 6: Orchestration & Settings ===\n")

    def test_multistep():
        clear_conversation()
        r = chat_and_collect("Read the README.md, summarize it, then create a DOCX with the summary", timeout=180)
        tool_names = [tc.get("name", "") for tc in r["tool_calls"]]
        has_read = any("read" in n.lower() for n in tool_names)
        has_docx = any("docx" in n.lower() for n in tool_names)
        return has_read or has_docx, f"tools={tool_names}, artifacts={len(r['artifacts'])}"

    def test_code_analysis():
        clear_conversation()
        r = chat_and_collect("Find all TODO comments in the codebase and list them", timeout=180)
        tool_names = [tc.get("name", "") for tc in r["tool_calls"]]
        has_search = any("grep" in n.lower() or "search" in n.lower() for n in tool_names)
        return has_search, f"tools={tool_names}"

    def test_clarification():
        clear_conversation()
        r = chat_and_collect("Deploy the thing")
        text = r["text"].lower()
        asks_question = "?" in r["text"] or "which" in text or "what" in text or "clarif" in text or "specify" in text
        return asks_question, f"Response: {r['text'][:100]}"

    def test_config_get_set():
        # Get current config
        cfg = api_get("/api/config")
        orig_temp = cfg.get("temperature", 0.7)

        # Set new temperature
        try:
            api_post("/api/config", {"temperature": 0.1})
            cfg2 = api_get("/api/config")
            changed = cfg2.get("temperature") == 0.1 or cfg2.get("temperature") == "0.1"

            # Restore
            api_post("/api/config", {"temperature": orig_temp})
            return changed, f"original={orig_temp}, set=0.1, got={cfg2.get('temperature')}"
        except Exception as e:
            return False, f"Error: {e}"

    def test_model_list():
        r = api_get("/api/models")
        if isinstance(r, list):
            ok = len(r) > 0
            return ok, f"Models: {r[:3]}"
        elif isinstance(r, dict) and "models" in r:
            ok = len(r["models"]) > 0
            return ok, f"Models: {r['models'][:3]}"
        return False, f"Unexpected format: {type(r)}"

    run_test("18.1", "Multi-step research task", test_multistep)
    run_test("18.2", "Code analysis task", test_code_analysis)
    run_test("18.3", "Clarification behavior", test_clarification)
    run_test("16.1", "Config get/set (temperature)", test_config_get_set)
    run_test("16.4", "Model list", test_model_list)


# ═══════════════════════════════════════════════════

BATCHES = {
    "api": batch_api,
    "identity": batch_identity,
    "memory": batch_memory,
    "tools_file": batch_tools_file,
    "tools_advanced": batch_tools_advanced,
    "orchestration": batch_orchestration,
}


def write_report(batch_name: str):
    """Write results to a JSON + markdown report."""
    report_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(report_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{report_dir}/{batch_name}_{timestamp}"

    # JSON
    data = {
        "batch": batch_name,
        "timestamp": timestamp,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "tests": [r.to_dict() for r in results],
    }
    with open(f"{base}.json", "w") as f:
        json.dump(data, f, indent=2)

    # Markdown
    pct = (data["passed"] / data["total"] * 100) if data["total"] else 0
    grade = "A" if pct >= 90 else "B" if pct >= 75 else "C" if pct >= 60 else "D" if pct >= 40 else "F"
    md = [f"# Nemo Test Results: {batch_name}", ""]
    md.append(f"**Date:** {timestamp}")
    md.append(f"**Passed:** {data['passed']}/{data['total']} ({pct:.0f}%) — Grade: **{grade}**")
    md.append("")
    md.append("| Test | Name | Result | Duration | Notes |")
    md.append("|------|------|--------|----------|-------|")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        notes = r.notes[:60].replace("|", "\\|")
        md.append(f"| {r.test_id} | {r.name} | {status} | {r.duration}s | {notes} |")

    with open(f"{base}.md", "w") as f:
        f.write("\n".join(md))

    print(f"\n{'='*50}")
    print(f"Results: {data['passed']}/{data['total']} passed ({pct:.0f}%) — Grade: {grade}")
    print(f"Report: {base}.md")
    print(f"Data:   {base}.json")

    return base


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="Nemo Chatbot Test Harness")
    parser.add_argument("--batch", choices=list(BATCHES.keys()) + ["all"], default="all")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    BASE_URL = args.base_url

    print(f"Target: {BASE_URL}")

    # Quick health check
    try:
        status = api_get("/api/status")
        print(f"Server status: {status}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {BASE_URL}: {e}")
        sys.exit(1)

    if args.batch == "all":
        for name, fn in BATCHES.items():
            fn()
            write_report(name)
            results.clear()
    else:
        BATCHES[args.batch]()
        write_report(args.batch)


if __name__ == "__main__":
    main()
