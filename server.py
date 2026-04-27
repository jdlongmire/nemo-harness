#!/usr/bin/env python3
"""
Nemo Harness - Harness-engineered chat interface for NVIDIA Nemotron

Built on harness engineering principles:
- Context engineering (hot/warm/cold memory tiers)
- Externalized state (SQLite memory, server-side conversations)
- Defense in depth (input validation, timeouts, circuit breakers)
- Observability (structured logging, correlation IDs, token tracking)
- Error recovery (circuit breakers, timeout with visible feedback)
- Behavioral alignment (configurable modes, memory-injected system prompts)

Usage:
    python server.py [--port 8091] [--host 0.0.0.0]
"""

import argparse
import asyncio
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import ssl
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from aiohttp import web
from dotenv import load_dotenv

# Load .env BEFORE importing tool modules (they read os.getenv at import time)
load_dotenv(Path(__file__).parent / '.env')

import memory_store
from mind_config import (
    DEFAULT_MODEL, MODES, MODE_HEADERS, GUIDE_BLOCKS, ALL_GUIDE_KEYS,
    build_mode_prompt, HISTORY_TOKEN_BUDGET, SUMMARY_TOKEN_BUDGET,
    VERBATIM_TOOL_RESULTS, INFERENCE_TIMEOUT, MAX_TOOL_ITERATIONS,
)
from tools.sandbox import Sandbox
from tools.registry import TOOL_REGISTRY
from tools.parser import ToolCallParser
from tools.context_sensor import classify_recent, get_relevant_tools, get_relevant_guides, ContextState, check_coherence
from inference_client import NemotronClient, InferenceResult, StreamEvent
import channels
import output_validator
import tools.file_tools
import tools.shell_tools
import tools.search_tools
import tools.web_tools
import tools.planning_tools
import tools.gitea_tools
import tools.rag_tools
import tools.document_tools
import tools.svg_tools

# --- Tool System Initialization ---
_sandbox_dirs = os.getenv('NEMO_SANDBOX_DIRS', '').strip()
if _sandbox_dirs:
    _sandbox = Sandbox([d.strip() for d in _sandbox_dirs.split(',') if d.strip()])
else:
    _sandbox = Sandbox()  # defaults to nemo-harness directory

tools.file_tools.register(_sandbox)
tools.shell_tools.register(_sandbox)
tools.search_tools.register(_sandbox)
tools.web_tools.register()
tools.planning_tools.register()
tools.gitea_tools.register()
tools.rag_tools.register()
tools.document_tools.register(_sandbox)
tools.svg_tools.register(_sandbox)

# Context window constants imported from mind_config


def estimate_tokens(text: str) -> int:
    """Estimate token count using chars/4 heuristic."""
    return len(text) // 4 if text else 0


def message_tokens(msg: dict) -> int:
    """Estimate tokens for a conversation message."""
    content = msg.get('content', '')
    if isinstance(content, str):
        return estimate_tokens(content) + 4  # role overhead
    return estimate_tokens(str(content)) + 4


def mask_observations(messages: list[dict], keep_recent: int = VERBATIM_TOOL_RESULTS) -> list[dict]:
    """Replace old tool result content with compact placeholders.

    Tool results use <tool_result name="..."> format. Keep the most recent
    `keep_recent` tool result messages verbatim; mask the rest.
    """
    tool_indices = [
        i for i, m in enumerate(messages)
        if m.get('role') == 'user' and '<tool_result' in m.get('content', '')
    ]
    to_mask = set(tool_indices[:-keep_recent]) if len(tool_indices) > keep_recent else set()
    result = []
    for i, msg in enumerate(messages):
        if i in to_mask:
            result.append({
                'role': msg['role'],
                'content': '[Previous tool results truncated]'
            })
        else:
            result.append(msg)
    return result


def window_by_tokens(messages: list[dict], budget: int = HISTORY_TOKEN_BUDGET) -> tuple[list[dict], list[dict]]:
    """Select recent messages that fit within token budget.

    Returns (windowed, evicted) where windowed fits the budget
    and evicted contains the rest. Never splits a tool-call from
    its tool response (assistant followed by user with tool_result).
    """
    windowed = []
    tokens_used = 0
    cut_index = 0  # everything before this index is evicted

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        cost = message_tokens(msg)
        if tokens_used + cost > budget:
            cut_index = i + 1
            break
        tokens_used += cost
    else:
        # All messages fit
        return list(messages), []

    # Don't split tool-call pairs: if the cut lands between an assistant
    # message and its tool-result user message, move the cut forward
    if cut_index < len(messages):
        msg_at_cut = messages[cut_index]
        if (cut_index > 0
                and msg_at_cut.get('role') == 'user'
                and '<tool_result' in msg_at_cut.get('content', '')
                and messages[cut_index - 1].get('role') == 'assistant'):
            # The user tool-result is included but the assistant call is not — drop both
            cut_index += 1

    evicted = messages[:cut_index]
    windowed = messages[cut_index:]
    return windowed, evicted


async def _summarize_via_model(text: str, existing_summary: str = '') -> str | None:
    """Call the inference API for a concise summary. Returns None on failure."""
    prompt = 'Summarize the following conversation excerpt in 3-5 bullet points. Be concise.'
    if existing_summary:
        prompt += f'\n\nPrior context:\n{existing_summary}'
    prompt += f'\n\nConversation:\n{text}'

    return await _inference_client.completion(
        model=CFG['model'],
        messages=[
            {'role': 'system', 'content': 'You are a summarization assistant. Output only bullet points.'},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0.3,
        max_tokens=SUMMARY_TOKEN_BUDGET,
        timeout=15.0,
    )


def _heuristic_summary(evicted: list[dict], existing_summary: str = '') -> str:
    """Fallback heuristic summary from evicted messages."""
    points = []
    if existing_summary:
        points.append(existing_summary)
    for msg in evicted:
        content = msg.get('content', '')
        if msg['role'] == 'user' and not content.startswith('[') and not content.startswith('<'):
            points.append(f"User asked: {content[:100]}")
        elif msg['role'] == 'assistant':
            first_sentence = content.split('.')[0][:120]
            if first_sentence.strip():
                points.append(f"Assistant: {first_sentence}")

    summary = '\n'.join(points[-10:])
    max_chars = SUMMARY_TOKEN_BUDGET * 4
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return summary


async def build_running_summary(evicted: list[dict], existing_summary: str = '') -> str:
    """Build a running summary from evicted messages.

    Uses the model itself for summarization (self-hosted = no per-call cost).
    Falls back to heuristic extraction on failure.
    """
    text_parts = []
    for msg in evicted:
        content = msg.get('content', '')
        if not content or content.startswith('[Previous tool'):
            continue
        role = msg.get('role', 'user')
        text_parts.append(f'{role}: {content[:300]}')

    if not text_parts:
        return existing_summary

    evicted_text = '\n'.join(text_parts[-15:])
    result = await _summarize_via_model(evicted_text, existing_summary)
    if result:
        return result
    return _heuristic_summary(evicted, existing_summary)


_running_summary: str = ''

# --- Structured Logging ---
logger = logging.getLogger('nemo-harness')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)


class CorrelatedLogger:
    """Logger adapter that prefixes correlation ID to every message."""

    def __init__(self, base_logger, correlation_id: str, model: str = ''):
        self._logger = base_logger
        self._cid = correlation_id
        self._model = model

    def info(self, msg, *args, **kwargs):
        prefix = f'[{self._cid}]' if not self._model else f'[{self._cid}] [{self._model}]'
        self._logger.info(f'{prefix} {msg}', *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        prefix = f'[{self._cid}]' if not self._model else f'[{self._cid}] [{self._model}]'
        self._logger.warning(f'{prefix} {msg}', *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        prefix = f'[{self._cid}]' if not self._model else f'[{self._cid}] [{self._model}]'
        self._logger.error(f'{prefix} {msg}', *args, **kwargs)


# --- Conversation Trace Logger ---
TRACE_DIR = Path(__file__).parent / 'traces'
TRACE_DIR.mkdir(exist_ok=True)


class TraceLogger:
    """Logs the full prompt/response chain for a conversation to a JSONL file."""

    def __init__(self, correlation_id: str, model: str):
        self._cid = correlation_id
        self._model = model
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        self._path = TRACE_DIR / f'{ts}_{correlation_id[:8]}.jsonl'

    def log(self, event_type: str, data: dict):
        entry = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'cid': self._cid,
            'model': self._model,
            'event': event_type,
            **data,
        }
        with open(self._path, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def prompt(self, messages: list, iteration: int):
        self.log('prompt', {'iteration': iteration, 'messages': messages})

    def response(self, raw_text: str, iteration: int):
        self.log('response', {'iteration': iteration, 'raw_text': raw_text})

    def tool_call(self, name: str, args: dict, iteration: int):
        self.log('tool_call', {'iteration': iteration, 'tool': name, 'args': args})

    def tool_result(self, name: str, result: dict, iteration: int):
        self.log('tool_result', {'iteration': iteration, 'tool': name, 'result': result})

    def final(self, text: str, iterations: int, elapsed: float):
        self.log('final', {'text': text[:2000], 'iterations': iterations, 'elapsed_s': round(elapsed, 2)})


# --- Behavioral Mode System (imported from mind_config.py) ---
# All Mind configuration (system prompts, guide blocks, mode definitions,
# temperature/token settings) lives in mind_config.py for structural 4M alignment.

# --- Runtime Config ---
CFG = {
    'api_base': os.getenv('NEMO_API_BASE', 'http://100.125.50.95:1237'),
    'model': os.getenv('NEMO_MODEL', DEFAULT_MODEL),
    'max_history': int(os.getenv('NEMO_MAX_HISTORY', '50')),
    'mode': 'default',
    'custom_system_prompt': None,  # overrides mode prompt when set
}

# --- Inference Client (Means abstraction) ---
_inference_client = NemotronClient(api_base=CFG['api_base'])

# --- Circuit Breaker ---
class CircuitBreaker:
    """Trip on repeated failures, auto-reset after cooldown."""

    def __init__(self, threshold: int = 5, cooldown: float = 60.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.last_failure = 0.0
        self.state = 'closed'  # closed=normal, open=tripped, half-open=testing

    def record_failure(self):
        self.failures += 1
        self.last_failure = time.monotonic()
        if self.failures >= self.threshold:
            self.state = 'open'
            logger.warning('Circuit breaker OPEN after %d failures', self.failures)

    def record_success(self):
        self.failures = 0
        self.state = 'closed'

    def can_proceed(self) -> bool:
        if self.state == 'closed':
            return True
        if self.state == 'open':
            if time.monotonic() - self.last_failure > self.cooldown:
                self.state = 'half-open'
                return True
            return False
        # half-open: allow one attempt
        return True


inference_breaker = CircuitBreaker(threshold=5, cooldown=60.0)

# --- Dynamic guide refresh ---
ENABLE_DYNAMIC_GUIDES = os.getenv('NEMO_ENABLE_DYNAMIC_GUIDES', 'true').lower() in ('true', '1', 'yes')
_context_state = ContextState()

# --- In-memory conversation history ---
conversation: list[dict] = []

# --- SSE Replay Buffer ---
# Jobs keyed by job_id, each has events list and metadata
jobs: dict[str, dict] = {}
active_job_id: str | None = None

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / 'web'
CONVERSATIONS_DIR = BASE_DIR / 'conversations'
EVALUATIONS_FILE = BASE_DIR / 'evaluations.jsonl'
USAGE_FILE = BASE_DIR / 'usage.jsonl'

CONVERSATIONS_DIR.mkdir(exist_ok=True)

# Action tag regex: [ACTION:remember|type|name|description|content]
ACTION_REMEMBER_RE = re.compile(
    r'\[ACTION:remember\|([^|]+)\|([^|]+)\|([^|]+)\|([^\]]+)\]',
    re.DOTALL
)

# --- Web Fetch Config ---
FETCH_TIMEOUT = 15  # seconds
FETCH_MAX_BYTES = 500 * 1024  # 500 KB


def _is_private_ip(hostname: str) -> bool:
    """Check if a hostname resolves to a private/reserved IP address."""
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return True
    except (socket.gaierror, ValueError):
        return True  # can't resolve = deny
    return False


def _strip_html_tags(html: str) -> str:
    """Extract readable text from HTML by stripping tags."""
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<head[^>]*>.*?</head>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<[^>]+>', ' ', html)
    html = re.sub(r'&nbsp;', ' ', html)
    html = re.sub(r'&amp;', '&', html)
    html = re.sub(r'&lt;', '<', html)
    html = re.sub(r'&gt;', '>', html)
    html = re.sub(r'&quot;', '"', html)
    html = re.sub(r'&#39;', "'", html)
    html = re.sub(r'\s+', ' ', html)
    return html.strip()


def get_effective_system_prompt(guide_keys: list[str] | None = None) -> str:
    """Build system prompt from mode + selected guides + memory context.

    If guide_keys is provided and dynamic guides are enabled, only those
    guide blocks are included.  Otherwise all guides are included (current behavior).
    """
    if CFG['custom_system_prompt']:
        base_prompt = CFG['custom_system_prompt']
    elif ENABLE_DYNAMIC_GUIDES and guide_keys is not None:
        mode_key = CFG['mode'] if CFG['mode'] in MODE_HEADERS else 'default'
        base_prompt = build_mode_prompt(mode_key, guide_keys)
    else:
        mode = MODES.get(CFG['mode'], MODES['default'])
        base_prompt = mode['system_prompt']

    # Inject tool schemas
    tool_block = TOOL_REGISTRY.prompt_block()
    if tool_block:
        base_prompt += f'\n\n{tool_block}'

    # Inject memory from SQLite via build_context_block()
    mem_block = memory_store.build_context_block()
    if mem_block:
        base_prompt += f'\n\n<memory>\n{mem_block}\n</memory>'

    return base_prompt


def get_effective_params() -> dict:
    """Get temperature and max_tokens from current mode."""
    mode = MODES.get(CFG['mode'], MODES['default'])
    return {
        'temperature': CFG.get('temperature_override', mode['temperature']),
        'max_tokens': mode['max_tokens'],
    }


def process_action_tags(text: str, clog: CorrelatedLogger) -> tuple[str, list[str]]:
    """Process action tags in response text. Returns (cleaned_text, actions_taken)."""
    actions = []

    def handle_remember(match):
        entry_type, name, description, content = match.group(1), match.group(2), match.group(3), match.group(4)
        try:
            result = memory_store.upsert_entry(entry_type.strip(), name.strip(), description.strip(), content.strip())
            actions.append(f'Remembered: {name.strip()} ({entry_type.strip()})')
            clog.info('Memory stored: type=%s name=%s', entry_type.strip(), name.strip())
            # Ch2: Mind -> Memory (persistence via action tag)
            channels.log_memory_persist(clog._cid, entry_type.strip(), name.strip())
        except Exception as e:
            actions.append(f'Failed to remember {name.strip()}: {e}')
            clog.error('Memory store failed: %s', e)
        return ''  # strip tag from visible output

    text = ACTION_REMEMBER_RE.sub(handle_remember, text)
    return text.strip(), actions


def log_token_usage(correlation_id: str, model: str, prompt_tokens: int, completion_tokens: int, mode: str = ''):
    """Append token usage to JSONL log."""
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'correlation_id': correlation_id,
        'model': model,
        'mode': mode or CFG['mode'],
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens,
    }
    try:
        with open(USAGE_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass


# ===== API HANDLERS =====

async def handle_status(request: web.Request) -> web.Response:
    """Health check with breaker state and memory health."""
    cid = str(uuid.uuid4())[:8]
    status_data = {
        'model': CFG['model'],
        'api_base': CFG['api_base'],
        'mode': CFG['mode'],
        'mode_name': MODES.get(CFG['mode'], MODES['default'])['name'],
        'circuit_breaker': inference_breaker.state,
        'memory': memory_store.db_health_check(),
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{CFG["api_base"]}/v1/models', timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    status_data['status'] = 'connected'
                    return web.json_response(status_data)
    except Exception:
        pass

    status_data['status'] = 'disconnected'
    return web.json_response(status_data, status=503)


async def handle_models(request: web.Request) -> web.Response:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{CFG["api_base"]}/v1/models', timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
                # Normalize: OpenAI returns {data: [...]}, we return flat list of model IDs
                if isinstance(data, dict) and 'data' in data:
                    models = [m['id'] if isinstance(m, dict) else m for m in data['data']]
                    return web.json_response(models)
                return web.json_response(data)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=502)


async def handle_modes(request: web.Request) -> web.Response:
    """List available modes or switch mode."""
    if request.method == 'GET':
        modes_list = {k: {'name': v['name'], 'model': v.get('model', DEFAULT_MODEL), 'active': k == CFG['mode']} for k, v in MODES.items()}
        return web.json_response(modes_list)

    body = await request.json()
    mode_key = body.get('mode', '').lower()
    if mode_key not in MODES:
        return web.json_response({'error': f'Unknown mode: {mode_key}', 'available': list(MODES.keys())}, status=400)

    CFG['mode'] = mode_key
    CFG['model'] = MODES[mode_key].get('model', DEFAULT_MODEL)
    CFG['custom_system_prompt'] = None  # clear override when switching modes
    logger.info('Mode switched to: %s (model=%s)', mode_key, CFG['model'])
    return web.json_response({'status': 'switched', 'mode': mode_key, 'name': MODES[mode_key]['name'], 'model': CFG['model']})


async def handle_history(request: web.Request) -> web.Response:
    return web.json_response(conversation)


async def handle_clear(request: web.Request) -> web.Response:
    global _running_summary, _context_state
    conversation.clear()
    _running_summary = ''
    _context_state = ContextState()
    return web.json_response({'status': 'cleared'})


async def handle_history_restore(request: web.Request) -> web.Response:
    body = await request.json()
    messages = body.get('messages', [])
    conversation.clear()
    for msg in messages:
        if msg.get('role') in ('user', 'assistant'):
            conversation.append({'role': msg['role'], 'content': msg.get('content', '')})
    while len(conversation) > 200:
        conversation.pop(0)
    return web.json_response({'status': 'restored', 'count': len(conversation)})


async def handle_chat(request: web.Request) -> web.Response:
    """Two-phase job queue: create job, return job_id for SSE streaming."""
    cid = str(uuid.uuid4())[:8]
    clog = CorrelatedLogger(logger, cid)

    body = await request.json()
    user_message = body.get('message', '').strip()
    if not user_message:
        return web.json_response({'error': 'Empty message'}, status=400)

    # Circuit breaker check
    if not inference_breaker.can_proceed():
        clog.warning('Circuit breaker OPEN, rejecting request')
        return web.json_response({
            'error': 'Inference server appears down. Retrying automatically in 60 seconds.',
            'circuit_breaker': 'open'
        }, status=503)

    model = body.get('model', CFG['model'])
    is_regenerate = body.get('regenerate', False)

    if is_regenerate:
        if conversation and conversation[-1]['role'] == 'assistant':
            conversation.pop()
    else:
        conversation.append({'role': 'user', 'content': user_message})

    # Dynamic guide refresh: classify intent from recent conversation
    if ENABLE_DYNAMIC_GUIDES:
        detected_intent, _scores = classify_recent(conversation, window=3)
        effective_intent = _context_state.update(detected_intent)
        guide_keys = get_relevant_guides(effective_intent)
        tool_subset = get_relevant_tools(effective_intent)
        tool_count_str = 'all' if tool_subset is None else str(len(tool_subset))
        clog.info('Intent: detected=%s effective=%s guides=%s tools=%s',
                  detected_intent, effective_intent, guide_keys, tool_count_str)
        channels.log_intent_selection(cid, detected_intent, effective_intent, guide_keys, tool_count_str)
        # Mission coherence check (Ch4): flag if intent mismatches active mode
        coherent, reason = check_coherence(effective_intent, CFG['mode'])
        channels.log_coherence_check(cid, effective_intent, CFG['mode'], coherent, reason)
        if not coherent:
            clog.warning('Mission coherence: %s', reason)
    else:
        guide_keys = None
        tool_subset = None

    # Build messages with token-aware windowing
    global _running_summary
    system_prompt = get_effective_system_prompt(guide_keys)
    # Ch1: Memory -> Mind (context injection)
    mem_block = memory_store.build_context_block()
    if mem_block:
        channels.log_memory_inject(cid, len(mem_block))
    system_msg = {'role': 'system', 'content': system_prompt}
    masked = mask_observations(list(conversation))
    windowed, evicted = window_by_tokens(masked, HISTORY_TOKEN_BUDGET)

    if evicted:
        _running_summary = await build_running_summary(evicted, _running_summary)
        # Ch2: Mind -> Memory (running summary)
        channels.log_summary_persist(cid, len(_running_summary))

    messages = [system_msg]
    if _running_summary:
        messages.append({
            'role': 'system',
            'content': f'Previous conversation summary:\n{_running_summary}'
        })
    messages.extend(windowed)

    # Get mode params (can be overridden per-request)
    mode_params = get_effective_params()
    temperature = float(body.get('temperature', mode_params['temperature']))
    max_tokens = int(body.get('max_tokens', mode_params['max_tokens']))

    # Create job for SSE replay buffer
    job_id = secrets.token_hex(8)
    job = {
        'events': [],
        'done': False,
        'notify': asyncio.Event(),
        'correlation_id': cid,
        'created_at': time.monotonic(),
    }
    jobs[job_id] = job

    # Cancel previous active job
    global active_job_id
    if active_job_id and active_job_id in jobs:
        old_job = jobs[active_job_id]
        if not old_job['done']:
            _emit_event(old_job, 'interrupted', {})
            old_job['done'] = True
            old_job['notify'].set()

    active_job_id = job_id

    # Prune old jobs (keep last 50)
    if len(jobs) > 100:
        sorted_ids = sorted(jobs.keys(), key=lambda k: jobs[k].get('created_at', 0))
        for old_id in sorted_ids[:len(jobs) - 50]:
            del jobs[old_id]

    # Launch processing task
    asyncio.create_task(_process_chat_job(job_id, model, messages, temperature, max_tokens, cid,
                                          tool_subset=tool_subset))

    clog.info('Job created: %s (mode=%s, model=%s)', job_id, CFG['mode'], model)
    return web.json_response({'job_id': job_id})


def _emit_event(job: dict, event_type: str, data: dict):
    """Add event to job's replay buffer and notify waiting streams."""
    event = {'type': event_type, **data}
    job['events'].append(event)
    job['notify'].set()


async def _call_inference(model: str, messages: list,
                          temperature: float, max_tokens: int, job: dict, clog,
                          tools: list[dict] | None = None) -> InferenceResult:
    """Single inference call via abstracted client. Streams tokens to job SSE."""
    full_response = []
    native_tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments_parts}
    had_error = False

    async for event in _inference_client.stream_completion(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, tools=tools, timeout=INFERENCE_TIMEOUT,
    ):
        if event.type == 'error':
            inference_breaker.record_failure()
            _emit_event(job, 'error', {'message': event.data.get('message', 'Unknown error')})
            clog.error('Inference error: %s', event.data.get('message', '')[:200])
            return InferenceResult(text='', tool_calls=[], error=True)

        if event.type == 'reasoning':
            _emit_event(job, 'reasoning', {'content': event.data['content']})
        elif event.type == 'content':
            full_response.append(event.data['content'])
            _emit_event(job, 'content', {'content': event.data['content']})
        elif event.type == 'tool_call_delta':
            tc_delta = event.data
            idx = tc_delta.get('index', 0)
            if idx not in native_tool_calls:
                native_tool_calls[idx] = {
                    'id': tc_delta.get('id', ''),
                    'name': '',
                    'arguments_parts': [],
                }
            entry = native_tool_calls[idx]
            if tc_delta.get('id'):
                entry['id'] = tc_delta['id']
            func = tc_delta.get('function', {})
            if func.get('name'):
                entry['name'] = func['name']
            if func.get('arguments'):
                entry['arguments_parts'].append(func['arguments'])
        elif event.type == 'usage':
            usage = event.data
            _emit_event(job, 'usage', {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
            })
            log_token_usage(clog._cid, model,
                            usage.get('prompt_tokens', 0),
                            usage.get('completion_tokens', 0))

    # Record success if we got here without error
    inference_breaker.record_success()

    # Assemble native tool calls
    assembled_calls = []
    for idx in sorted(native_tool_calls.keys()):
        entry = native_tool_calls[idx]
        assembled_calls.append({
            'id': entry['id'],
            'name': entry['name'],
            'arguments_str': ''.join(entry['arguments_parts']),
        })

    return InferenceResult(text=''.join(full_response), tool_calls=assembled_calls)


async def _process_chat_job(job_id: str, model: str, messages: list, temperature: float, max_tokens: int, cid: str,
                           tool_subset: set[str] | None = None):
    """Process inference with tool-calling loop. Max MAX_TOOL_ITERATIONS iterations."""
    job = jobs[job_id]
    clog = CorrelatedLogger(logger, cid, model=model)
    trace = TraceLogger(cid, model)
    start_time = time.monotonic()

    _emit_event(job, 'status', {'message': f'Sending to {model}...'})

    # Build native OpenAI tools list (filtered by intent when dynamic guides enabled)
    openai_tools = TOOL_REGISTRY.openai_tools(only=tool_subset)

    final_text = ''
    iteration = 0

    while iteration < MAX_TOOL_ITERATIONS:
        iteration += 1

        trace.prompt(messages, iteration)
        result = await _call_inference(model, messages, temperature, max_tokens, job, clog,
                                       tools=openai_tools)
        if result.error:
            trace.log('error', {'iteration': iteration, 'message': 'inference error'})
            job['done'] = True
            job['notify'].set()
            return

        trace.response(result.text, iteration)

        # Process action tags (memory) in text content
        cleaned_text, actions = process_action_tags(result.text, clog)
        for action in actions:
            _emit_event(job, 'action', {'message': action})

        # --- Native tool calls (preferred) ---
        if result.tool_calls:
            clog.info('Iteration %d: %d native tool call(s)', iteration, len(result.tool_calls))

            # Build the assistant message with tool_calls for conversation history
            assistant_msg = {'role': 'assistant', 'content': cleaned_text or None}
            assistant_msg['tool_calls'] = [
                {
                    'id': tc['id'],
                    'type': 'function',
                    'function': {
                        'name': tc['name'],
                        'arguments': tc['arguments_str'],
                    },
                }
                for tc in result.tool_calls
            ]
            messages.append(assistant_msg)

            for tc in result.tool_calls:
                try:
                    args = json.loads(tc['arguments_str'])
                except json.JSONDecodeError:
                    args = {}
                    clog.warning('Failed to parse tool args for %s: %s', tc['name'], tc['arguments_str'][:100])

                _emit_event(job, 'tool_call', {'name': tc['name'], 'args': args})
                clog.info('Executing tool: %s(%s)', tc['name'], json.dumps(args)[:200])
                trace.tool_call(tc['name'], args, iteration)

                exec_result = await TOOL_REGISTRY.execute(tc['name'], args)

                result_text = json.dumps(exec_result, default=str)[:4000]
                _emit_event(job, 'tool_result', {'name': tc['name'], 'result': exec_result})
                clog.info('Tool %s result: success=%s', tc['name'], exec_result.get('success', False))
                trace.tool_result(tc['name'], exec_result, iteration)

                # Send result back as role: "tool" message (OpenAI format)
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tc['id'],
                    'content': result_text,
                })

            _emit_event(job, 'status', {'message': f'Processing tool results (iteration {iteration})...'})
            continue

        # --- Fallback: text-based tool call parsing ---
        text_without_tools, parsed_calls = ToolCallParser.parse(cleaned_text)

        if not parsed_calls:
            # No tool calls: this is the final response
            final_text = cleaned_text
            break

        clog.info('Iteration %d: %d text-parsed tool call(s) (fallback)', iteration, len(parsed_calls))
        messages.append({'role': 'assistant', 'content': result.text})

        tool_results = []
        for tc in parsed_calls:
            _emit_event(job, 'tool_call', {'name': tc.name, 'args': tc.args})
            clog.info('Executing tool: %s(%s)', tc.name, json.dumps(tc.args)[:200])
            trace.tool_call(tc.name, tc.args, iteration)

            exec_result = await TOOL_REGISTRY.execute(tc.name, tc.args)

            result_text = json.dumps(exec_result, default=str)[:4000]
            _emit_event(job, 'tool_result', {'name': tc.name, 'result': exec_result})
            clog.info('Tool %s result: success=%s', tc.name, exec_result.get('success', False))
            trace.tool_result(tc.name, exec_result, iteration)

            if exec_result.get('success', False):
                tool_feedback = f'<tool_result name="{tc.name}" status="success">\n{result_text}\n</tool_result>'
            else:
                error_msg = exec_result.get('error', 'Unknown error')
                tool_feedback = (
                    f'<tool_result name="{tc.name}" status="error">\n'
                    f'ERROR: Tool "{tc.name}" failed: {error_msg}\n'
                    f'Full result: {result_text}\n'
                    f'You should inform the user about this error or try a different approach.\n'
                    f'</tool_result>'
                )
            tool_results.append(tool_feedback)

        results_block = '\n'.join(tool_results)
        messages.append({'role': 'user', 'content': results_block})
        _emit_event(job, 'status', {'message': f'Processing tool results (iteration {iteration})...'})

    else:
        # Hit max iterations
        clog.warning('Hit max tool iterations (%d)', MAX_TOOL_ITERATIONS)
        _emit_event(job, 'status', {'message': 'Reached tool iteration limit'})
        final_text = text_without_tools if 'text_without_tools' in dir() and text_without_tools else 'I reached the maximum number of tool iterations. Here is what I found so far.'

    # Morals post-check: validate output before delivery (Ch3: Morals->Mind)
    if final_text:
        validation = output_validator.validate_output(final_text, cid=cid)
        if not validation.passed:
            clog.warning('Output validation flagged %d violation(s): %s',
                         len(validation.violations), '; '.join(validation.violations))
            final_text = output_validator.redact_secrets(final_text)

    # Store final response in conversation
    # Token windowing handles context selection at query time;
    # keep a safety cap to prevent unbounded memory growth
    if final_text:
        conversation.append({'role': 'assistant', 'content': final_text})
        while len(conversation) > 200:
            conversation.pop(0)

    elapsed = time.monotonic() - start_time
    trace.final(final_text, iteration, elapsed)
    clog.info('Job %s completed in %.1fs (%d chars, %d iterations)', job_id, elapsed, len(final_text), iteration)

    _emit_event(job, 'done', {})
    job['done'] = True
    job['notify'].set()


async def handle_stream(request: web.Request) -> web.Response:
    """SSE stream endpoint with Last-Event-ID resume support."""
    job_id = request.match_info['job_id']

    if job_id not in jobs:
        return web.json_response({'error': 'Job not found'}, status=404)

    job = jobs[job_id]

    # Resume from Last-Event-ID
    last_id = request.headers.get('Last-Event-ID')
    offset = int(last_id) + 1 if last_id and last_id.isdigit() else 0

    response = web.StreamResponse(
        status=200,
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )
    await response.prepare(request)

    # SSE retry interval
    await response.write(b'retry: 3000\n: connected\n\n')

    sent = offset
    while True:
        # Send any buffered events
        while sent < len(job['events']):
            event = job['events'][sent]
            event_data = json.dumps(event)
            await response.write(f'id: {sent}\ndata: {event_data}\n\n'.encode())
            sent += 1

            if event['type'] in ('done', 'error', 'interrupted'):
                return response

        if job['done']:
            return response

        # Wait for new events with keepalive
        job['notify'].clear()
        try:
            await asyncio.wait_for(job['notify'].wait(), timeout=5.0)
        except asyncio.TimeoutError:
            # Keepalive comment
            try:
                await response.write(b': keepalive\n\n')
            except (ConnectionResetError, ConnectionAbortedError):
                return response


async def handle_config(request: web.Request) -> web.Response:
    if request.method == 'GET':
        mode = MODES.get(CFG['mode'], MODES['default'])
        effective_temp = CFG.get('temperature_override', mode['temperature'])
        return web.json_response({
            **CFG,
            'temperature': effective_temp,
            'system_prompt': get_effective_system_prompt(),
            'mode_name': mode['name'],
            'mode_temperature': mode['temperature'],
            'mode_max_tokens': mode['max_tokens'],
            'dynamic_guides': ENABLE_DYNAMIC_GUIDES,
            'current_intent': _context_state.current_intent,
        })

    body = await request.json()
    if 'system_prompt' in body:
        CFG['custom_system_prompt'] = body['system_prompt'] or None
    if 'max_history' in body:
        CFG['max_history'] = int(body['max_history'])
    if 'model' in body:
        CFG['model'] = body['model']
    if 'api_base' in body:
        CFG['api_base'] = body['api_base']
        _inference_client.api_base = body['api_base']
    if 'temperature' in body:
        CFG['temperature_override'] = float(body['temperature'])
    return web.json_response({'status': 'updated'})


# ===== MEMORY ENDPOINTS =====

async def handle_memory_list(request: web.Request) -> web.Response:
    entry_type = request.query.get('type')
    entries = memory_store.get_entries(entry_type)
    return web.json_response(entries)


async def handle_memory_search(request: web.Request) -> web.Response:
    query = request.query.get('q', '')
    if not query:
        return web.json_response({'error': 'Missing query parameter q'}, status=400)
    entries = memory_store.search_entries(query)
    return web.json_response(entries)


async def handle_memory_add(request: web.Request) -> web.Response:
    body = await request.json()
    # Auto-generate name from content if not provided
    if not body.get('name') and body.get('content'):
        body['name'] = body['content'][:50].strip()
    required = ('type', 'name', 'content')
    missing = [f for f in required if not body.get(f)]
    if missing:
        return web.json_response({'error': f'Missing fields: {", ".join(missing)}'}, status=400)

    try:
        entry = memory_store.upsert_entry(
            body['type'], body['name'],
            body.get('description', ''), body['content']
        )
        return web.json_response({'status': 'saved', 'entry': entry})
    except ValueError as e:
        return web.json_response({'error': str(e)}, status=400)


async def handle_memory_delete(request: web.Request) -> web.Response:
    entry_id = int(request.match_info['id'])
    if memory_store.delete_entry(entry_id):
        return web.json_response({'status': 'deleted'})
    return web.json_response({'error': 'Not found'}, status=404)


# ===== CONVERSATION PERSISTENCE =====

async def handle_conversations_save(request: web.Request) -> web.Response:
    body = await request.json()
    convo_id = body.get('id')
    if not convo_id:
        # Auto-generate ID if not provided
        convo_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')

    safe_id = ''.join(c for c in convo_id if c.isalnum() or c in '-_')
    if not safe_id:
        return web.json_response({'error': 'Invalid conversation id'}, status=400)

    messages = body.get('messages', [])
    title = body.get('title', '')
    if not title and messages:
        first_user = next((m for m in messages if m.get('role') == 'user'), None)
        if first_user:
            title = first_user['content'][:50]

    now = datetime.now(timezone.utc).isoformat()
    filepath = CONVERSATIONS_DIR / f'{safe_id}.json'

    created = now
    if filepath.exists():
        try:
            existing = json.loads(filepath.read_text())
            created = existing.get('created', now)
        except Exception:
            pass

    data = {
        'id': safe_id, 'title': title, 'messages': messages,
        'created': created, 'updated': now,
    }
    filepath.write_text(json.dumps(data, indent=2))
    return web.json_response({'status': 'saved', 'id': safe_id})


async def handle_conversations_list(request: web.Request) -> web.Response:
    convos = []
    for f in sorted(CONVERSATIONS_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            convos.append({
                'id': data.get('id', f.stem), 'title': data.get('title', 'Untitled'),
                'created': data.get('created', ''), 'updated': data.get('updated', ''),
                'message_count': len(data.get('messages', [])),
            })
        except Exception:
            continue
    return web.json_response(convos)


async def handle_conversation_get(request: web.Request) -> web.Response:
    convo_id = request.match_info['id']
    safe_id = ''.join(c for c in convo_id if c.isalnum() or c in '-_')
    filepath = CONVERSATIONS_DIR / f'{safe_id}.json'
    if not filepath.exists():
        return web.json_response({'error': 'Conversation not found'}, status=404)
    try:
        data = json.loads(filepath.read_text())
        return web.json_response(data)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def handle_conversation_delete(request: web.Request) -> web.Response:
    convo_id = request.match_info['id']
    safe_id = ''.join(c for c in convo_id if c.isalnum() or c in '-_')
    filepath = CONVERSATIONS_DIR / f'{safe_id}.json'
    if filepath.exists():
        filepath.unlink()
        return web.json_response({'status': 'deleted'})
    return web.json_response({'error': 'Not found'}, status=404)


# ===== EVALUATION =====

async def handle_evaluate(request: web.Request) -> web.Response:
    body = await request.json()
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'score': body.get('score', 0),
        'session_id': body.get('session_id', ''),
        'message_preview': body.get('message_preview', ''),
        'model': CFG['model'],
        'mode': CFG['mode'],
    }
    try:
        with open(EVALUATIONS_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        logger.error('Failed to write evaluation: %s', e)
        return web.json_response({'error': str(e)}, status=500)
    return web.json_response({'status': 'recorded'})


# ===== WEB FETCH =====

async def handle_fetch(request: web.Request) -> web.Response:
    """Fetch a URL server-side and return its text content."""
    cid = str(uuid.uuid4())[:8]
    clog = CorrelatedLogger(logger, cid)

    body = await request.json()
    url = body.get('url', '').strip()
    if not url:
        return web.json_response({'error': 'Missing url'}, status=400)

    # Validate URL scheme
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        clog.warning('Fetch rejected: invalid scheme %s', parsed.scheme)
        return web.json_response({'error': 'Only http and https URLs are allowed'}, status=400)

    if not parsed.hostname:
        return web.json_response({'error': 'Invalid URL'}, status=400)

    # Block private/local IPs
    if _is_private_ip(parsed.hostname):
        clog.warning('Fetch rejected: private IP for %s', parsed.hostname)
        return web.json_response({'error': 'Access to private/local addresses is not allowed'}, status=403)

    clog.info('Fetch started: %s', url)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT),
                max_redirects=5,
                headers={'User-Agent': 'NemoHarness/1.0'},
            ) as resp:
                if resp.status != 200:
                    clog.warning('Fetch HTTP %d for %s', resp.status, url)
                    return web.json_response({
                        'error': f'HTTP {resp.status}',
                        'url': url,
                    }, status=502)

                content_type = resp.content_type or 'application/octet-stream'

                # Read with size limit
                raw = await resp.content.read(FETCH_MAX_BYTES + 1)
                truncated = len(raw) > FETCH_MAX_BYTES
                if truncated:
                    raw = raw[:FETCH_MAX_BYTES]

                # Detect encoding
                encoding = resp.charset or 'utf-8'
                try:
                    text = raw.decode(encoding, errors='replace')
                except (LookupError, UnicodeDecodeError):
                    text = raw.decode('utf-8', errors='replace')

                # Strip HTML tags for html content
                if 'html' in content_type:
                    text = _strip_html_tags(text)

                clog.info('Fetch complete: %s (%s, %d chars, truncated=%s)',
                          url, content_type, len(text), truncated)

                return web.json_response({
                    'url': url,
                    'content_type': content_type,
                    'content': text,
                    'truncated': truncated,
                })

    except asyncio.TimeoutError:
        clog.warning('Fetch timeout for %s', url)
        return web.json_response({'error': f'Timeout after {FETCH_TIMEOUT}s', 'url': url}, status=504)
    except aiohttp.InvalidURL:
        return web.json_response({'error': 'Invalid URL format', 'url': url}, status=400)
    except aiohttp.ClientError as e:
        clog.warning('Fetch error for %s: %s', url, e)
        return web.json_response({'error': f'Fetch failed: {e}', 'url': url}, status=502)


# ===== ARTIFACT FILE SERVING =====

OUTPUT_DIR = BASE_DIR / 'output'

MIME_MAP = {
    '.svg': 'image/svg+xml',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.json': 'application/json',
    '.md': 'text/markdown',
    '.txt': 'text/plain',
    '.pdf': 'application/pdf',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.py': 'text/x-python',
    '.yaml': 'text/yaml',
    '.yml': 'text/yaml',
}


async def handle_output_file(request: web.Request) -> web.Response:
    """Serve files from the output/ directory for artifact preview/download."""
    filename = request.match_info['filename']
    # Sanitize: no path traversal
    safe_name = Path(filename).name
    filepath = (OUTPUT_DIR / safe_name).resolve()
    try:
        filepath.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        return web.Response(status=403, text='Access denied')

    if not filepath.exists() or not filepath.is_file():
        return web.json_response({'error': 'File not found'}, status=404)

    ext = filepath.suffix.lower()
    content_type = MIME_MAP.get(ext, 'application/octet-stream')

    # For binary formats, serve as download with proper content-disposition
    binary_exts = {'.docx', '.xlsx', '.pptx', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp'}
    if ext in binary_exts:
        return web.FileResponse(filepath, headers={
            'Content-Type': content_type,
            'Content-Disposition': f'inline; filename="{safe_name}"',
        })

    # Text-based: read and serve
    try:
        text = filepath.read_text(encoding='utf-8')
        return web.Response(text=text, content_type=content_type, charset='utf-8')
    except UnicodeDecodeError:
        return web.FileResponse(filepath)


# ===== APP SETUP =====

def create_app() -> web.Application:
    app = web.Application()

    # Core API
    app.router.add_get('/api/status', handle_status)
    app.router.add_get('/api/models', handle_models)
    app.router.add_get('/api/history', handle_history)
    app.router.add_post('/api/history/restore', handle_history_restore)
    app.router.add_post('/api/clear', handle_clear)
    app.router.add_post('/api/chat', handle_chat)
    app.router.add_get('/api/stream/{job_id}', handle_stream)
    app.router.add_get('/api/config', handle_config)
    app.router.add_post('/api/config', handle_config)

    # Behavioral modes
    app.router.add_get('/api/modes', handle_modes)
    app.router.add_post('/api/modes', handle_modes)

    # Memory
    app.router.add_get('/api/memory', handle_memory_list)
    app.router.add_get('/api/memory/search', handle_memory_search)
    app.router.add_post('/api/memory', handle_memory_add)
    app.router.add_delete('/api/memory/{id}', handle_memory_delete)

    # Conversations
    app.router.add_post('/api/conversations', handle_conversations_save)
    app.router.add_get('/api/conversations', handle_conversations_list)
    app.router.add_get('/api/conversations/{id}', handle_conversation_get)
    app.router.add_delete('/api/conversations/{id}', handle_conversation_delete)

    # Evaluation
    app.router.add_post('/api/evaluate', handle_evaluate)

    # Web Fetch
    app.router.add_post('/api/fetch', handle_fetch)

    # Serve generated artifacts from output/
    app.router.add_get('/api/files/output/{filename}', handle_output_file)

    # Gitea webhook indexer
    from indexer import create_indexer_app
    app.add_subapp('/indexer/', create_indexer_app())

    # Static files
    app.router.add_get('/', lambda r: web.FileResponse(WEB_DIR / 'index.html'))
    app.router.add_get('/{filename}', serve_web_file)

    return app


async def serve_web_file(request: web.Request) -> web.Response:
    filename = request.match_info['filename']
    filepath = WEB_DIR / filename
    # Prevent path traversal
    try:
        filepath.resolve().relative_to(WEB_DIR.resolve())
    except ValueError:
        return web.Response(status=403)
    if filepath.exists() and filepath.is_file():
        return web.FileResponse(filepath)
    return web.FileResponse(WEB_DIR / 'index.html')


async def run_server():
    parser = argparse.ArgumentParser(description='Nemo Harness - Harness-Engineered Nemotron Chat')
    parser.add_argument('--port', type=int, default=int(os.getenv('NEMO_PORT', '8091')))
    parser.add_argument('--host', default=os.getenv('NEMO_HOST', '0.0.0.0'))
    args = parser.parse_args()

    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()

    # TLS via Tailscale certs
    cert_dir = BASE_DIR / 'certs'
    cert_file = cert_dir / 'thinxai-workstation.crt'
    key_file = cert_dir / 'thinxai-workstation.key'

    ssl_ctx = None
    if cert_file.exists() and key_file.exists():
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(str(cert_file), str(key_file))
        logger.info('TLS enabled via Tailscale certs')

    site = web.TCPSite(runner, args.host, args.port, ssl_context=ssl_ctx)
    await site.start()

    proto = 'https' if ssl_ctx else 'http'
    logger.info('Nemo Harness running at %s://%s:%d', proto, args.host, args.port)
    logger.info('Model: %s @ %s | Mode: %s', CFG['model'], CFG['api_base'], CFG['mode'])

    while True:
        await asyncio.sleep(3600)


if __name__ == '__main__':
    asyncio.run(run_server())
