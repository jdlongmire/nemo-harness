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

import memory_store
from tools.sandbox import Sandbox
from tools.registry import TOOL_REGISTRY
from tools.parser import ToolCallParser
import tools.file_tools
import tools.shell_tools
import tools.search_tools
import tools.web_tools

load_dotenv(Path(__file__).parent / '.env')

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

MAX_TOOL_ITERATIONS = 10

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


# --- Behavioral Mode System ---
DEFAULT_MODEL = 'nvidia/nemotron-3-nano-4b'

_CAPABILITIES = (
    'Your capabilities:\n'
    '- /fetch <url> : fetch and read web pages\n'
    '- /mode <name> or /modes : switch modes (default, technical, creative, research)\n'
    '- /remember <text> : store info to persistent memory\n'
    '- /recall <query> : search persistent memory\n'
    '- /forget <id> : delete a memory entry\n'
    '- Mode and model switching via UI dropdowns\n'
    '- Conversation save/load/delete, response regeneration, thumbs up/down evaluation\n'
    '\nYou also have tool-use capabilities for file operations, shell commands, and code search.\n'
    'Use tools when the user asks you to read, write, edit, search, or run commands.\n'
)

_BEHAVIORAL_CORE = (
    'Core rules:\n'
    '- Truth over satisfaction: never sacrifice accuracy for approval.\n'
    '- Say "I don\'t know" when you do not know. Distinguish computation from pattern-matching.\n'
    '- Present conclusions as proposals, not pronouncements. Respect user agency.\n'
    '- Moderate tone: helpful without being effusive. Skepticism over enthusiasm.\n'
    '- Obstacles are opportunities for analysis, not reasons to weaken claims.\n'
    '- Acknowledge valid corrections; push back if a correction is wrong.\n'
    '- Distinguish primary from secondary sources. Flag confidence levels.\n'
    '- Direct assertions, no hedging. Concrete before abstract. Critical of claims, not persons.\n'
    '- Never use em dashes; use colons, parentheses, or en dashes instead.\n'
    '- Use markdown. Use lists only when content demands enumeration. '
    'No symmetric reversals ("not X, but Y").\n'
    '- When you use web_search or web_fetch, ALWAYS list your sources at the end of your response. '
    'Format as a numbered "Sources" section with the title and URL of each page you referenced. '
    'Do not just summarize what you found: cite where you found it so the user can verify.\n'
)

MODES = {
    'default': {
        'name': 'Default',
        'model': DEFAULT_MODEL,
        'system_prompt': (
            'You are Nemo. You are NOT a generic AI assistant. You are a purpose-built assistant '
            'created for the ThinxAI platform, running on NVIDIA Nemotron inference. '
            'When asked who you are, say: "I\'m Nemo, the ThinxAI assistant. I run on NVIDIA Nemotron '
            'and I\'m built for research, technical work, and creative tasks. I can read and write files, '
            'run shell commands, search codebases, fetch web pages, and remember things across conversations."\n\n'
            + _CAPABILITIES + '\n'
            + _BEHAVIORAL_CORE + '\n'
            'Mode: Default. Balanced tone. Provide clear, accurate, concise responses.'
        ),
        'temperature': 0.7,
        'max_tokens': 2048,
    },
    'technical': {
        'name': 'Technical',
        'model': DEFAULT_MODEL,
        'system_prompt': (
            'You are Nemo, the ThinxAI technical assistant running on NVIDIA Nemotron.\n\n'
            + _CAPABILITIES + '\n'
            + _BEHAVIORAL_CORE + '\n'
            'Mode: Technical. Prioritize accuracy over brevity. Include code examples when relevant. '
            'Use structured formatting (headers, lists, code blocks). '
            'Distinguish established facts from inferences.'
        ),
        'temperature': 0.3,
        'max_tokens': 4096,
    },
    'creative': {
        'name': 'Creative',
        'model': DEFAULT_MODEL,
        'system_prompt': (
            'You are Nemo, the ThinxAI creative assistant running on NVIDIA Nemotron.\n\n'
            + _CAPABILITIES + '\n'
            + _BEHAVIORAL_CORE + '\n'
            'Mode: Creative. Write with vivid language, varied sentence structure, and narrative flow. '
            'Take creative risks. Explore ideas from unexpected angles.'
        ),
        'temperature': 1.0,
        'max_tokens': 4096,
    },
    'research': {
        'name': 'Research',
        'model': DEFAULT_MODEL,
        'system_prompt': (
            'You are Nemo, the ThinxAI research assistant running on NVIDIA Nemotron.\n\n'
            + _CAPABILITIES + '\n'
            + _BEHAVIORAL_CORE + '\n'
            'Mode: Research. Analyze claims carefully. Distinguish evidence from inference. '
            'Cite reasoning steps explicitly. Flag assumptions. '
            'When uncertain, state so clearly rather than speculating.'
        ),
        'temperature': 0.4,
        'max_tokens': 4096,
    },
}

# --- Runtime Config ---
CFG = {
    'api_base': os.getenv('NEMO_API_BASE', 'http://100.125.50.95:1237'),
    'model': os.getenv('NEMO_MODEL', DEFAULT_MODEL),
    'max_history': int(os.getenv('NEMO_MAX_HISTORY', '50')),
    'mode': 'default',
    'custom_system_prompt': None,  # overrides mode prompt when set
}

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

INFERENCE_TIMEOUT = 120  # seconds

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


def get_effective_system_prompt() -> str:
    """Build system prompt from mode + memory context + tool schemas."""
    if CFG['custom_system_prompt']:
        base_prompt = CFG['custom_system_prompt']
    else:
        mode = MODES.get(CFG['mode'], MODES['default'])
        base_prompt = mode['system_prompt']

    # Inject tool schemas
    tool_block = TOOL_REGISTRY.prompt_block()
    if tool_block:
        base_prompt += f'\n\n{tool_block}'

    # Inject memory context
    mem_block = memory_store.build_context_block()
    if mem_block:
        base_prompt += f'\n\n<memory>\n{mem_block}\n</memory>'

    return base_prompt


def get_effective_params() -> dict:
    """Get temperature and max_tokens from current mode."""
    mode = MODES.get(CFG['mode'], MODES['default'])
    return {
        'temperature': mode['temperature'],
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
    conversation.clear()
    return web.json_response({'status': 'cleared'})


async def handle_history_restore(request: web.Request) -> web.Response:
    body = await request.json()
    messages = body.get('messages', [])
    conversation.clear()
    for msg in messages:
        if msg.get('role') in ('user', 'assistant'):
            conversation.append({'role': msg['role'], 'content': msg.get('content', '')})
    while len(conversation) > CFG['max_history']:
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

    # Build messages with behavioral system prompt
    messages = [{'role': 'system', 'content': get_effective_system_prompt()}]
    messages.extend(conversation[-CFG['max_history']:])

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
    asyncio.create_task(_process_chat_job(job_id, model, messages, temperature, max_tokens, cid))

    clog.info('Job created: %s (mode=%s, model=%s)', job_id, CFG['mode'], model)
    return web.json_response({'job_id': job_id})


def _emit_event(job: dict, event_type: str, data: dict):
    """Add event to job's replay buffer and notify waiting streams."""
    event = {'type': event_type, **data}
    job['events'].append(event)
    job['notify'].set()


async def _call_inference(session: aiohttp.ClientSession, model: str, messages: list,
                          temperature: float, max_tokens: int, job: dict, clog) -> str | None:
    """Single inference call. Streams tokens to job SSE. Returns full response text or None on error."""
    full_response = []

    try:
        payload = {
            'model': model,
            'messages': messages,
            'stream': True,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }

        async with session.post(
            f'{CFG["api_base"]}/v1/chat/completions',
            json=payload,
            timeout=aiohttp.ClientTimeout(total=INFERENCE_TIMEOUT)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                inference_breaker.record_failure()
                _emit_event(job, 'error', {'message': f'API error {resp.status}: {error_text}'})
                clog.error('API error %d: %s', resp.status, error_text[:200])
                return None

            inference_breaker.record_success()

            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if not line.startswith('data: '):
                    continue
                data = line[6:]
                if data == '[DONE]':
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk.get('choices', [{}])[0].get('delta', {})

                    if delta.get('reasoning_content'):
                        _emit_event(job, 'reasoning', {'content': delta['reasoning_content']})

                    if delta.get('content'):
                        full_response.append(delta['content'])
                        _emit_event(job, 'content', {'content': delta['content']})

                    usage = chunk.get('usage')
                    if usage:
                        _emit_event(job, 'usage', {
                            'prompt_tokens': usage.get('prompt_tokens', 0),
                            'completion_tokens': usage.get('completion_tokens', 0),
                        })
                        log_token_usage(clog._cid, model,
                                        usage.get('prompt_tokens', 0),
                                        usage.get('completion_tokens', 0))
                except json.JSONDecodeError:
                    continue

    except asyncio.TimeoutError:
        inference_breaker.record_failure()
        _emit_event(job, 'error', {'message': f'Inference timed out after {INFERENCE_TIMEOUT}s'})
        clog.error('Inference timeout after %ds', INFERENCE_TIMEOUT)
        return None
    except aiohttp.ClientError as e:
        inference_breaker.record_failure()
        _emit_event(job, 'error', {'message': f'Connection error: {e}'})
        clog.error('Connection error: %s', e)
        return None
    except Exception as e:
        inference_breaker.record_failure()
        _emit_event(job, 'error', {'message': f'Unexpected error: {e}'})
        clog.error('Unexpected error: %s', e)
        return None

    return ''.join(full_response)


async def _process_chat_job(job_id: str, model: str, messages: list, temperature: float, max_tokens: int, cid: str):
    """Process inference with tool-calling loop. Max MAX_TOOL_ITERATIONS iterations."""
    job = jobs[job_id]
    clog = CorrelatedLogger(logger, cid, model=model)
    trace = TraceLogger(cid, model)
    start_time = time.monotonic()

    _emit_event(job, 'status', {'message': f'Sending to {model}...'})

    final_text = ''
    iteration = 0

    async with aiohttp.ClientSession() as session:
        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1

            trace.prompt(messages, iteration)
            response_text = await _call_inference(session, model, messages, temperature, max_tokens, job, clog)
            if response_text is None:
                trace.log('error', {'iteration': iteration, 'message': 'inference returned None'})
                job['done'] = True
                job['notify'].set()
                return

            trace.response(response_text, iteration)

            # Process action tags (memory)
            cleaned_text, actions = process_action_tags(response_text, clog)
            for action in actions:
                _emit_event(job, 'action', {'message': action})

            # Check for tool calls
            text_without_tools, tool_calls = ToolCallParser.parse(cleaned_text)

            if not tool_calls:
                # No tool calls: this is the final response
                final_text = cleaned_text
                break

            # Execute tool calls
            clog.info('Iteration %d: %d tool call(s) detected', iteration, len(tool_calls))

            # Add assistant message with tool calls to conversation context
            messages.append({'role': 'assistant', 'content': response_text})

            tool_results = []
            for tc in tool_calls:
                _emit_event(job, 'tool_call', {'name': tc.name, 'args': tc.args})
                clog.info('Executing tool: %s(%s)', tc.name, json.dumps(tc.args)[:200])
                trace.tool_call(tc.name, tc.args, iteration)

                result = await TOOL_REGISTRY.execute(tc.name, tc.args)

                result_text = json.dumps(result, default=str)[:2000]
                _emit_event(job, 'tool_result', {'name': tc.name, 'result': result})
                clog.info('Tool %s result: success=%s', tc.name, result.get('success', False))
                trace.tool_result(tc.name, result, iteration)

                tool_results.append(f'<tool_result name="{tc.name}">\n{result_text}\n</tool_result>')

            # Feed results back as a user message for next iteration
            results_block = '\n'.join(tool_results)
            messages.append({'role': 'user', 'content': results_block})

            # Clear content event so UI knows more is coming
            _emit_event(job, 'status', {'message': f'Processing tool results (iteration {iteration})...'})

        else:
            # Hit max iterations
            clog.warning('Hit max tool iterations (%d)', MAX_TOOL_ITERATIONS)
            _emit_event(job, 'status', {'message': 'Reached tool iteration limit'})
            final_text = text_without_tools if text_without_tools else 'I reached the maximum number of tool iterations. Here is what I found so far.'

    # Store final response in conversation
    if final_text:
        conversation.append({'role': 'assistant', 'content': final_text})
        while len(conversation) > CFG['max_history']:
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
        return web.json_response({
            **CFG,
            'system_prompt': get_effective_system_prompt(),
            'mode_name': mode['name'],
            'mode_temperature': mode['temperature'],
            'mode_max_tokens': mode['max_tokens'],
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
        return web.json_response({'error': 'Missing conversation id'}, status=400)

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
