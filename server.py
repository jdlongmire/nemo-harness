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
import json
import logging
import os
import re
import secrets
import ssl
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from aiohttp import web
from dotenv import load_dotenv

import memory_store

load_dotenv(Path(__file__).parent / '.env')

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


# --- Behavioral Mode System ---
DEFAULT_MODEL = 'nvidia/nemotron-3-nano-4b'

MODES = {
    'default': {
        'name': 'Default',
        'model': DEFAULT_MODEL,
        'system_prompt': (
            'You are a helpful AI assistant powered by NVIDIA Nemotron. '
            'You provide clear, accurate, and concise responses. '
            'When uncertain, say so rather than guessing.'
        ),
        'temperature': 0.7,
        'max_tokens': 2048,
    },
    'technical': {
        'name': 'Technical',
        'model': DEFAULT_MODEL,
        'system_prompt': (
            'You are a precise technical assistant powered by NVIDIA Nemotron. '
            'Prioritize accuracy over brevity. Include code examples when relevant. '
            'Use structured formatting (headers, lists, code blocks). '
            'When uncertain, say so. Distinguish between established facts and inferences.'
        ),
        'temperature': 0.3,
        'max_tokens': 4096,
    },
    'creative': {
        'name': 'Creative',
        'model': DEFAULT_MODEL,
        'system_prompt': (
            'You are a creative writing assistant powered by NVIDIA Nemotron. '
            'Write with vivid language, varied sentence structure, and narrative flow. '
            'Take creative risks. Explore ideas from unexpected angles.'
        ),
        'temperature': 1.0,
        'max_tokens': 4096,
    },
    'research': {
        'name': 'Research',
        'model': DEFAULT_MODEL,
        'system_prompt': (
            'You are a research assistant powered by NVIDIA Nemotron. '
            'Analyze claims carefully. Distinguish evidence from inference. '
            'Cite reasoning steps explicitly. Flag assumptions. '
            'When you do not know something, say so clearly rather than speculating.'
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


def get_effective_system_prompt() -> str:
    """Build system prompt from mode + memory context."""
    if CFG['custom_system_prompt']:
        base_prompt = CFG['custom_system_prompt']
    else:
        mode = MODES.get(CFG['mode'], MODES['default'])
        base_prompt = mode['system_prompt']

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


async def _process_chat_job(job_id: str, model: str, messages: list, temperature: float, max_tokens: int, cid: str):
    """Process inference request and emit SSE events to job buffer."""
    job = jobs[job_id]
    clog = CorrelatedLogger(logger, cid, model=model)
    start_time = time.monotonic()

    _emit_event(job, 'status', {'message': f'Sending to {model}...'})

    full_response = []
    reasoning_content = []

    try:
        async with aiohttp.ClientSession() as session:
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
                    job['done'] = True
                    job['notify'].set()
                    clog.error('API error %d: %s', resp.status, error_text[:200])
                    return

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
                            reasoning_content.append(delta['reasoning_content'])
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
                            log_token_usage(cid, model,
                                            usage.get('prompt_tokens', 0),
                                            usage.get('completion_tokens', 0))

                    except json.JSONDecodeError:
                        continue

    except asyncio.TimeoutError:
        inference_breaker.record_failure()
        _emit_event(job, 'error', {'message': f'Inference timed out after {INFERENCE_TIMEOUT}s'})
        clog.error('Inference timeout after %ds', INFERENCE_TIMEOUT)
    except aiohttp.ClientError as e:
        inference_breaker.record_failure()
        _emit_event(job, 'error', {'message': f'Connection error: {e}'})
        clog.error('Connection error: %s', e)
    except Exception as e:
        inference_breaker.record_failure()
        _emit_event(job, 'error', {'message': f'Unexpected error: {e}'})
        clog.error('Unexpected error: %s', e)

    # Process response
    assistant_msg = ''.join(full_response)
    if assistant_msg:
        # Process action tags (e.g., [ACTION:remember|...])
        cleaned_msg, actions = process_action_tags(assistant_msg, clog)

        # Surface any action results
        for action in actions:
            _emit_event(job, 'action', {'message': action})

        conversation.append({'role': 'assistant', 'content': cleaned_msg})
        while len(conversation) > CFG['max_history']:
            conversation.pop(0)

    elapsed = time.monotonic() - start_time
    clog.info('Job %s completed in %.1fs (%d chars)', job_id, elapsed, len(assistant_msg))

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
