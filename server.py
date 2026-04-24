#!/usr/bin/env python3
"""
Nemo Harness - Chat interface for NVIDIA Nemotron via OpenAI-compatible API

Proxies chat requests to a Nemotron inference server (LM Studio, Ollama, etc.)
and serves a ThinxS-style web frontend.

Usage:
    python server.py [--port 8090] [--host 0.0.0.0]

Environment variables:
    NEMO_API_BASE  - Nemotron API base URL (default: http://100.125.50.95:1237)
    NEMO_MODEL     - Model name (default: nvidia/nemotron-3-nano-4b)
    NEMO_PORT      - Server port (default: 8090)
"""

import argparse
import asyncio
import json
import logging
import os
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from aiohttp import web
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

logger = logging.getLogger('nemo-harness')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

# --- Config (mutable dict so handlers can update at runtime) ---
CFG = {
    'api_base': os.getenv('NEMO_API_BASE', 'http://100.125.50.95:1237'),
    'model': os.getenv('NEMO_MODEL', 'nvidia/nemotron-3-nano-4b'),
    'system_prompt': os.getenv('NEMO_SYSTEM_PROMPT', 'You are a helpful AI assistant powered by NVIDIA Nemotron.'),
    'max_history': int(os.getenv('NEMO_MAX_HISTORY', '50')),
}

# --- In-memory conversation history (single user) ---
conversation: list[dict] = []

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / 'web'
CONVERSATIONS_DIR = BASE_DIR / 'conversations'
EVALUATIONS_FILE = BASE_DIR / 'evaluations.jsonl'

# Ensure conversations directory exists
CONVERSATIONS_DIR.mkdir(exist_ok=True)


# ===== EXISTING ENDPOINTS =====

async def handle_status(request: web.Request) -> web.Response:
    """Health check and model info."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{CFG["api_base"]}/v1/models', timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    return web.json_response({'status': 'connected', 'model': CFG['model'], 'api_base': CFG['api_base']})
    except Exception:
        pass
    return web.json_response({'status': 'disconnected', 'model': CFG['model'], 'api_base': CFG['api_base']}, status=503)


async def handle_models(request: web.Request) -> web.Response:
    """Proxy model list from upstream."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{CFG["api_base"]}/v1/models', timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
                return web.json_response(data)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=502)


async def handle_history(request: web.Request) -> web.Response:
    """Return conversation history."""
    return web.json_response(conversation)


async def handle_clear(request: web.Request) -> web.Response:
    """Clear conversation history."""
    conversation.clear()
    return web.json_response({'status': 'cleared'})


async def handle_history_restore(request: web.Request) -> web.Response:
    """Restore conversation history from client (used when loading a saved conversation)."""
    body = await request.json()
    messages = body.get('messages', [])
    conversation.clear()
    for msg in messages:
        if msg.get('role') in ('user', 'assistant'):
            conversation.append({'role': msg['role'], 'content': msg.get('content', '')})
    # Trim to max_history
    while len(conversation) > CFG['max_history']:
        conversation.pop(0)
    return web.json_response({'status': 'restored', 'count': len(conversation)})


async def handle_chat(request: web.Request) -> web.Response:
    """Send a message and stream the response via SSE."""
    body = await request.json()
    user_message = body.get('message', '').strip()
    if not user_message:
        return web.json_response({'error': 'Empty message'}, status=400)

    # Allow model override per request
    model = body.get('model', CFG['model'])

    # If regenerating, remove the last assistant message and re-use the user message
    if body.get('regenerate'):
        # Remove the last assistant message from history if present
        if conversation and conversation[-1]['role'] == 'assistant':
            conversation.pop()
    else:
        conversation.append({'role': 'user', 'content': user_message})

    # Build messages for the API
    messages = [{'role': 'system', 'content': CFG['system_prompt']}]
    # Keep only the last max_history messages
    messages.extend(conversation[-CFG['max_history']:])

    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )
    await response.prepare(request)

    full_response = []
    reasoning_content = []

    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                'model': model,
                'messages': messages,
                'stream': True,
                'temperature': float(body.get('temperature', 0.7)),
                'max_tokens': int(body.get('max_tokens', 2048)),
            }

            async with session.post(
                f'{CFG["api_base"]}/v1/chat/completions',
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    event = json.dumps({'type': 'error', 'message': f'API error {resp.status}: {error_text}'})
                    await response.write(f'data: {event}\n\n'.encode())
                    await response.write(b'data: {"type":"done"}\n\n')
                    return response

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

                        # Handle reasoning content (chain-of-thought)
                        if delta.get('reasoning_content'):
                            reasoning_content.append(delta['reasoning_content'])
                            event = json.dumps({'type': 'reasoning', 'content': delta['reasoning_content']})
                            await response.write(f'data: {event}\n\n'.encode())

                        # Handle response content
                        if delta.get('content'):
                            full_response.append(delta['content'])
                            event = json.dumps({'type': 'content', 'content': delta['content']})
                            await response.write(f'data: {event}\n\n'.encode())

                        # Usage info in the final chunk
                        usage = chunk.get('usage')
                        if usage:
                            event = json.dumps({'type': 'usage', **usage})
                            await response.write(f'data: {event}\n\n'.encode())

                    except json.JSONDecodeError:
                        continue

    except asyncio.TimeoutError:
        event = json.dumps({'type': 'error', 'message': 'Request timed out'})
        await response.write(f'data: {event}\n\n'.encode())
    except Exception as e:
        event = json.dumps({'type': 'error', 'message': str(e)})
        await response.write(f'data: {event}\n\n'.encode())

    # Save assistant response to history
    assistant_msg = ''.join(full_response)
    if assistant_msg:
        conversation.append({'role': 'assistant', 'content': assistant_msg})
        # Trim history
        while len(conversation) > CFG['max_history']:
            conversation.pop(0)

    await response.write(b'data: {"type":"done"}\n\n')
    return response


async def handle_config(request: web.Request) -> web.Response:
    """Get/set runtime config."""
    if request.method == 'GET':
        return web.json_response(CFG)

    body = await request.json()
    for key in ('model', 'api_base', 'system_prompt'):
        if key in body:
            CFG[key] = body[key]
    if 'max_history' in body:
        CFG['max_history'] = int(body['max_history'])
    return web.json_response({'status': 'updated'})


# ===== CONVERSATION PERSISTENCE =====

async def handle_conversations_save(request: web.Request) -> web.Response:
    """Save or update a conversation."""
    body = await request.json()
    convo_id = body.get('id')
    if not convo_id:
        return web.json_response({'error': 'Missing conversation id'}, status=400)

    # Sanitize ID to prevent path traversal
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

    # Load existing to preserve created timestamp
    created = now
    if filepath.exists():
        try:
            existing = json.loads(filepath.read_text())
            created = existing.get('created', now)
        except Exception:
            pass

    data = {
        'id': safe_id,
        'title': title,
        'messages': messages,
        'created': created,
        'updated': now,
    }

    filepath.write_text(json.dumps(data, indent=2))
    return web.json_response({'status': 'saved', 'id': safe_id})


async def handle_conversations_list(request: web.Request) -> web.Response:
    """List all saved conversations."""
    convos = []
    for f in sorted(CONVERSATIONS_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            convos.append({
                'id': data.get('id', f.stem),
                'title': data.get('title', 'Untitled'),
                'created': data.get('created', ''),
                'updated': data.get('updated', ''),
                'message_count': len(data.get('messages', [])),
            })
        except Exception:
            continue
    return web.json_response(convos)


async def handle_conversation_get(request: web.Request) -> web.Response:
    """Load a specific conversation."""
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
    """Delete a conversation."""
    convo_id = request.match_info['id']
    safe_id = ''.join(c for c in convo_id if c.isalnum() or c in '-_')
    filepath = CONVERSATIONS_DIR / f'{safe_id}.json'

    if filepath.exists():
        filepath.unlink()
        return web.json_response({'status': 'deleted'})
    return web.json_response({'error': 'Not found'}, status=404)


# ===== EVALUATION ENDPOINT =====

async def handle_evaluate(request: web.Request) -> web.Response:
    """Accept thumbs up/down evaluations, append to evaluations.jsonl."""
    body = await request.json()
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'score': body.get('score', 0),
        'session_id': body.get('session_id', ''),
        'message_preview': body.get('message_preview', ''),
        'model': CFG['model'],
    }

    try:
        with open(EVALUATIONS_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        logger.error(f'Failed to write evaluation: {e}')
        return web.json_response({'error': str(e)}, status=500)

    return web.json_response({'status': 'recorded'})


# ===== APP SETUP =====

def create_app() -> web.Application:
    app = web.Application()

    # API routes - existing
    app.router.add_get('/api/status', handle_status)
    app.router.add_get('/api/models', handle_models)
    app.router.add_get('/api/history', handle_history)
    app.router.add_post('/api/history/restore', handle_history_restore)
    app.router.add_post('/api/clear', handle_clear)
    app.router.add_post('/api/chat', handle_chat)
    app.router.add_get('/api/config', handle_config)
    app.router.add_post('/api/config', handle_config)

    # API routes - new
    app.router.add_post('/api/conversations', handle_conversations_save)
    app.router.add_get('/api/conversations', handle_conversations_list)
    app.router.add_get('/api/conversations/{id}', handle_conversation_get)
    app.router.add_delete('/api/conversations/{id}', handle_conversation_delete)
    app.router.add_post('/api/evaluate', handle_evaluate)

    # Static files
    app.router.add_get('/', lambda r: web.FileResponse(WEB_DIR / 'index.html'))
    app.router.add_get('/{filename}', serve_web_file)

    return app


async def serve_web_file(request: web.Request) -> web.Response:
    filename = request.match_info['filename']
    filepath = WEB_DIR / filename
    if filepath.exists() and filepath.is_file():
        return web.FileResponse(filepath)
    return web.FileResponse(WEB_DIR / 'index.html')


async def run_server():
    parser = argparse.ArgumentParser(description='Nemo Harness - Nemotron Chat Server')
    parser.add_argument('--port', type=int, default=int(os.getenv('NEMO_PORT', '8090')))
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
        logger.info("TLS enabled via Tailscale certs")

    site = web.TCPSite(runner, args.host, args.port, ssl_context=ssl_ctx)
    await site.start()

    proto = 'https' if ssl_ctx else 'http'
    logger.info(f'Nemo Harness running at {proto}://{args.host}:{args.port}')
    logger.info(f'Model: {CFG["model"]} @ {CFG["api_base"]}')

    while True:
        await asyncio.sleep(3600)


if __name__ == '__main__':
    asyncio.run(run_server())
