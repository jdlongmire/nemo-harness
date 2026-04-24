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
import time
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

WEB_DIR = Path(__file__).parent / 'web'


async def handle_status(request: web.Request) -> web.Response:
    """Health check and model info."""
    # Quick probe of the upstream API
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


async def handle_chat(request: web.Request) -> web.Response:
    """Send a message and stream the response via SSE."""
    body = await request.json()
    user_message = body.get('message', '').strip()
    if not user_message:
        return web.json_response({'error': 'Empty message'}, status=400)

    # Allow model override per request
    model = body.get('model', CFG['model'])

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


def create_app() -> web.Application:
    app = web.Application()

    # API routes
    app.router.add_get('/api/status', handle_status)
    app.router.add_get('/api/models', handle_models)
    app.router.add_get('/api/history', handle_history)
    app.router.add_post('/api/clear', handle_clear)
    app.router.add_post('/api/chat', handle_chat)
    app.router.add_get('/api/config', handle_config)
    app.router.add_post('/api/config', handle_config)

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


def main():
    parser = argparse.ArgumentParser(description='Nemo Harness - Nemotron Chat Server')
    parser.add_argument('--port', type=int, default=int(os.getenv('NEMO_PORT', '8090')))
    parser.add_argument('--host', default=os.getenv('NEMO_HOST', '0.0.0.0'))
    args = parser.parse_args()

    app = create_app()
    logger.info(f'Starting Nemo Harness on {args.host}:{args.port}')
    logger.info(f'Model: {CFG["model"]} @ {CFG["api_base"]}')
    web.run_app(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
