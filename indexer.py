"""Gitea webhook indexer for the RAG system.

Receives push events from Gitea and indexes changed files into ChromaDB.
Runs as a lightweight aiohttp server alongside the main harness.

Usage:
    python indexer.py [--port 9090]

Or import and mount as a sub-application in server.py.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from urllib.parse import quote

import aiohttp
from aiohttp import web

from tools.rag_tools import index_content, delete_source

logger = logging.getLogger('nemo.indexer')

GITEA_URL = os.getenv('GITEA_URL', 'http://localhost:3000')
GITEA_TOKEN = os.getenv('GITEA_TOKEN', '')
GITEA_OWNER = os.getenv('GITEA_OWNER', 'nemo')
WEBHOOK_SECRET = os.getenv('GITEA_WEBHOOK_SECRET', '')

# File extensions worth indexing
INDEXABLE_EXTENSIONS = {
    '.py', '.js', '.ts', '.go', '.rs', '.java', '.c', '.cpp', '.h',
    '.md', '.txt', '.rst', '.yaml', '.yml', '.json', '.toml',
    '.sh', '.bash', '.dockerfile', '.sql',
    '.html', '.css', '.scss',
}

MAX_FILE_SIZE = 100 * 1024  # 100KB per file


def _should_index(path: str) -> bool:
    """Decide if a file path should be indexed."""
    p = Path(path)
    if p.suffix.lower() not in INDEXABLE_EXTENSIONS:
        return False
    # Skip vendored/generated content
    skip_dirs = {'node_modules', 'vendor', '.git', '__pycache__', 'dist', 'build', '.venv'}
    return not any(part in skip_dirs for part in p.parts)


async def _fetch_file_content(repo: str, path: str, ref: str = 'main') -> str | None:
    """Fetch a file's content from Gitea API."""
    url = f'{GITEA_URL}/api/v1/repos/{GITEA_OWNER}/{repo}/contents/{quote(path, safe="/")}'
    headers = {}
    if GITEA_TOKEN:
        headers['Authorization'] = f'token {GITEA_TOKEN}'

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params={'ref': ref},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                content = data.get('content', '')
                encoding = data.get('encoding', 'base64')
                if encoding == 'base64':
                    return base64.b64decode(content).decode('utf-8', errors='replace')
                return content
    except Exception as e:
        logger.warning(f'Failed to fetch {repo}/{path}: {e}')
        return None


async def handle_webhook(request: web.Request) -> web.Response:
    """Handle Gitea push webhook events."""
    # Verify secret if configured
    if WEBHOOK_SECRET:
        sig = request.headers.get('X-Gitea-Signature', '')
        if not sig:
            return web.json_response({'error': 'Missing signature'}, status=401)
        import hmac
        import hashlib
        body = await request.read()
        expected = hmac.new(
            WEBHOOK_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return web.json_response({'error': 'Invalid signature'}, status=401)
        payload = json.loads(body)
    else:
        payload = await request.json()

    repo_name = payload.get('repository', {}).get('name', '')
    ref = payload.get('ref', 'refs/heads/main')
    commits = payload.get('commits', [])

    if not repo_name or not commits:
        return web.json_response({'status': 'ignored', 'reason': 'no commits'})

    # Collect all changed files
    added = set()
    modified = set()
    removed = set()
    for commit in commits:
        added.update(commit.get('added', []))
        modified.update(commit.get('modified', []))
        removed.update(commit.get('removed', []))

    # Remove deleted files from index
    for path in removed:
        source = f'{repo_name}/{path}'
        delete_source(source)

    # Index added/modified files
    indexed = 0
    branch = ref.split('/')[-1] if '/' in ref else ref
    for path in added | modified:
        if not _should_index(path):
            continue
        content = await _fetch_file_content(repo_name, path, ref=branch)
        if content and len(content) <= MAX_FILE_SIZE:
            source = f'{repo_name}/{path}'
            count = index_content(content, source, repo=repo_name, metadata={
                'branch': branch,
                'file_type': Path(path).suffix,
            })
            indexed += count

    logger.info(f'Webhook: {repo_name} push — '
                f'{len(added)} added, {len(modified)} modified, {len(removed)} removed — '
                f'{indexed} chunks indexed')

    return web.json_response({
        'status': 'indexed',
        'repo': repo_name,
        'files_processed': len(added | modified),
        'files_removed': len(removed),
        'chunks_indexed': indexed,
    })


async def handle_reindex(request: web.Request) -> web.Response:
    """Trigger a full reindex of a repository."""
    data = await request.json()
    repo = data.get('repo', '').strip()
    if not repo:
        return web.json_response({'error': 'repo is required'}, status=400)

    # List all files via Gitea API tree endpoint
    url = f'{GITEA_URL}/api/v1/repos/{GITEA_OWNER}/{repo}/git/trees/HEAD'
    headers = {}
    if GITEA_TOKEN:
        headers['Authorization'] = f'token {GITEA_TOKEN}'

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params={'recursive': 'true'},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    return web.json_response({'error': f'Gitea API {resp.status}'}, status=502)
                tree_data = await resp.json()
    except Exception as e:
        return web.json_response({'error': str(e)}, status=502)

    entries = tree_data.get('tree', [])
    file_paths = [e['path'] for e in entries
                  if e.get('type') == 'blob' and _should_index(e['path'])]

    total_chunks = 0
    for path in file_paths:
        content = await _fetch_file_content(repo, path)
        if content and len(content) <= MAX_FILE_SIZE:
            source = f'{repo}/{path}'
            count = index_content(content, source, repo=repo, metadata={
                'file_type': Path(path).suffix,
            })
            total_chunks += count

    logger.info(f'Reindex: {repo} — {len(file_paths)} files, {total_chunks} chunks')
    return web.json_response({
        'status': 'reindexed',
        'repo': repo,
        'files': len(file_paths),
        'chunks': total_chunks,
    })


def create_indexer_app() -> web.Application:
    """Create the indexer web application (can be mounted as sub-app)."""
    app = web.Application()
    app.router.add_post('/webhook', handle_webhook)
    app.router.add_post('/reindex', handle_reindex)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gitea webhook indexer')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    app = create_indexer_app()
    logger.info(f'Indexer starting on {args.host}:{args.port}')
    web.run_app(app, host=args.host, port=args.port)
