"""Gitea API tools for the Nemo tool-calling system.

Provides repo management, issue tracking, and file operations
against a local Gitea instance. Fully self-contained for air-gapped use.
"""

from __future__ import annotations

import base64
import os
from urllib.parse import quote

import aiohttp

from tools.registry import ToolDef, TOOL_REGISTRY

GITEA_URL = os.getenv('GITEA_URL', 'http://localhost:3000')
GITEA_TOKEN = os.getenv('GITEA_TOKEN', '')
GITEA_OWNER = os.getenv('GITEA_OWNER', 'nemo')

API_TIMEOUT = aiohttp.ClientTimeout(total=15)


def _headers():
    h = {'Content-Type': 'application/json'}
    if GITEA_TOKEN:
        h['Authorization'] = f'token {GITEA_TOKEN}'
    return h


async def _api(method: str, path: str, json_data: dict | None = None) -> dict:
    """Make a Gitea API call."""
    url = f'{GITEA_URL}/api/v1{path}'
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method, url, json=json_data,
                headers=_headers(), timeout=API_TIMEOUT,
            ) as resp:
                body = await resp.json() if resp.content_type == 'application/json' else {}
                if resp.status >= 400:
                    msg = body.get('message', '') if isinstance(body, dict) else str(body)
                    return {'error': f'Gitea API {resp.status}: {msg}'}
                return body
    except aiohttp.ClientError as e:
        return {'error': f'Gitea connection failed: {e}'}


async def _create_repo(name: str, description: str = '', private: str = 'false') -> dict:
    """Create a new repository on Gitea."""
    name = name.strip()
    if not name:
        return {'error': 'Repository name is required'}
    result = await _api('POST', '/user/repos', {
        'name': name,
        'description': description,
        'private': private.lower() == 'true',
        'auto_init': True,
    })
    if 'error' in result:
        return result
    return {
        'repo': result.get('full_name', name),
        'url': result.get('html_url', ''),
        'clone_url': result.get('clone_url', ''),
        'message': f'Repository "{name}" created',
    }


async def _commit_file(repo: str, path: str, content: str, message: str = '') -> dict:
    """Create or update a file in a Gitea repository."""
    repo = repo.strip()
    path = path.strip().lstrip('/')
    if not repo or not path:
        return {'error': 'Both repo and path are required'}
    if not message:
        message = f'Update {path}'

    encoded = base64.b64encode(content.encode()).decode()
    api_path = f'/repos/{GITEA_OWNER}/{repo}/contents/{quote(path, safe="/")}'

    # Check if file exists (need SHA for updates)
    existing = await _api('GET', api_path)
    sha = existing.get('sha') if isinstance(existing, dict) and 'sha' in existing else None

    payload = {'content': encoded, 'message': message}
    if sha:
        payload['sha'] = sha

    method = 'PUT'
    result = await _api(method, api_path, payload)
    if 'error' in result:
        return result

    action = 'updated' if sha else 'created'
    return {'message': f'File {path} {action} in {repo}', 'path': path, 'action': action}


async def _create_issue(repo: str, title: str, body: str = '') -> dict:
    """Create an issue in a Gitea repository."""
    repo = repo.strip()
    title = title.strip()
    if not repo or not title:
        return {'error': 'Both repo and title are required'}

    result = await _api('POST', f'/repos/{GITEA_OWNER}/{repo}/issues', {
        'title': title,
        'body': body,
    })
    if 'error' in result:
        return result
    return {
        'number': result.get('number'),
        'title': title,
        'url': result.get('html_url', ''),
        'message': f'Issue #{result.get("number")} created in {repo}',
    }


async def _list_issues(repo: str, state: str = 'open') -> dict:
    """List issues in a Gitea repository."""
    repo = repo.strip()
    if not repo:
        return {'error': 'Repository name is required'}

    state = state.strip().lower()
    if state not in ('open', 'closed', 'all'):
        state = 'open'

    result = await _api('GET', f'/repos/{GITEA_OWNER}/{repo}/issues?state={state}&limit=20')
    if isinstance(result, dict) and 'error' in result:
        return result

    issues = []
    for issue in (result if isinstance(result, list) else []):
        issues.append({
            'number': issue.get('number'),
            'title': issue.get('title', ''),
            'state': issue.get('state', ''),
            'labels': [l.get('name', '') for l in issue.get('labels', [])],
            'created': issue.get('created_at', ''),
        })
    return {'repo': repo, 'state': state, 'issues': issues, 'count': len(issues)}


async def _search_repos(query: str = '') -> dict:
    """Search/list repositories on Gitea."""
    params = f'?limit=20'
    if query.strip():
        params += f'&q={quote(query.strip())}'

    result = await _api('GET', f'/repos/search{params}')
    if isinstance(result, dict) and 'error' in result:
        return result

    # Gitea search returns {data: [...], ok: true}
    repos_data = result.get('data', []) if isinstance(result, dict) else []
    repos = []
    for r in repos_data:
        repos.append({
            'name': r.get('name', ''),
            'full_name': r.get('full_name', ''),
            'description': r.get('description', ''),
            'url': r.get('html_url', ''),
            'stars': r.get('stars_count', 0),
            'updated': r.get('updated_at', ''),
        })
    return {'query': query, 'repos': repos, 'count': len(repos)}


def register():
    """Register Gitea tools with the global registry."""

    TOOL_REGISTRY.register(ToolDef(
        name='create_repo',
        description='Create a new Git repository on the local Gitea server. Use this to start a new project.',
        parameters={
            'name': {'type': 'string', 'description': 'Repository name (alphanumeric, hyphens, underscores)'},
            'description': {'type': 'string', 'description': 'Short repository description'},
            'private': {'type': 'string', 'description': '"true" or "false" (default: false)'},
        },
        required=['name'],
        handler=_create_repo,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='commit_file',
        description='Create or update a file in a Gitea repository. Automatically handles create vs update.',
        parameters={
            'repo': {'type': 'string', 'description': 'Repository name'},
            'path': {'type': 'string', 'description': 'File path within the repo (e.g., "src/main.py")'},
            'content': {'type': 'string', 'description': 'File content to write'},
            'message': {'type': 'string', 'description': 'Commit message (auto-generated if empty)'},
        },
        required=['repo', 'path', 'content'],
        handler=_commit_file,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='create_issue',
        description='Create an issue in a Gitea repository to track work, bugs, or features.',
        parameters={
            'repo': {'type': 'string', 'description': 'Repository name'},
            'title': {'type': 'string', 'description': 'Issue title'},
            'body': {'type': 'string', 'description': 'Issue description/body (markdown)'},
        },
        required=['repo', 'title'],
        handler=_create_issue,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='list_issues',
        description='List issues in a Gitea repository. Filter by state (open/closed/all).',
        parameters={
            'repo': {'type': 'string', 'description': 'Repository name'},
            'state': {'type': 'string', 'description': 'Filter: "open", "closed", or "all" (default: open)'},
        },
        required=['repo'],
        handler=_list_issues,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='search_repos',
        description='Search for repositories on the local Gitea server. Lists all repos if no query given.',
        parameters={
            'query': {'type': 'string', 'description': 'Search query (empty to list all repos)'},
        },
        required=[],
        handler=_search_repos,
    ))
