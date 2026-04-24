"""Search tools: glob and grep."""

from __future__ import annotations
import fnmatch
import re
from pathlib import Path

from tools.sandbox import Sandbox
from tools.registry import ToolDef, TOOL_REGISTRY

MAX_RESULTS = 100


def register(sandbox: Sandbox):
    """Register search tools with the global registry."""

    async def glob_search(pattern: str, path: str = '.') -> dict:
        resolved = sandbox.check_path(path)
        if not resolved.is_dir():
            return {'error': f'Not a directory: {path}'}
        matches = []
        for item in resolved.rglob(pattern):
            if len(matches) >= MAX_RESULTS:
                break
            try:
                sandbox.check_path(str(item))
                matches.append(str(item))
            except PermissionError:
                continue
        return {'pattern': pattern, 'matches': matches, 'count': len(matches)}

    async def grep_search(query: str, path: str = '.', file_pattern: str = '*') -> dict:
        resolved = sandbox.check_path(path)
        if not resolved.is_dir():
            return {'error': f'Not a directory: {path}'}

        try:
            regex = re.compile(query, re.IGNORECASE)
        except re.error as e:
            return {'error': f'Invalid regex: {e}'}

        results = []
        files_searched = 0
        for fpath in resolved.rglob(file_pattern):
            if not fpath.is_file():
                continue
            try:
                sandbox.check_path(str(fpath))
            except PermissionError:
                continue

            files_searched += 1
            try:
                text = fpath.read_text(errors='replace')
            except Exception:
                continue

            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append({
                        'file': str(fpath),
                        'line': i,
                        'content': line.strip()[:200],
                    })
                    if len(results) >= MAX_RESULTS:
                        return {
                            'query': query,
                            'results': results,
                            'files_searched': files_searched,
                            'truncated': True,
                        }

        return {
            'query': query,
            'results': results,
            'files_searched': files_searched,
            'truncated': False,
        }

    TOOL_REGISTRY.register(ToolDef(
        name='glob_search',
        description='Find files matching a glob pattern recursively',
        parameters={
            'pattern': {'type': 'string', 'description': 'Glob pattern (e.g., "*.py", "**/*.json")'},
            'path': {'type': 'string', 'description': 'Directory to search in'},
        },
        required=['pattern'],
        handler=glob_search,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='grep_search',
        description='Search file contents for a regex pattern',
        parameters={
            'query': {'type': 'string', 'description': 'Regex pattern to search for'},
            'path': {'type': 'string', 'description': 'Directory to search in'},
            'file_pattern': {'type': 'string', 'description': 'File glob filter (e.g., "*.py")'},
        },
        required=['query'],
        handler=grep_search,
    ))
