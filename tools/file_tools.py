"""File system tools: read, write, edit."""

from __future__ import annotations
from pathlib import Path

from tools.sandbox import Sandbox
from tools.registry import ToolDef, TOOL_REGISTRY


def register(sandbox: Sandbox):
    """Register file tools with the global registry."""

    async def read_file(file_path: str, offset: int = 0, limit: int = 100) -> dict:
        resolved = sandbox.check_path(file_path)
        if not resolved.exists():
            return {'error': f'File not found: {file_path}'}
        if resolved.is_dir():
            return {'error': f'Path is a directory: {file_path}'}
        text = resolved.read_text(errors='replace')
        lines = text.splitlines()
        total = len(lines)
        chunk = lines[offset:offset + limit]
        numbered = [f'{i + offset + 1}: {line}' for i, line in enumerate(chunk)]
        return {
            'content': '\n'.join(numbered),
            'total_lines': total,
            'showing': f'{offset + 1}-{min(offset + limit, total)}',
            'success': True,
        }

    async def write_file(file_path: str, content: str) -> dict:
        resolved = sandbox.check_path(file_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return {'message': f'Wrote {len(content)} chars to {file_path}'}

    async def edit_file(file_path: str, old_string: str, new_string: str) -> dict:
        resolved = sandbox.check_path(file_path)
        if not resolved.exists():
            return {'error': f'File not found: {file_path}'}
        text = resolved.read_text()
        count = text.count(old_string)
        if count == 0:
            return {'error': 'old_string not found in file'}
        if count > 1:
            return {'error': f'old_string matches {count} locations; provide more context to make it unique'}
        text = text.replace(old_string, new_string, 1)
        resolved.write_text(text)
        return {'message': f'Edited {file_path}: replaced 1 occurrence'}

    async def list_dir(path: str = '.') -> dict:
        resolved = sandbox.check_path(path)
        if not resolved.is_dir():
            return {'error': f'Not a directory: {path}'}
        entries = []
        for item in sorted(resolved.iterdir()):
            kind = 'dir' if item.is_dir() else 'file'
            size = item.stat().st_size if item.is_file() else 0
            entries.append({'name': item.name, 'type': kind, 'size': size})
        return {'path': str(resolved), 'entries': entries}

    TOOL_REGISTRY.register(ToolDef(
        name='read_file',
        description='Read a file and return its contents with line numbers. Returns max 100 lines by default; use offset/limit to paginate larger files.',
        parameters={
            'file_path': {'type': 'string', 'description': 'Path to file'},
            'offset': {'type': 'integer', 'description': 'Starting line (0-based)'},
            'limit': {'type': 'integer', 'description': 'Max lines to return'},
        },
        required=['file_path'],
        handler=read_file,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='write_file',
        description='Create or overwrite a file with given content',
        parameters={
            'file_path': {'type': 'string', 'description': 'Path to file'},
            'content': {'type': 'string', 'description': 'File content to write'},
        },
        handler=write_file,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='edit_file',
        description='Replace a unique string in a file with a new string',
        parameters={
            'file_path': {'type': 'string', 'description': 'Path to file'},
            'old_string': {'type': 'string', 'description': 'Exact text to find (must be unique)'},
            'new_string': {'type': 'string', 'description': 'Replacement text'},
        },
        handler=edit_file,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='list_dir',
        description='List files and directories at a given path',
        parameters={
            'path': {'type': 'string', 'description': 'Directory path to list'},
        },
        handler=list_dir,
    ))
