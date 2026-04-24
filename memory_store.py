"""
Memory Store for Nemo Harness

SQLite-backed persistent memory with typed entries (user, feedback, project, reference).
Provides context injection for system prompts and CRUD operations.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger('nemo-harness.memory')

DB_PATH = Path(__file__).parent / 'memory' / 'memory.db'

_VALID_TYPES = ('user', 'feedback', 'project', 'reference')
_TYPE_MAP = {
    'user': 'user', 'feedback': 'feedback',
    'project': 'project', 'projects': 'project',
    'reference': 'reference', 'references': 'reference',
    'account': 'reference', 'infrastructure': 'reference',
    'lesson': 'feedback', 'lessons': 'feedback',
}


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def init_db():
    """Create the memory table if it doesn't exist."""
    conn = _get_conn()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memory_entries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                type        TEXT NOT NULL CHECK(type IN ('user','feedback','project','reference')),
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_type ON memory_entries(type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_name ON memory_entries(name)')
        conn.commit()
    finally:
        conn.close()


def upsert_entry(entry_type: str, name: str, description: str, content: str) -> dict:
    """Insert or update a memory entry. Returns the entry dict."""
    entry_type = _TYPE_MAP.get(entry_type.lower(), entry_type.lower())
    if entry_type not in _VALID_TYPES:
        raise ValueError(f'Invalid memory type: {entry_type}')

    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        existing = conn.execute(
            'SELECT id, created_at FROM memory_entries WHERE name = ?', (name,)
        ).fetchone()

        if existing:
            conn.execute(
                'UPDATE memory_entries SET type=?, description=?, content=?, updated_at=? WHERE id=?',
                (entry_type, description, content, now, existing['id'])
            )
            entry_id = existing['id']
            created = existing['created_at']
        else:
            cursor = conn.execute(
                'INSERT INTO memory_entries (type, name, description, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)',
                (entry_type, name, description, content, now, now)
            )
            entry_id = cursor.lastrowid
            created = now

        conn.commit()
        return {'id': entry_id, 'type': entry_type, 'name': name, 'description': description,
                'content': content, 'created_at': created, 'updated_at': now}
    finally:
        conn.close()


def get_entries(entry_type: str = None) -> list[dict]:
    """Get all entries, optionally filtered by type."""
    conn = _get_conn()
    try:
        if entry_type:
            entry_type = _TYPE_MAP.get(entry_type.lower(), entry_type.lower())
            rows = conn.execute(
                'SELECT * FROM memory_entries WHERE type = ? ORDER BY updated_at DESC', (entry_type,)
            ).fetchall()
        else:
            rows = conn.execute('SELECT * FROM memory_entries ORDER BY type, updated_at DESC').fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def search_entries(query: str) -> list[dict]:
    """Search entries by name or content (case-insensitive)."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            'SELECT * FROM memory_entries WHERE name LIKE ? OR content LIKE ? ORDER BY updated_at DESC',
            (f'%{query}%', f'%{query}%')
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_entry(entry_id: int) -> bool:
    conn = _get_conn()
    try:
        cursor = conn.execute('DELETE FROM memory_entries WHERE id = ?', (entry_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def build_context_block(max_chars_per_entry: int = 400) -> str:
    """Build a memory context block for system prompt injection."""
    entries = get_entries()
    if not entries:
        return ''

    lines = []
    for e in entries:
        tag = e['type'].upper()
        content = e['content'][:max_chars_per_entry]
        lines.append(f'[{tag}] {e["name"]}: {content}')

    return '\n'.join(lines)


def db_health_check() -> dict:
    """Check database health."""
    try:
        size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        size_mb = size / (1024 * 1024)
        conn = _get_conn()
        try:
            count = conn.execute('SELECT COUNT(*) FROM memory_entries').fetchone()[0]
        finally:
            conn.close()

        status = 'OK' if size_mb < 10 else ('WARN' if size_mb < 50 else 'CRITICAL')
        return {'size_bytes': size, 'size_mb': round(size_mb, 2), 'entry_count': count, 'status': status}
    except Exception as e:
        return {'size_bytes': 0, 'size_mb': 0, 'entry_count': 0, 'status': f'ERROR: {e}'}


# Initialize on import
init_db()
