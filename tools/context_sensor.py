"""Lightweight intent classifier for dynamic guide and tool selection.

Classifies recent user messages into broad intent categories using keyword
matching (Tier 1 deterministic). No LLM cost, sub-millisecond latency.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# --- Intent definitions ---

INTENTS = ('coding', 'document', 'research', 'planning', 'git', 'conversation')

# Keywords that signal each intent.  Checked against lowercased user text.
_INTENT_KEYWORDS: dict[str, list[str]] = {
    'coding': [
        'code', 'function', 'class', 'bug', 'error', 'fix', 'implement', 'refactor',
        'debug', 'compile', 'syntax', 'variable', 'import', 'module', 'api',
        'endpoint', 'test', 'unittest', 'pytest', 'script', 'run_command',
        'python', 'javascript', 'typescript', 'rust', 'bash', 'shell',
        'read_file', 'write_file', 'edit_file', 'grep', 'glob',
    ],
    'document': [
        'document', 'docx', 'pptx', 'xlsx', 'spreadsheet', 'presentation',
        'slide', 'report', 'pdf', 'svg', 'diagram', 'chart', 'table',
        'create_docx', 'create_pptx', 'create_xlsx', 'create_svg', 'edit_svg',
        'letter', 'memo', 'template', 'format', 'design',
    ],
    'research': [
        'research', 'search', 'find', 'look up', 'investigate', 'analyze',
        'compare', 'review', 'study', 'evidence', 'source', 'citation',
        'web_search', 'web_fetch', 'semantic_search', 'article', 'paper',
        'what is', 'how does', 'explain', 'summarize',
    ],
    'planning': [
        'plan', 'task', 'step', 'roadmap', 'milestone', 'schedule',
        'prioritize', 'organize', 'break down', 'todo', 'checklist',
        'create_plan', 'get_plan', 'update_task', 'add_task', 'list_plans',
        'strategy', 'approach', 'scope',
    ],
    'git': [
        'git', 'commit', 'branch', 'merge', 'pull request', 'push',
        'repo', 'repository', 'issue', 'clone', 'diff', 'log',
        'create_repo', 'commit_file', 'create_issue', 'list_issues', 'search_repos',
        'gitea', 'github',
    ],
}

# Compiled regex patterns for each intent (word-boundary matching)
_INTENT_PATTERNS: dict[str, re.Pattern] = {
    intent: re.compile(
        r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b',
        re.IGNORECASE,
    )
    for intent, keywords in _INTENT_KEYWORDS.items()
}

# Tools relevant to each intent
INTENT_TOOLS: dict[str, set[str]] = {
    'coding': {
        'read_file', 'write_file', 'edit_file', 'list_dir', 'run_command',
        'glob_search', 'grep_search',
    },
    'document': {
        'read_file', 'write_file', 'list_dir',
        'create_docx', 'create_xlsx', 'create_pptx', 'create_svg', 'edit_svg',
    },
    'research': {
        'read_file', 'list_dir',
        'web_search', 'web_fetch', 'semantic_search', 'index_text', 'rag_status',
        'glob_search', 'grep_search',
    },
    'planning': {
        'read_file', 'list_dir',
        'create_plan', 'get_plan', 'update_task', 'add_task', 'list_plans',
    },
    'git': {
        'read_file', 'list_dir', 'run_command',
        'create_repo', 'commit_file', 'create_issue', 'list_issues', 'search_repos',
        'glob_search', 'grep_search',
    },
    'conversation': set(),  # empty = use all tools (fallback)
}

# Behavioral guide keys relevant to each intent
INTENT_GUIDES: dict[str, list[str]] = {
    'coding': ['behavioral_core', 'memory', 'clarification'],
    'document': ['behavioral_core', 'design', 'memory', 'clarification'],
    'research': ['behavioral_core', 'memory', 'clarification'],
    'planning': ['behavioral_core', 'planning', 'memory', 'clarification'],
    'git': ['behavioral_core', 'memory'],
    'conversation': ['behavioral_core', 'memory', 'clarification', 'design', 'planning'],
}

# Hysteresis: require this many consecutive signals before switching
HYSTERESIS_THRESHOLD = 2


@dataclass
class ContextState:
    """Tracks intent history for hysteresis."""
    history: list[str] = field(default_factory=list)
    current_intent: str = 'conversation'

    def update(self, detected: str) -> str:
        """Push a detected intent. Returns the effective intent after hysteresis."""
        self.history.append(detected)
        # Keep only last 5
        if len(self.history) > 5:
            self.history = self.history[-5:]

        # If same as current, no change needed
        if detected == self.current_intent:
            return self.current_intent

        # Check if last N detections agree on the new intent
        recent = self.history[-HYSTERESIS_THRESHOLD:]
        if len(recent) >= HYSTERESIS_THRESHOLD and all(r == detected for r in recent):
            self.current_intent = detected

        return self.current_intent


def classify_message(text: str) -> tuple[str, dict[str, int]]:
    """Classify a single message into an intent category.

    Returns (best_intent, scores_dict).
    """
    scores: dict[str, int] = {}
    text_lower = text.lower()

    for intent, pattern in _INTENT_PATTERNS.items():
        matches = pattern.findall(text_lower)
        scores[intent] = len(matches)

    if not any(scores.values()):
        return 'conversation', scores

    best = max(scores, key=lambda k: scores[k])
    return best, scores


def classify_recent(messages: list[dict], window: int = 3) -> tuple[str, dict[str, int]]:
    """Classify the intent from the last `window` user messages.

    Returns (best_intent, aggregate_scores).
    """
    user_msgs = [m['content'] for m in messages if m.get('role') == 'user' and isinstance(m.get('content'), str)]
    recent = user_msgs[-window:]

    if not recent:
        return 'conversation', {}

    aggregate: Counter[str] = Counter()
    for msg in recent:
        _, scores = classify_message(msg)
        aggregate.update(scores)

    if not any(aggregate.values()):
        return 'conversation', dict(aggregate)

    best = max(aggregate, key=lambda k: aggregate[k])
    return best, dict(aggregate)


def get_relevant_tools(intent: str) -> set[str] | None:
    """Return tool names relevant to the intent, or None for 'use all'."""
    tools = INTENT_TOOLS.get(intent, set())
    return tools if tools else None


def get_relevant_guides(intent: str) -> list[str]:
    """Return guide keys relevant to the intent."""
    return INTENT_GUIDES.get(intent, INTENT_GUIDES['conversation'])
