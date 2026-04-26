"""Hybrid XML/ReAct parser for extracting tool calls from model output."""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ParsedToolCall:
    name: str
    args: dict
    raw_match: str  # the full matched text to strip from response


# Pattern 1: XML-style <tool_call>{"name": ..., "args": ...}</tool_call>
_XML_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL
)

# Pattern 2: ReAct-style "Action: tool_name\nAction Input: {json}"
_REACT_PATTERN = re.compile(
    r'Action:\s*(\w+)\s*\n\s*Action Input:\s*(\{.*?\})(?:\n|$)',
    re.DOTALL
)

# Pattern 3: Bare JSON with "name" and "args" keys (fallback)
_BARE_JSON_PATTERN = re.compile(
    r'\{"name"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL
)

# Pattern 4: Orphaned closing tag — {args_json}</tool_call> without opening tag
_ORPHAN_CLOSE_PATTERN = re.compile(
    r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*</tool_call>',
    re.DOTALL
)

# Pattern 5: Bare args JSON (no tags at all) — last resort
_BARE_ARGS_PATTERN = re.compile(
    r'^\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*$',
    re.DOTALL
)

# Pattern 6: OpenAI-style <function=name><parameter=key>value</parameter>...</function>
_FUNCTION_TAG_PATTERN = re.compile(
    r'<function=(\w+)>(.*?)</function>',
    re.DOTALL
)
_PARAMETER_TAG_PATTERN = re.compile(
    r'<parameter=(\w+)>(.*?)</parameter>',
    re.DOTALL
)

# Map arg signatures to tool names for when the model omits the name
_ARG_SIGNATURE_MAP = {
    frozenset(['file_path', 'content']): 'write_file',
    frozenset(['file_path', 'old_string', 'new_string']): 'edit_file',
    frozenset(['file_path']): 'read_file',
    frozenset(['path']): 'list_dir',
    frozenset(['command']): 'run_command',
    frozenset(['pattern']): 'glob_search',
    frozenset(['query']): 'web_search',
    frozenset(['url']): 'web_fetch',
    frozenset(['query', 'path']): 'grep_search',
    frozenset(['query', 'file_pattern']): 'grep_search',
    frozenset(['query', 'path', 'file_pattern']): 'grep_search',
    frozenset(['name', 'goal', 'tasks']): 'create_plan',
    frozenset(['task_id', 'status']): 'update_task',
    frozenset(['description']): 'add_task',
    frozenset(['name', 'description']): 'create_repo',
    frozenset(['repo', 'path', 'content']): 'commit_file',
    frozenset(['repo', 'path', 'content', 'message']): 'commit_file',
    frozenset(['repo', 'title']): 'create_issue',
    frozenset(['repo', 'title', 'body']): 'create_issue',
    frozenset(['repo']): 'list_issues',
    frozenset(['repo', 'state']): 'list_issues',
    frozenset(['content', 'source']): 'index_text',
    frozenset(['query', 'repo']): 'semantic_search',
    frozenset(['query', 'repo', 'top_k']): 'semantic_search',
}


def _infer_tool_name(args: dict) -> str | None:
    """Infer tool name from argument keys when the model omits it."""
    keys = frozenset(args.keys())
    # Direct match
    if keys in _ARG_SIGNATURE_MAP:
        return _ARG_SIGNATURE_MAP[keys]
    # Try subsets (required args only, ignoring optional extras)
    for sig, name in _ARG_SIGNATURE_MAP.items():
        if sig <= keys:  # sig is subset of keys
            return name
    return None


class ToolCallParser:
    """Extracts tool calls from model output using multiple parsing strategies."""

    @staticmethod
    def parse(text: str) -> tuple[str, list[ParsedToolCall]]:
        """Parse tool calls from text. Returns (cleaned_text, tool_calls)."""
        calls = []

        # Try XML pattern first (preferred)
        for match in _XML_PATTERN.finditer(text):
            try:
                data = json.loads(match.group(1))
                if 'name' in data:
                    calls.append(ParsedToolCall(
                        name=data['name'],
                        args=data.get('args', {}),
                        raw_match=match.group(0),
                    ))
                elif 'name' not in data:
                    # Model put args directly inside <tool_call> tags
                    inferred = _infer_tool_name(data)
                    if inferred:
                        log.info('Inferred tool name %r from args in XML block', inferred)
                        calls.append(ParsedToolCall(
                            name=inferred,
                            args=data,
                            raw_match=match.group(0),
                        ))
            except json.JSONDecodeError:
                continue

        # Try orphaned closing tag pattern (model forgot opening <tool_call>)
        if not calls:
            for match in _ORPHAN_CLOSE_PATTERN.finditer(text):
                try:
                    data = json.loads(match.group(1))
                    if 'name' in data and 'args' in data:
                        calls.append(ParsedToolCall(
                            name=data['name'],
                            args=data.get('args', {}),
                            raw_match=match.group(0),
                        ))
                    else:
                        inferred = _infer_tool_name(data)
                        if inferred:
                            log.info('Inferred tool name %r from orphaned </tool_call> block', inferred)
                            calls.append(ParsedToolCall(
                                name=inferred,
                                args=data,
                                raw_match=match.group(0),
                            ))
                except json.JSONDecodeError:
                    continue

        # Try ReAct pattern
        if not calls:
            for match in _REACT_PATTERN.finditer(text):
                try:
                    args = json.loads(match.group(2))
                    calls.append(ParsedToolCall(
                        name=match.group(1),
                        args=args,
                        raw_match=match.group(0),
                    ))
                except json.JSONDecodeError:
                    continue

        # OpenAI-style <function=name><parameter=key>value</parameter>...</function>
        if not calls:
            for match in _FUNCTION_TAG_PATTERN.finditer(text):
                func_name = match.group(1)
                body = match.group(2)
                args = {}
                for pmatch in _PARAMETER_TAG_PATTERN.finditer(body):
                    args[pmatch.group(1)] = pmatch.group(2).strip()
                if args:
                    log.info('Parsed function-tag tool call: %s(%s)', func_name, list(args.keys()))
                    calls.append(ParsedToolCall(
                        name=func_name,
                        args=args,
                        raw_match=match.group(0),
                    ))

        # Bare JSON with "name" and "args" keys
        if not calls:
            for match in _BARE_JSON_PATTERN.finditer(text):
                try:
                    args = json.loads(match.group(2))
                    calls.append(ParsedToolCall(
                        name=match.group(1),
                        args=args,
                        raw_match=match.group(0),
                    ))
                except json.JSONDecodeError:
                    continue

        # Last resort: bare args JSON (entire response is just args)
        if not calls:
            match = _BARE_ARGS_PATTERN.match(text)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if isinstance(data, dict) and 'name' in data and 'args' in data:
                        calls.append(ParsedToolCall(
                            name=data['name'],
                            args=data['args'],
                            raw_match=match.group(0),
                        ))
                    elif isinstance(data, dict):
                        inferred = _infer_tool_name(data)
                        if inferred:
                            log.info('Inferred tool name %r from bare args JSON', inferred)
                            calls.append(ParsedToolCall(
                                name=inferred,
                                args=data,
                                raw_match=match.group(0),
                            ))
                except json.JSONDecodeError:
                    pass

        # Strip tool call text from response
        cleaned = text
        for call in calls:
            cleaned = cleaned.replace(call.raw_match, '')

        # Also strip any orphaned </tool_call> tags
        cleaned = re.sub(r'</tool_call>', '', cleaned)

        # Clean up residual whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        return cleaned, calls
