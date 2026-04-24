"""Hybrid XML/ReAct parser for extracting tool calls from model output."""

from __future__ import annotations
import json
import re
from dataclasses import dataclass


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
            except json.JSONDecodeError:
                continue

        # Try ReAct pattern if no XML matches
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

        # Bare JSON fallback (only if nothing else matched)
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

        # Strip tool call text from response
        cleaned = text
        for call in calls:
            cleaned = cleaned.replace(call.raw_match, '')

        # Clean up residual whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        return cleaned, calls
