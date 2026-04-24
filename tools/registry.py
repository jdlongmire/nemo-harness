"""Tool registry with schema definitions for the Nemo tool-calling system."""

from __future__ import annotations
import json
from typing import Any, Callable, Awaitable


class ToolDef:
    """Definition of a single tool."""

    def __init__(self, name: str, description: str, parameters: dict[str, dict],
                 handler: Callable[..., Awaitable[dict]], required: list[str] | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters  # {param_name: {type, description}}
        self.required = required or list(parameters.keys())
        self.handler = handler

    def schema_for_prompt(self) -> str:
        """Format tool schema for injection into the system prompt."""
        params_desc = []
        for pname, pdef in self.parameters.items():
            req = ' (required)' if pname in self.required else ' (optional)'
            params_desc.append(f'    {pname}: {pdef["type"]} - {pdef["description"]}{req}')
        params_block = '\n'.join(params_desc)
        return f'{self.name}: {self.description}\n  Parameters:\n{params_block}'


class ToolRegistry:
    """Registry of all available tools."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef):
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def prompt_block(self) -> str:
        """Generate the tool description block for injection into the system prompt."""
        lines = ['You have access to the following tools. To use a tool, output a tool call block:']
        lines.append('')
        lines.append('<tool_call>')
        lines.append('{"name": "tool_name", "args": {"param1": "value1"}}')
        lines.append('</tool_call>')
        lines.append('')
        lines.append('You may call multiple tools in sequence. After each tool call, you will receive')
        lines.append('the result in a <tool_result> block. Use the result to inform your next response.')
        lines.append('When you have enough information, respond directly without a tool call.')
        lines.append('')
        lines.append('Available tools:')
        lines.append('')
        for tool in self._tools.values():
            lines.append(tool.schema_for_prompt())
            lines.append('')
        return '\n'.join(lines)

    async def execute(self, name: str, args: dict[str, Any]) -> dict:
        """Execute a tool by name with given args. Returns result dict."""
        tool = self._tools.get(name)
        if not tool:
            return {'error': f'Unknown tool: {name}', 'success': False}

        # Validate required params
        missing = [p for p in tool.required if p not in args]
        if missing:
            return {'error': f'Missing required parameters: {missing}', 'success': False}

        try:
            result = await tool.handler(**args)
            return {**result, 'success': True}
        except PermissionError as e:
            return {'error': f'Permission denied: {e}', 'success': False}
        except FileNotFoundError as e:
            return {'error': f'File not found: {e}', 'success': False}
        except Exception as e:
            return {'error': f'{type(e).__name__}: {e}', 'success': False}


# Global registry instance
TOOL_REGISTRY = ToolRegistry()
