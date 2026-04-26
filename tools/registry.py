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

    def openai_tools(self, only: set[str] | None = None) -> list[dict]:
        """Generate OpenAI-style tool definitions for the native tools API.

        If `only` is provided, restrict output to those tool names.
        """
        tools = []
        for tool in self._tools.values():
            if only is not None and tool.name not in only:
                continue
            properties = {}
            for pname, pdef in tool.parameters.items():
                properties[pname] = {
                    'type': pdef['type'],
                    'description': pdef['description'],
                }
            tools.append({
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': {
                        'type': 'object',
                        'properties': properties,
                        'required': tool.required,
                    },
                },
            })
        return tools

    def prompt_block(self) -> str:
        """Generate a compact tool-use reminder for the system prompt.

        The actual tool schemas are now sent via the native OpenAI tools API,
        so this block only contains behavioral guidance.
        """
        return (
            '# TOOL USE\n'
            'You have tools available via function calling. Use them.\n'
            'When the user asks you to read a file, write a file, list a directory, '
            'run a command, search code, search the web, fetch a URL, create a document, '
            'or make a plan: call the appropriate tool. Do not describe what you would do.\n'
            'After a tool returns a result, use it to formulate your response to the user.\n'
            'If a tool fails, tell the user what happened and try an alternative.\n'
        )

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
            has_error = isinstance(result, dict) and 'error' in result
            return {**result, 'success': not has_error}
        except PermissionError as e:
            return {'error': f'Permission denied: {e}', 'success': False}
        except FileNotFoundError as e:
            return {'error': f'File not found: {e}', 'success': False}
        except Exception as e:
            return {'error': f'{type(e).__name__}: {e}', 'success': False}


# Global registry instance
TOOL_REGISTRY = ToolRegistry()
