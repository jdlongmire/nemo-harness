"""Sandboxed shell execution tool."""

from __future__ import annotations
import asyncio

from tools.sandbox import Sandbox
from tools.registry import ToolDef, TOOL_REGISTRY

SHELL_TIMEOUT = 30  # seconds


def register(sandbox: Sandbox):
    """Register shell tools with the global registry."""

    async def run_command(command: str, timeout: int = SHELL_TIMEOUT) -> dict:
        sandbox.check_command(command)
        cwd = sandbox.get_cwd()
        timeout = min(timeout, 120)  # hard cap at 2 minutes

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            stdout_text = stdout.decode('utf-8', errors='replace')[:50000]
            stderr_text = stderr.decode('utf-8', errors='replace')[:10000]

            return {
                'stdout': stdout_text,
                'stderr': stderr_text,
                'exit_code': proc.returncode,
            }
        except asyncio.TimeoutError:
            proc.kill()
            return {'error': f'Command timed out after {timeout}s', 'exit_code': -1}

    TOOL_REGISTRY.register(ToolDef(
        name='run_command',
        description='Execute a shell command in the sandbox directory',
        parameters={
            'command': {'type': 'string', 'description': 'Shell command to execute'},
            'timeout': {'type': 'integer', 'description': 'Timeout in seconds (max 120)'},
        },
        required=['command'],
        handler=run_command,
    ))
