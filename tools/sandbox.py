"""Directory-scoped sandbox for tool execution security."""

import os
from pathlib import Path


class Sandbox:
    """Enforces directory-scoped access for all file and shell operations."""

    def __init__(self, allowed_dirs: list[str] | None = None):
        if allowed_dirs:
            self._allowed = [Path(d).resolve() for d in allowed_dirs]
        else:
            # Default: nemo-harness working directory
            self._allowed = [Path(__file__).parent.parent.resolve()]

    @property
    def allowed_dirs(self) -> list[Path]:
        return list(self._allowed)

    def check_path(self, path: str) -> Path:
        """Validate and resolve a path. Raises PermissionError if outside sandbox."""
        resolved = Path(path).resolve()
        for allowed in self._allowed:
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue
        raise PermissionError(
            f'Path {path} is outside allowed directories: '
            f'{[str(d) for d in self._allowed]}'
        )

    def check_command(self, command: str) -> str:
        """Basic command validation. Returns command if safe, raises on dangerous patterns."""
        dangerous = [
            'rm -rf /', 'mkfs', 'dd if=', ':(){', 'chmod -R 777 /',
            '> /dev/sd', 'shutdown', 'reboot', 'init 0', 'init 6',
        ]
        cmd_lower = command.lower().strip()
        for pattern in dangerous:
            if pattern in cmd_lower:
                raise PermissionError(f'Blocked dangerous command pattern: {pattern}')
        return command

    def get_cwd(self) -> str:
        """Return the primary sandbox directory as working directory for shell commands."""
        return str(self._allowed[0])
