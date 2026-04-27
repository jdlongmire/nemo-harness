"""Output validation layer for Morals post-checks.

4M Module: Morals (Layer 6: output validation)
Purpose: Check model output after generation for policy violations.

This addresses Gap 4 from the conformance audit: "No output validation layer."
Violations are logged via Ch3 (Morals->Mind) and optionally filtered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import channels

# Patterns that should never appear in model output
_SECRET_PATTERNS = [
    (re.compile(r'(?:api[_-]?key|secret|token|password)\s*[:=]\s*["\']?[A-Za-z0-9+/=_-]{20,}', re.IGNORECASE),
     'potential_secret_leak'),
    (re.compile(r'(?:sk-|ghp_|gho_|github_pat_)[A-Za-z0-9]{20,}'),
     'api_key_pattern'),
]

# Sandbox path patterns that should not be revealed to users
_PATH_LEAK_PATTERNS = [
    (re.compile(r'/etc/(?:passwd|shadow|sudoers)'),
     'sensitive_path_reference'),
    (re.compile(r'(?:rm\s+-rf\s+/|mkfs\s|dd\s+if=|:$$\)\{)'),
     'dangerous_command_in_output'),
]


@dataclass
class ValidationResult:
    """Result of output validation."""
    passed: bool = True
    violations: list[str] = field(default_factory=list)
    filtered_text: str = ''


def validate_output(text: str, cid: str = '') -> ValidationResult:
    """Run Morals post-checks on model output text.

    Returns ValidationResult with pass/fail status and any violations found.
    Violations are logged via Ch3 channel. Text is not modified by default;
    the caller decides whether to filter.
    """
    result = ValidationResult(filtered_text=text)

    if not text:
        return result

    # Check for secret/credential leaks
    for pattern, violation_name in _SECRET_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            result.passed = False
            detail = f'{violation_name}: {len(matches)} match(es)'
            result.violations.append(detail)
            channels.log_morals_violation(cid, 'output_secrets', detail)

    # Check for dangerous command patterns in output
    for pattern, violation_name in _PATH_LEAK_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            result.passed = False
            detail = f'{violation_name}: {matches[0][:50]}'
            result.violations.append(detail)
            channels.log_morals_violation(cid, 'output_dangerous', detail)

    if result.passed:
        channels.log_morals_pass(cid, 'output_validation')

    return result


def redact_secrets(text: str) -> str:
    """Redact detected secrets from output text."""
    for pattern, _ in _SECRET_PATTERNS:
        text = pattern.sub('[REDACTED]', text)
    return text
