"""Inference client abstraction for swappable model providers.

4M Module: Means (execution substrate)
Provides: provider-agnostic interface for LLM inference calls.

Current implementations:
  - NemotronClient: Local NVIDIA Nemotron via OpenAI-compatible API
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

import aiohttp

logger = logging.getLogger('nemo-harness.inference')


@dataclass
class StreamEvent:
    """A single event from a streaming inference response."""
    type: str  # 'content', 'reasoning', 'tool_call_delta', 'usage', 'done', 'error'
    data: dict = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result from a single inference call."""
    text: str
    tool_calls: list[dict]  # [{id, name, arguments_str}]
    error: bool = False


class InferenceClient(ABC):
    """Abstract interface for LLM inference providers."""

    @abstractmethod
    async def stream_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        timeout: float = 120.0,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion. Yields StreamEvent objects."""
        ...

    @abstractmethod
    async def completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout: float = 15.0,
    ) -> str | None:
        """Non-streaming completion (for summarization, etc). Returns text or None on error."""
        ...


class NemotronClient(InferenceClient):
    """OpenAI-compatible client for local Nemotron inference."""

    def __init__(self, api_base: str):
        self.api_base = api_base

    async def stream_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        timeout: float = 120.0,
    ) -> AsyncIterator[StreamEvent]:
        payload = {
            'model': model,
            'messages': messages,
            'stream': True,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if tools:
            payload['tools'] = tools

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.api_base}/v1/chat/completions',
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        yield StreamEvent(type='error', data={'message': f'API error {resp.status}: {error_text}'})
                        return

                    async for line in resp.content:
                        line = line.decode('utf-8').strip()
                        if not line.startswith('data: '):
                            continue
                        data = line[6:]
                        if data == '[DONE]':
                            break

                        try:
                            chunk = json.loads(data)
                            delta = chunk.get('choices', [{}])[0].get('delta', {})

                            if delta.get('reasoning_content'):
                                yield StreamEvent(type='reasoning', data={'content': delta['reasoning_content']})

                            if delta.get('content'):
                                yield StreamEvent(type='content', data={'content': delta['content']})

                            for tc_delta in delta.get('tool_calls', []):
                                yield StreamEvent(type='tool_call_delta', data=tc_delta)

                            usage = chunk.get('usage')
                            if usage:
                                yield StreamEvent(type='usage', data=usage)

                        except json.JSONDecodeError:
                            continue

        except TimeoutError:
            yield StreamEvent(type='error', data={'message': f'Inference timed out after {timeout}s'})
        except aiohttp.ClientError as e:
            yield StreamEvent(type='error', data={'message': f'Connection error: {e}'})
        except Exception as e:
            yield StreamEvent(type='error', data={'message': f'Unexpected error: {e}'})

    async def completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout: float = 15.0,
    ) -> str | None:
        payload = {
            'model': model,
            'messages': messages,
            'stream': False,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.api_base}/v1/chat/completions',
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    return data['choices'][0]['message']['content'].strip()
        except Exception:
            return None
