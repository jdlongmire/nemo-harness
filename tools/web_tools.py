"""Web fetch tool for the Nemo tool-calling system."""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

import aiohttp

from tools.registry import ToolDef, TOOL_REGISTRY

FETCH_TIMEOUT = 15
FETCH_MAX_BYTES = 500 * 1024


def _is_private_ip(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return True
    except (socket.gaierror, ValueError):
        return True
    return False


def _strip_html_tags(html: str) -> str:
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<head[^>]*>.*?</head>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<[^>]+>', ' ', html)
    html = re.sub(r'&nbsp;', ' ', html)
    html = re.sub(r'&amp;', '&', html)
    html = re.sub(r'&lt;', '<', html)
    html = re.sub(r'&gt;', '>', html)
    html = re.sub(r'&quot;', '"', html)
    html = re.sub(r'&#39;', "'", html)
    html = re.sub(r'\s+', ' ', html)
    return html.strip()


async def _web_fetch(url: str) -> dict:
    """Fetch a URL and return its text content."""
    url = url.strip()
    if not url:
        return {'error': 'Missing url'}

    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return {'error': 'Only http and https URLs are allowed'}

    if not parsed.hostname:
        return {'error': 'Invalid URL'}

    if _is_private_ip(parsed.hostname):
        return {'error': 'Access to private/local addresses is not allowed'}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT),
                max_redirects=5,
                headers={'User-Agent': 'NemoHarness/1.0'},
            ) as resp:
                if resp.status != 200:
                    return {'error': f'HTTP {resp.status}', 'url': url}

                content_type = resp.content_type or 'application/octet-stream'
                raw = await resp.content.read(FETCH_MAX_BYTES + 1)
                truncated = len(raw) > FETCH_MAX_BYTES
                if truncated:
                    raw = raw[:FETCH_MAX_BYTES]

                encoding = resp.charset or 'utf-8'
                try:
                    text = raw.decode(encoding, errors='replace')
                except (LookupError, UnicodeDecodeError):
                    text = raw.decode('utf-8', errors='replace')

                if 'html' in content_type:
                    text = _strip_html_tags(text)

                return {
                    'url': url,
                    'content_type': content_type,
                    'content': text[:4000],  # cap for model context
                    'truncated': truncated or len(text) > 4000,
                    'success': True,
                }

    except Exception as e:
        return {'error': f'Fetch failed: {e}', 'url': url}


async def _web_search(query: str) -> dict:
    """Search the web using DuckDuckGo and return result snippets."""
    query = query.strip()
    if not query:
        return {'error': 'Missing search query'}

    search_url = 'https://html.duckduckgo.com/html/'
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                search_url,
                data={'q': query},
                timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT),
                headers={'User-Agent': 'NemoHarness/1.0'},
            ) as resp:
                if resp.status != 200:
                    return {'error': f'Search failed: HTTP {resp.status}'}
                html = await resp.text()

        results = []
        # Parse DuckDuckGo HTML results
        snippets = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        )
        for href, title, snippet in snippets[:5]:
            # DDG wraps URLs in a redirect; extract the actual URL
            actual = re.search(r'uddg=([^&]+)', href)
            if actual:
                from urllib.parse import unquote
                href = unquote(actual.group(1))
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            results.append({'title': title, 'url': href, 'snippet': snippet})

        if not results:
            return {'query': query, 'results': [], 'message': 'No results found', 'success': True}

        return {'query': query, 'results': results, 'success': True}

    except Exception as e:
        return {'error': f'Search failed: {e}'}


def register():
    TOOL_REGISTRY.register(ToolDef(
        name='web_search',
        description='Search the web for a topic. Returns titles, URLs, and snippets for the top results. Use this FIRST when researching a topic, before using web_fetch on specific URLs.',
        parameters={
            'query': {'type': 'string', 'description': 'Search query text'},
        },
        handler=_web_search,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='web_fetch',
        description='Fetch and read a specific web page URL. Returns the text content of the page. Use web_search first to find relevant URLs, then use this to read them.',
        parameters={
            'url': {'type': 'string', 'description': 'The URL to fetch (http or https)'},
        },
        handler=_web_fetch,
    ))
