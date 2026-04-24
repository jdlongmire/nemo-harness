"""Semantic search (RAG) tools for the Nemo tool-calling system.

Uses ChromaDB for vector storage and Ollama for local embeddings.
Fully self-contained for air-gapped deployment.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from tools.registry import ToolDef, TOOL_REGISTRY

logger = logging.getLogger('nemo.rag')

# Configuration via environment
CHROMA_DIR = os.getenv('CHROMA_DIR', str(Path(__file__).parent.parent / 'chromadb_data'))
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE', '512'))
CHUNK_OVERLAP = int(os.getenv('RAG_CHUNK_OVERLAP', '64'))
TOP_K = int(os.getenv('RAG_TOP_K', '5'))

# ChromaDB client (lazy init)
_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


class OllamaEmbedder(chromadb.EmbeddingFunction):
    """Local embedding function using Ollama API."""

    def __init__(self, base_url: str = OLLAMA_URL, model: str = EMBED_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        import requests
        embeddings = []
        for text in input:
            resp = requests.post(
                f'{self.base_url}/api/embeddings',
                json={'model': self.model, 'prompt': text},
                timeout=30,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()['embedding'])
        return embeddings


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection with Ollama embeddings."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    _collection = _chroma_client.get_or_create_collection(
        name='nemo_rag',
        embedding_function=OllamaEmbedder(),
        metadata={'hnsw:space': 'cosine'},
    )
    logger.info(f'ChromaDB collection "nemo_rag" ready ({_collection.count()} documents)')
    return _collection


def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    lines = text.split('\n')
    chunks = []
    current = []
    current_len = 0

    for i, line in enumerate(lines):
        words = len(line.split())
        if current_len + words > CHUNK_SIZE and current:
            chunk_text = '\n'.join(current)
            chunk_id = hashlib.sha256(f'{source}:{i}:{chunk_text[:100]}'.encode()).hexdigest()[:16]
            chunks.append({
                'id': f'{source}_{chunk_id}',
                'text': chunk_text,
                'metadata': {'source': source, 'chunk_index': len(chunks)},
            })
            # Keep overlap
            overlap_lines = []
            overlap_len = 0
            for prev_line in reversed(current):
                wc = len(prev_line.split())
                if overlap_len + wc > CHUNK_OVERLAP:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_len += wc
            current = overlap_lines
            current_len = overlap_len

        current.append(line)
        current_len += words

    if current:
        chunk_text = '\n'.join(current)
        chunk_id = hashlib.sha256(f'{source}:end:{chunk_text[:100]}'.encode()).hexdigest()[:16]
        chunks.append({
            'id': f'{source}_{chunk_id}',
            'text': chunk_text,
            'metadata': {'source': source, 'chunk_index': len(chunks)},
        })

    return chunks


def index_content(content: str, source: str, repo: str = '', metadata: dict | None = None):
    """Index content into ChromaDB. Called by the webhook indexer."""
    collection = _get_collection()
    chunks = _chunk_text(content, source)
    if not chunks:
        return 0

    ids = [c['id'] for c in chunks]
    documents = [c['text'] for c in chunks]
    metadatas = []
    for c in chunks:
        m = {**c['metadata']}
        if repo:
            m['repo'] = repo
        if metadata:
            m.update(metadata)
        metadatas.append(m)

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info(f'Indexed {len(chunks)} chunks from {source} (repo: {repo})')
    return len(chunks)


def delete_source(source: str):
    """Remove all chunks for a given source file."""
    collection = _get_collection()
    results = collection.get(where={'source': source})
    if results['ids']:
        collection.delete(ids=results['ids'])
        logger.info(f'Deleted {len(results["ids"])} chunks for {source}')


async def _semantic_search(query: str, repo: str = '', top_k: str = '') -> dict:
    """Search indexed content semantically."""
    query = query.strip()
    if not query:
        return {'error': 'Search query is required'}

    k = int(top_k) if top_k.strip() else TOP_K
    k = min(k, 20)  # cap

    start = time.time()
    try:
        collection = _get_collection()
        where = {'repo': repo} if repo.strip() else None
        results = collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
        )
    except Exception as e:
        return {'error': f'Search failed: {e}'}

    elapsed = round(time.time() - start, 3)

    hits = []
    if results and results.get('documents'):
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i] if results.get('metadatas') else {}
            dist = results['distances'][0][i] if results.get('distances') else None
            hits.append({
                'content': doc[:500],  # cap per chunk for model context
                'source': meta.get('source', 'unknown'),
                'repo': meta.get('repo', ''),
                'score': round(1 - dist, 4) if dist is not None else None,  # cosine similarity
                'chunk_index': meta.get('chunk_index', 0),
            })

    logger.info(f'semantic_search: query="{query[:50]}" repo="{repo}" k={k} '
                f'hits={len(hits)} time={elapsed}s')

    return {
        'query': query,
        'results': hits,
        'count': len(hits),
        'search_time_s': elapsed,
        'total_indexed': collection.count(),
    }


async def _index_text(content: str, source: str, repo: str = '') -> dict:
    """Manually index text content into the RAG store."""
    content = content.strip()
    source = source.strip()
    if not content or not source:
        return {'error': 'Both content and source are required'}

    try:
        count = index_content(content, source, repo)
        return {'message': f'Indexed {count} chunks from {source}', 'chunks': count}
    except Exception as e:
        return {'error': f'Indexing failed: {e}'}


async def _rag_status() -> dict:
    """Get RAG system status."""
    try:
        collection = _get_collection()
        count = collection.count()

        # Get unique sources and repos
        all_meta = collection.get(include=['metadatas'])
        sources = set()
        repos = set()
        for m in (all_meta.get('metadatas') or []):
            if m.get('source'):
                sources.add(m['source'])
            if m.get('repo'):
                repos.add(m['repo'])

        return {
            'total_chunks': count,
            'unique_sources': len(sources),
            'repos': sorted(repos),
            'embed_model': EMBED_MODEL,
            'ollama_url': OLLAMA_URL,
            'chroma_dir': CHROMA_DIR,
        }
    except Exception as e:
        return {'error': f'RAG status check failed: {e}'}


def register():
    """Register RAG tools with the global registry."""

    TOOL_REGISTRY.register(ToolDef(
        name='semantic_search',
        description='Search indexed repository content by meaning. Returns the most relevant code, docs, and issues. Use this to find context before answering questions about projects.',
        parameters={
            'query': {'type': 'string', 'description': 'Natural language search query'},
            'repo': {'type': 'string', 'description': 'Filter to a specific repo name (empty for all)'},
            'top_k': {'type': 'string', 'description': 'Number of results (default: 5, max: 20)'},
        },
        required=['query'],
        handler=_semantic_search,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='index_text',
        description='Manually index text content into the semantic search store. Use this to add context that will be searchable later.',
        parameters={
            'content': {'type': 'string', 'description': 'Text content to index'},
            'source': {'type': 'string', 'description': 'Source identifier (e.g., "project-x/README.md")'},
            'repo': {'type': 'string', 'description': 'Repository name for filtering (optional)'},
        },
        required=['content', 'source'],
        handler=_index_text,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='rag_status',
        description='Check the status of the semantic search system. Shows indexed content count, repos, and configuration.',
        parameters={},
        required=[],
        handler=_rag_status,
    ))
