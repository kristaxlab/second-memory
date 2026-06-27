"""RAG Worker: subscribes to Redis pub/sub and embeds documents via LightRAG."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

import aiohttp
import redis.asyncio as aioredis
from bs4 import BeautifulSoup
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc
from pypdf import PdfReader

from config import Config

logger = logging.getLogger(__name__)

# Maximum document text size (in characters) passed to LightRAG at once
_MAX_CHUNK_CHARS = 200_000


def _build_rag(config: Config) -> LightRAG:
    """Instantiate LightRAG with PostgreSQL storage backends."""
    os.makedirs(config.working_dir, exist_ok=True)

    # LightRAG PostgreSQL storage reads these from env; set them explicitly so
    # the caller only needs to configure Config (or the environment).
    os.environ.setdefault("POSTGRES_HOST", config.postgres_host)
    os.environ.setdefault("POSTGRES_PORT", str(config.postgres_port))
    os.environ.setdefault("POSTGRES_USER", config.postgres_user)
    os.environ.setdefault("POSTGRES_PASSWORD", config.postgres_password)
    os.environ.setdefault("POSTGRES_DATABASE", config.postgres_db)

    llm_model = config.llm_model
    embedding_model = config.embedding_model

    async def llm_func(prompt: str, **kwargs: Any) -> str:
        return await gpt_4o_mini_complete(prompt, model=llm_model, **kwargs)

    async def embed_func(texts: list[str]) -> list[list[float]]:
        return await openai_embed(texts, model=embedding_model)

    return LightRAG(
        working_dir=config.working_dir,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGGraphStorage",
        doc_status_storage="PGDocStatusStorage",
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=config.embedding_dim,
            max_token_size=config.embedding_max_tokens,
            func=embed_func,
        ),
    )


def _parse_message(raw: bytes | str) -> tuple[str, dict[str, Any]]:
    """Return ``(url, metadata)`` from a raw Redis message.

    Supports two formats:
    - Plain URL string: ``"https://example.com/doc.pdf"``
    - JSON object:     ``{"url": "https://...", "metadata": {...}}``
    """
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    text = text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            url = payload.get("url", "")
            metadata = payload.get("metadata", {})
            return url, metadata
    except json.JSONDecodeError:
        pass
    return text, {}


async def _fetch_bytes(session: aiohttp.ClientSession, url: str) -> tuple[bytes, str]:
    """Download *url* and return ``(content, content_type)``."""
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
        return await response.read(), content_type


def _extract_text(content: bytes, content_type: str, url: str) -> str:
    """Convert raw bytes to plain text depending on content type / URL extension."""
    ext = urlparse(url).path.lower().rsplit(".", 1)[-1]

    if content_type == "application/pdf" or ext == "pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if content_type.startswith("text/html") or ext in ("html", "htm"):
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    # Fall back to UTF-8 plain text
    return content.decode("utf-8", errors="replace")


class RAGWorker:
    """Subscribes to a Redis channel and embeds received document links via LightRAG."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._rag: LightRAG | None = None
        self._redis: aioredis.Redis | None = None

    async def _get_rag(self) -> LightRAG:
        if self._rag is None:
            self._rag = _build_rag(self.config)
            await self._rag.initialize_storages()
        return self._rag

    async def embed_document(self, url: str, metadata: dict[str, Any] | None = None) -> None:
        """Download *url*, extract text and insert it into the RAG store."""
        logger.info("Embedding document: %s", url)
        try:
            async with aiohttp.ClientSession() as session:
                content, content_type = await _fetch_bytes(session, url)

            text = _extract_text(content, content_type, url)
            if not text.strip():
                logger.warning("No text extracted from %s – skipping", url)
                return

            rag = await self._get_rag()
            # LightRAG's insert is synchronous; run it in a thread to avoid
            # blocking the event loop.
            await asyncio.get_event_loop().run_in_executor(
                None, rag.insert, text[:_MAX_CHUNK_CHARS]
            )
            logger.info("Successfully embedded document: %s", url)

        except aiohttp.ClientError as exc:
            logger.error("HTTP error while fetching %s: %s", url, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while embedding %s: %s", url, exc)

    async def _process_message(self, data: bytes | str) -> None:
        url, metadata = _parse_message(data)
        if not url:
            logger.warning("Received empty or invalid message; skipping")
            return
        await self.embed_document(url, metadata)

    async def run(self) -> None:
        """Connect to Redis, subscribe to the configured channel and process messages."""
        logger.info(
            "Connecting to Redis at %s, channel=%s",
            self.config.redis_url,
            self.config.redis_channel,
        )
        self._redis = aioredis.from_url(
            self.config.redis_url,
            decode_responses=False,
        )
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self.config.redis_channel)
        logger.info("Subscribed to Redis channel '%s'", self.config.redis_channel)

        try:
            async for message in pubsub.listen():
                if message.get("type") == "message":
                    await self._process_message(message["data"])
        finally:
            await pubsub.unsubscribe(self.config.redis_channel)
            await self._redis.aclose()

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
