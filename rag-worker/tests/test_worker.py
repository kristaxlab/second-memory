"""Unit tests for the rag-worker module."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import Config
from worker import RAGWorker, _extract_text, _parse_message


# ---------------------------------------------------------------------------
# _parse_message
# ---------------------------------------------------------------------------


class TestParseMessage:
    def test_plain_url_bytes(self):
        url, meta = _parse_message(b"https://example.com/doc.pdf")
        assert url == "https://example.com/doc.pdf"
        assert meta == {}

    def test_plain_url_string(self):
        url, meta = _parse_message("https://example.com/doc.html")
        assert url == "https://example.com/doc.html"
        assert meta == {}

    def test_json_with_url_and_metadata(self):
        payload = json.dumps({"url": "https://example.com/report.pdf", "metadata": {"title": "Report"}})
        url, meta = _parse_message(payload)
        assert url == "https://example.com/report.pdf"
        assert meta == {"title": "Report"}

    def test_json_without_metadata_key(self):
        payload = json.dumps({"url": "https://example.com/page"})
        url, meta = _parse_message(payload)
        assert url == "https://example.com/page"
        assert meta == {}

    def test_json_missing_url_returns_empty(self):
        payload = json.dumps({"metadata": {"title": "No URL"}})
        url, meta = _parse_message(payload)
        assert url == ""

    def test_whitespace_is_stripped(self):
        url, _ = _parse_message(b"  https://example.com/  ")
        assert url == "https://example.com/"


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_plain_html(self):
        html = b"<html><body><p>Hello world</p></body></html>"
        text = _extract_text(html, "text/html", "https://example.com/page.html")
        assert "Hello world" in text

    def test_strips_script_tags(self):
        html = b"<html><body><script>alert(1)</script><p>Content</p></body></html>"
        text = _extract_text(html, "text/html", "https://example.com/")
        assert "alert" not in text
        assert "Content" in text

    def test_plain_text_fallback(self):
        raw = b"Just plain text content"
        text = _extract_text(raw, "text/plain", "https://example.com/notes.txt")
        assert text == "Just plain text content"

    def test_html_detected_by_extension(self):
        html = b"<html><body><p>Via ext</p></body></html>"
        text = _extract_text(html, "application/octet-stream", "https://example.com/page.html")
        assert "Via ext" in text


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.redis_channel == "document_links"
        assert cfg.postgres_port == 5432
        assert cfg.embedding_dim == 1536

    def test_postgres_dsn(self):
        cfg = Config()
        cfg.postgres_user = "user"
        cfg.postgres_password = "pass"
        cfg.postgres_host = "db"
        cfg.postgres_port = 5432
        cfg.postgres_db = "mydb"
        assert cfg.postgres_dsn == "postgresql://user:pass@db:5432/mydb"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("REDIS_CHANNEL", "custom_channel")
        monkeypatch.setenv("POSTGRES_DB", "custom_db")
        cfg = Config()
        assert cfg.redis_channel == "custom_channel"
        assert cfg.postgres_db == "custom_db"


# ---------------------------------------------------------------------------
# RAGWorker
# ---------------------------------------------------------------------------


class TestRAGWorkerProcessMessage:
    """Test _process_message without any network or Redis calls."""

    @pytest.mark.asyncio
    async def test_empty_url_is_skipped(self):
        worker = RAGWorker(Config())
        worker.embed_document = AsyncMock()
        await worker._process_message(b"")
        worker.embed_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_plain_url_is_embedded(self):
        worker = RAGWorker(Config())
        worker.embed_document = AsyncMock()
        await worker._process_message(b"https://example.com/doc.pdf")
        worker.embed_document.assert_awaited_once_with("https://example.com/doc.pdf", {})

    @pytest.mark.asyncio
    async def test_json_url_is_embedded(self):
        worker = RAGWorker(Config())
        worker.embed_document = AsyncMock()
        payload = json.dumps({"url": "https://example.com/report.pdf", "metadata": {"k": "v"}})
        await worker._process_message(payload.encode())
        worker.embed_document.assert_awaited_once_with(
            "https://example.com/report.pdf", {"k": "v"}
        )


class TestRAGWorkerEmbedDocument:
    """Test embed_document with mocked HTTP and RAG."""

    @pytest.mark.asyncio
    async def test_successful_embedding(self):
        worker = RAGWorker(Config())

        mock_rag = MagicMock()
        mock_rag.initialize_storages = AsyncMock()
        mock_rag.insert = MagicMock()
        worker._rag = mock_rag

        html_content = b"<html><body><p>Test content</p></body></html>"

        with patch("worker.aiohttp.ClientSession") as mock_session_cls:
            mock_response = AsyncMock()
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.read = AsyncMock(return_value=html_content)
            mock_response.raise_for_status = MagicMock()
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            await worker.embed_document("https://example.com/page.html")

        mock_rag.insert.assert_called_once()
        inserted_text = mock_rag.insert.call_args[0][0]
        assert "Test content" in inserted_text

    @pytest.mark.asyncio
    async def test_metadata_prepended_to_text(self):
        worker = RAGWorker(Config())

        mock_rag = MagicMock()
        mock_rag.initialize_storages = AsyncMock()
        mock_rag.insert = MagicMock()
        worker._rag = mock_rag

        html_content = b"<html><body><p>Body text</p></body></html>"

        with patch("worker.aiohttp.ClientSession") as mock_session_cls:
            mock_response = AsyncMock()
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.read = AsyncMock(return_value=html_content)
            mock_response.raise_for_status = MagicMock()
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            await worker.embed_document(
                "https://example.com/page.html",
                metadata={"title": "My Doc", "author": "Alice"},
            )

        mock_rag.insert.assert_called_once()
        inserted_text = mock_rag.insert.call_args[0][0]
        assert "title: My Doc" in inserted_text
        assert "author: Alice" in inserted_text
        assert "Body text" in inserted_text

    @pytest.mark.asyncio
    async def test_http_error_is_handled_gracefully(self):
        """_process_message must swallow HTTP errors without crashing the worker."""
        import aiohttp

        worker = RAGWorker(Config())

        with patch("worker.aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(
                side_effect=aiohttp.ClientConnectionError("connection refused")
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            # _process_message catches ClientError – should not raise
            await worker._process_message(b"https://example.com/missing.pdf")

    @pytest.mark.asyncio
    async def test_empty_text_skips_insert(self):
        worker = RAGWorker(Config())

        mock_rag = MagicMock()
        mock_rag.initialize_storages = AsyncMock()
        mock_rag.insert = MagicMock()
        worker._rag = mock_rag

        with patch("worker.aiohttp.ClientSession") as mock_session_cls:
            mock_response = AsyncMock()
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.read = AsyncMock(return_value=b"<html><body></body></html>")
            mock_response.raise_for_status = MagicMock()
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            await worker.embed_document("https://example.com/empty.html")

        mock_rag.insert.assert_not_called()
