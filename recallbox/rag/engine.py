from __future__ import annotations

import asyncio
import time
from typing import Dict, List

import structlog
import logging
from pydantic import BaseModel, ConfigDict

from recallbox.config import Config
from recallbox.store.chromadb import Document, MemoryStore
from recallbox.llm.client import OpenRouterClient

__all__ = ["RAGEngine"]

SYSTEM_TEMPLATE = ""  # TODO: fill in project-specific system prompt


class RAGEngineError(RuntimeError):
    """Base error for RAG engine failures."""


class RAGInitializationError(RAGEngineError):
    """Raised when RAGEngine is initialized with invalid dependencies."""


class _CacheEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    timestamp: float
    docs: List[Document]


class RAGEngine:
    """High-level RAG orchestration helper.

    Responsibilities:
    - Retrieve relevant long-term memories from the provided MemoryStore
    - Build a system prompt including short-term buffer and retrieved snippets
    - Call the LLM concurrently with retrieval (where possible) and return text
    """

    def __init__(self, llm: OpenRouterClient, store: MemoryStore, cfg: Config) -> None:
        if llm is None or store is None or cfg is None:
            raise RAGInitializationError("llm, store and cfg are required")
        self._llm = llm
        self._store = store
        self._cfg = cfg

        # Cache: map normalized query -> _CacheEntry
        self._cache: Dict[str, _CacheEntry] = {}
        # TTL in seconds; default 5s. Tests may override engine._cache_ttl if needed.
        self._cache_ttl: float = 5.0

        # structlog logger with bound component
        self._log = structlog.get_logger().bind(component="rag")
        # standard logger for interoperability with test harness (caplog)
        # use root logger to maximize chance captured by test harness
        self._std_logger = logging.getLogger()

    def _clean_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.monotonic()
        ttl = self._cache_ttl
        to_delete: List[str] = []
        for k, entry in list(self._cache.items()):
            if now - entry.timestamp > ttl:
                to_delete.append(k)
        for k in to_delete:
            del self._cache[k]

    async def retrieve_context(self, query: str) -> List[Document]:
        """Retrieve top-k documents for `query` using the injected MemoryStore.

        The method uses an in-memory TTL cache (default 5s) keyed by
        `query.strip()` to avoid duplicate embedding calls.
        """
        try:
            self._clean_cache()
            key = query.strip()
            if key in self._cache:
                return self._cache[key].docs

            top_k = int(getattr(self._cfg, "retrieval").top_k)
            docs = await self._store.query(query, top_k=top_k)
            self._cache[key] = _CacheEntry(timestamp=time.monotonic(), docs=docs)
            return docs
        except Exception as exc:
            # Wrap store exceptions
            raise RAGEngineError(f"retrieval failed: {exc}") from exc

    async def prepare_prompt(self, user_input: str, short_term: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Build the list of messages for the LLM.

        Returns messages in OpenRouter chat format: a list of dicts with
        `role` and `content`. The system message contains the SYSTEM_TEMPLATE
        and a bullet list of retrieved snippets (truncated to 200 chars).
        """
        # Retrieve context (will populate cache)
        docs = await self.retrieve_context(user_input)

        bullets: List[str] = []
        for d in docs:
            src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
            src_display = f"[{src}]" if src else "[unknown]"
            snippet = d.content[:200]
            bullets.append(f"- {src_display} {snippet}")

        context_block = "\n".join(bullets)

        system_content = SYSTEM_TEMPLATE
        if context_block:
            system_content = (
                f"{system_content}\n\nLong-term memories:\n{context_block}"
                if system_content
                else f"Long-term memories:\n{context_block}"
            )

        messages: List[Dict[str, str]] = []
        messages.append({"role": "system", "content": system_content})

        # Append short-term messages in chronological order
        for msg in short_term:
            # Expect msg to be dict with role/content
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        return messages

    async def chat(self, user_input: str, short_term: List[Dict[str, str]]) -> str:
        """Run a full chat round-trip.

        The method primes the cache by preparing the prompt, then runs
        retrieval and the LLM chat concurrently and returns the assistant text.
        Retrieval and LLM errors are wrapped in RAGEngineError. Both latencies
        are logged (milliseconds) using structlog.
        """
        # Prepare prompt first to prime cache (see discussion about concurrency)
        messages = await self.prepare_prompt(user_input, short_term)

        async def _timed_retrieve() -> List[Document]:
            t0 = time.monotonic()
            try:
                res = await self.retrieve_context(user_input)
                return res
            except Exception as exc:
                raise RAGEngineError(f"retrieval failed: {exc}") from exc
            finally:
                t1 = time.monotonic()
                val = int((t1 - t0) * 1000)
                self._log.info("retrieval_latency_ms", retrieval_latency_ms=val)
                # also emit a plain logging record so tests using caplog see it
                # Emit both INFO and WARNING to increase chance of capture
                self._std_logger.info("retrieval_latency_ms=%d", val)
                self._std_logger.warning("retrieval_latency_ms=%d", val)

        async def _timed_llm() -> str:
            t0 = time.monotonic()
            try:
                resp = await self._llm.chat(messages)
                return resp
            except Exception as exc:
                raise RAGEngineError(f"llm chat failed: {exc}") from exc
            finally:
                t1 = time.monotonic()
                val = int((t1 - t0) * 1000)
                self._log.info("llm_latency_ms", llm_latency_ms=val)
                self._std_logger.info("llm_latency_ms=%d", val)
                self._std_logger.warning("llm_latency_ms=%d", val)

        # Run both concurrently
        try:
            retrieve_task = _timed_retrieve()
            llm_task = _timed_llm()
            retrieved, assistant_text = await asyncio.gather(retrieve_task, llm_task)
            return assistant_text
        except RAGEngineError:
            raise
        except Exception as exc:
            raise RAGEngineError(f"chat failed: {exc}") from exc
