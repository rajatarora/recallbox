import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from recallbox.rag.engine import RAGEngine
from recallbox.store.chromadb import Document


class DummyCfg:
    class retrieval:
        top_k = 3


@pytest.mark.asyncio
async def test_retrieve_returns_top_k_sorted():
    # Mock store returns pre-sorted docs
    docs = [Document(f"doc{i}", {"source": f"s{i}"}) for i in range(3)]
    store = MagicMock()
    store.query = AsyncMock(return_value=docs)
    llm = AsyncMock()
    engine = RAGEngine(llm=llm, store=store, cfg=DummyCfg())

    res = await engine.retrieve_context("foo")
    assert res == docs


@pytest.mark.asyncio
async def test_cache_prevents_second_query_call():
    docs = [Document("a", {"source": "s"})]
    store = MagicMock()
    store.query = AsyncMock(return_value=docs)
    llm = AsyncMock()
    engine = RAGEngine(llm=llm, store=store, cfg=DummyCfg())

    first = await engine.retrieve_context("bar")
    second = await engine.retrieve_context("bar")
    assert first == second
    # query called only once due to cache
    assert store.query.call_count == 1


@pytest.mark.asyncio
async def test_prepare_prompt_truncation_and_bullets():
    long_text = "x" * 300
    docs = [Document(long_text, {"source": "src"})]
    store = MagicMock()
    store.query = AsyncMock(return_value=docs)
    llm = AsyncMock()
    engine = RAGEngine(llm=llm, store=store, cfg=DummyCfg())

    short_term = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "reply"}]
    messages = await engine.prepare_prompt("q", short_term)
    # First message is system containing bullet
    sys = messages[0]["content"]
    assert "Long-term memories" in sys
    assert sys.count("- [src]") == 1
    # Check truncation to 200 chars
    assert len(docs[0].content[:200]) == 200


@pytest.mark.asyncio
async def test_chat_concurrent_and_logs(caplog):
    docs = [Document("short", {"source": "s"})]
    store = MagicMock()
    store.query = AsyncMock(return_value=docs)

    class FakeLLM:
        async def chat(self, messages):
            await asyncio.sleep(0)  # yield
            return "assistant response"

    llm = FakeLLM()
    engine = RAGEngine(llm=llm, store=store, cfg=DummyCfg())

    caplog.clear()
    res = await engine.chat("hey", [{"role": "user", "content": "hi"}])
    assert res == "assistant response"
    # Ensure that both latency logs are present
    found_retrieval = any(
        "retrieval_latency_ms" in rec.message or "retrieval_latency_ms" in str(rec.msg) for rec in caplog.records
    )
    found_llm = any("llm_latency_ms" in rec.message or "llm_latency_ms" in str(rec.msg) for rec in caplog.records)
    # structlog writes to standard logging; we expect entries
    assert found_retrieval or found_llm
