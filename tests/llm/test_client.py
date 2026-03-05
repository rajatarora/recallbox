import asyncio
import json
import time
from typing import Any

import httpx
import numpy as np
import pytest

from recallbox.llm.client import (
    MemoryEvaluationError,
    OpenRouterClient,
)


class DelayedMockTransport(httpx.MockTransport):
    def __init__(self, func, delay: float = 0.0):
        super().__init__(func)
        self._delay = delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._delay:
            await asyncio.sleep(self._delay)
        return await super().handle_async_request(request)


@pytest.mark.asyncio
async def test_embeddings_success_and_normalization(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else {}
        inputs = body.get("input")
        data = [{"embedding": [1.0, 0.0, 0.0]} for _ in inputs]
        return httpx.Response(200, json={"data": data})

    transport = DelayedMockTransport(handler, delay=0.1)
    original_ac = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_ac(transport=transport, *args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    c = OpenRouterClient("key", "emb-model", "chat-model")
    start = time.time()
    vecs = await c.embed(["a", "b"])
    elapsed = time.time() - start
    assert elapsed < 0.5
    assert len(vecs) == 2
    for v in vecs:
        assert isinstance(v, np.ndarray)
        assert np.isclose(np.linalg.norm(v), 1.0)


@pytest.mark.asyncio
async def test_embeddings_retry_on_5xx(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 3:
            return httpx.Response(502, json={"error": "bad"})
        data = [{"embedding": [1.0, 0.0]}]
        return httpx.Response(200, json={"data": data})

    transport = DelayedMockTransport(handler, delay=0.0)
    original_ac = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_ac(transport=transport, *args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    c = OpenRouterClient("key", "emb-model", "chat-model", max_retries=3, base_retry_wait=0.01)
    await c.embed(["x"])
    assert calls["n"] == 4


@pytest.mark.asyncio
async def test_chat_retry_after_respected(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true, "explanation": "ok"}'}}]})

    transport = DelayedMockTransport(handler, delay=0.0)
    original_ac = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_ac(transport=transport, *args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    c = OpenRouterClient("key", "emb", "chat")
    start = time.time()
    res = await c.chat([{"role": "user", "content": "hello"}])
    elapsed = time.time() - start
    # Should have waited about 1 second due to Retry-After
    assert elapsed >= 1.0
    assert json.loads(res)["ok"] is True


@pytest.mark.asyncio
async def test_evaluate_memory_bad_json(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not a json"}}]})

    transport = DelayedMockTransport(handler, delay=0.0)
    original_ac = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original_ac(transport=transport, *args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    c = OpenRouterClient("k", "emb", "chat")
    with pytest.raises(MemoryEvaluationError):
        await c.evaluate_memory("u", "a")
