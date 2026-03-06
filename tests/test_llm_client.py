import json
import httpx
import numpy as np
import pytest

from recallbox.llm.client import (
    OpenRouterClient,
    OpenRouterError,
    EmbeddingError,
    ChatError,
    MemoryEvaluationError,
)


class FakeResp:
    def __init__(self, status_code=200, headers=None, text="", json_data=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self._json = json_data or {}

    def json(self):
        return self._json


class FakeAsyncClientCM:
    def __init__(self, resp: FakeResp, raise_on_request: Exception | None = None):
        self._resp = resp
        self._raise = raise_on_request

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method, url, json=None, headers=None):
        if self._raise:
            raise self._raise
        return self._resp


@pytest.mark.asyncio
async def test_request_retry_429(monkeypatch, no_sleep):
    # First response 429 with Retry-After, second 200
    r1 = FakeResp(status_code=429, headers={"Retry-After": "0"}, text="ratelimit")
    r2 = FakeResp(status_code=200, json_data={"ok": True})
    seq = [r1, r2]

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClientCM(seq.pop(0))

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", fake_client_factory)
    client = OpenRouterClient(
        api_key="x", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", base_url="https://api"
    )
    resp = await client._request_with_retry("GET", "/test", {})
    assert resp == {"ok": True}


@pytest.mark.asyncio
async def test_request_500_then_fail(monkeypatch, no_sleep):
    r500 = FakeResp(status_code=500, text="boom")

    # Only server errors
    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClientCM(r500)

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", fake_client_factory)
    client = OpenRouterClient(
        api_key="x",
        embedding_model="e",
        chat_model="c",
        memory_prompt_path="/dev/null",
        base_url="https://api",
        max_retries=1,
    )
    with pytest.raises(OpenRouterError):
        await client._request_with_retry("GET", "/test", {})


@pytest.mark.asyncio
async def test_request_http_error(monkeypatch, no_sleep):
    # Simulate httpx.HTTPError being raised
    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClientCM(None, raise_on_request=httpx.HTTPError("boom"))

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", fake_client_factory)
    client = OpenRouterClient(
        api_key="x",
        embedding_model="e",
        chat_model="c",
        memory_prompt_path="/dev/null",
        base_url="https://api",
        max_retries=0,
    )
    with pytest.raises(OpenRouterError):
        await client._request_with_retry("GET", "/test", {})


@pytest.mark.asyncio
async def test_embed_success(monkeypatch, no_sleep):
    # Return two embeddings
    emb = [[0.0, 3.0], [4.0, 0.0]]
    r = FakeResp(status_code=200, json_data={"data": [{"embedding": emb[0]}, {"embedding": emb[1]}]})

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClientCM(r)

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", fake_client_factory)
    client = OpenRouterClient(
        api_key="x", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", base_url="https://api"
    )
    vecs = await client.embed(["a", "b"])
    assert len(vecs) == 2
    # Check normalization ~ length 1
    for v in vecs:
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_embed_errors(monkeypatch, no_sleep):
    client = OpenRouterClient(
        api_key="x", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", base_url="https://api"
    )
    # empty
    assert await client.embed([]) == []
    # too many
    with pytest.raises(EmbeddingError):
        await client.embed(["x"] * 65)


@pytest.mark.asyncio
async def test_embed_unexpected_response(monkeypatch, no_sleep):
    r = FakeResp(status_code=200, json_data={"data": "notalist"})

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClientCM(r)

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", fake_client_factory)
    client = OpenRouterClient(
        api_key="x", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", base_url="https://api"
    )
    with pytest.raises(EmbeddingError):
        await client.embed(["a"])


@pytest.mark.asyncio
async def test_chat_variants(monkeypatch, no_sleep):
    # choice with dict message
    r1 = FakeResp(status_code=200, json_data={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
    # choice with text
    r2 = FakeResp(status_code=200, json_data={"choices": [{"text": "plain"}]})
    # missing choices
    r3 = FakeResp(status_code=200, json_data={})

    seq = [r1, r2, r3]

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClientCM(seq.pop(0))

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", fake_client_factory)
    client = OpenRouterClient(
        api_key="x", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", base_url="https://api"
    )
    out1 = await client.chat([{"role": "user", "content": "hi"}])
    assert out1 == "ok"
    out2 = await client.chat([{"role": "user", "content": "hi"}])
    assert out2 == "plain"
    with pytest.raises(ChatError):
        await client.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_chat_nonstring_content(monkeypatch, no_sleep):
    r = FakeResp(status_code=200, json_data={"choices": [{"message": {"content": {"ok": True}}}]})

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClientCM(r)

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", fake_client_factory)
    client = OpenRouterClient(
        api_key="x", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", base_url="https://api"
    )
    out = await client.chat([{"role": "user", "content": "hi"}])
    # Should be JSON string
    assert isinstance(out, str)
    assert json.loads(out) == {"ok": True}


@pytest.mark.asyncio
async def test_evaluate_memory_file_missing(monkeypatch, tmp_path):
    # memory prompt path missing
    client = OpenRouterClient(
        api_key="x",
        embedding_model="e",
        chat_model="c",
        memory_prompt_path=str(tmp_path / "nope"),
        base_url="https://api",
    )
    with pytest.raises(MemoryEvaluationError):
        await client.evaluate_memory("u", "a")


@pytest.mark.asyncio
async def test_evaluate_memory_parse(monkeypatch, tmp_path):
    # Create prompt file
    p = tmp_path / "prompt.txt"
    p.write_text("{user}|||{assistant}")

    # Make chat return a JSON string
    resp_text = json.dumps({"ok": True, "explanation": "fine"})
    r = FakeResp(status_code=200, json_data={"choices": [{"text": resp_text}]})

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClientCM(r)

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", fake_client_factory)
    client = OpenRouterClient(
        api_key="x", embedding_model="e", chat_model="c", memory_prompt_path=str(p), base_url="https://api"
    )
    ok, explanation = await client.evaluate_memory("u", "a")
    assert ok is True
    assert explanation == "fine"
