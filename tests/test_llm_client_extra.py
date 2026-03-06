import pytest

from recallbox.llm.client import OpenRouterClient, OpenRouterError, ChatError


def test_base_url_from_config_attr(monkeypatch):
    class C:
        openrouter_base_url = "https://example.org/"

    c = C()
    cl = OpenRouterClient(api_key="k", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", config=c)
    assert cl.base_url == "https://example.org"


def test_base_url_from_non_str_config(monkeypatch):
    # config doesn't have openrouter_base_url -> uses str(config)
    cfg = 12345
    cl = OpenRouterClient(api_key="k", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", config=cfg)
    assert cl.base_url == str(cfg).rstrip("/")


def test_base_url_fallback_get_config_exception(monkeypatch):
    # Cause import get_config to raise by monkeypatching recallbox.config.get_config
    import recallbox.config as cfgmod

    monkeypatch.setattr(cfgmod, "get_config", lambda: (_ for _ in ()).throw(Exception("nope")))
    cl = OpenRouterClient(api_key="k", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null")
    assert cl.base_url == "https://openrouter.ai/api/v1"


class FakeResp:
    def __init__(self, status_code=200, headers=None, text="", json_data=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self._json = json_data or {}

    def json(self):
        return self._json


class FakeAsyncClient:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method, url, json=None, headers=None):
        return self._resp




@pytest.mark.asyncio
async def test_request_429_no_retry_after_raises(monkeypatch):
    def factory(*args, **kwargs):
        return FakeAsyncClient(FakeResp(status_code=429, headers={}, text="rl"))

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", factory)
    cl = OpenRouterClient(
        api_key="k", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", max_retries=0
    )
    with pytest.raises(OpenRouterError):
        await cl._request_with_retry("GET", "/x", {})


@pytest.mark.asyncio
async def test_request_400_raises(monkeypatch):
    def factory(*a, **kw):
        return FakeAsyncClient(FakeResp(status_code=400, text="bad"))

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", factory)
    cl = OpenRouterClient(
        api_key="k", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", max_retries=0
    )
    with pytest.raises(OpenRouterError):
        await cl._request_with_retry("GET", "/x", {})


@pytest.mark.asyncio
async def test_retry_after_parse_fallback(monkeypatch, no_sleep):
    # First response 429 with non-float Retry-After, second 200
    seq = [
        FakeResp(status_code=429, headers={"Retry-After": "notfloat"}, text="r"),
        FakeResp(status_code=200, json_data={"ok": True}),
    ]

    def factory(*a, **kw):
        return FakeAsyncClient(seq.pop(0))

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", factory)
    cl = OpenRouterClient(
        api_key="k", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null", base_retry_wait=0.0
    )
    res = await cl._request_with_retry("GET", "/x", {})
    assert res == {"ok": True}


@pytest.mark.asyncio
async def test_chat_no_content_raises(monkeypatch):
    # choices with message but content missing
    def factory(*a, **kw):
        return FakeAsyncClient(FakeResp(status_code=200, json_data={"choices": [{"message": {}}]}))

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", factory)
    cl = OpenRouterClient(api_key="k", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null")
    with pytest.raises(ChatError):
        await cl.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_chat_nonserializable_content_returns_str(monkeypatch):
    class Bad:
        def __str__(self):
            return "BAD"

    def factory(*a, **kw):
        return FakeAsyncClient(FakeResp(status_code=200, json_data={"choices": [{"message": {"content": Bad()}}]}))

    monkeypatch.setattr("recallbox.llm.client.httpx.AsyncClient", factory)
    cl = OpenRouterClient(api_key="k", embedding_model="e", chat_model="c", memory_prompt_path="/dev/null")
    out = await cl.chat([{"role": "user", "content": "hi"}])
    assert out == "BAD"


@pytest.mark.asyncio
async def test_evaluate_memory_format_failure(monkeypatch, tmp_path):
    # Patch Path.read_text to return a non-str so .replace fails
    from pathlib import Path

    Path.read_text

    def fake_read_text(self, encoding="utf-8"):
        return object()

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    cl = OpenRouterClient(api_key="k", embedding_model="e", chat_model="c", memory_prompt_path=str(tmp_path / "p"))
    with pytest.raises(Exception):
        await cl.evaluate_memory("u", "a")
