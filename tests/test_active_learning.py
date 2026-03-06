import asyncio
from typing import Any

import pytest

from recallbox.active_learning import evaluate_and_store, _heuristic_extract
from recallbox.store.chromadb import Document


class DummyStore:
    def __init__(self):
        self.added: list[list[Document]] = []

    async def add_documents(self, docs: list[Document]) -> None:
        # Simulate small I/O latency
        await asyncio.sleep(0)
        self.added.append(docs)


class DummyLLM:
    def __init__(self, resp: Any = None, exc: Exception | None = None):
        self.resp = resp
        self.exc = exc

    async def evaluate_memory(self, user: str, assistant: str):
        await asyncio.sleep(0)
        if self.exc:
            raise self.exc
        return self.resp


@pytest.mark.asyncio
async def test_llm_returns_memory_adds_document():
    store = DummyStore()
    llm = DummyLLM(resp=(True, "Alice moved to Paris in 2020"))

    await evaluate_and_store("Where did Alice live?", "Alice moved to Paris in 2020", store, llm)

    assert len(store.added) == 1
    docs = store.added[0]
    assert len(docs) == 1
    assert docs[0].content == "Alice moved to Paris in 2020"
    assert docs[0].metadata["source"] == "ai_extracted"


@pytest.mark.asyncio
async def test_llm_returns_no_memory_no_add():
    store = DummyStore()
    llm = DummyLLM(resp=(False, ""))

    await evaluate_and_store("Hello", "Hi there", store, llm)

    assert len(store.added) == 0


@pytest.mark.asyncio
async def test_malformed_json_runs_heuristic_and_may_store():
    store = DummyStore()

    class BadJSONError(RuntimeError):
        pass

    # Case: heuristic finds a sentence
    assistant = "I remember that Bob likes skiing. Some other text."
    llm = DummyLLM(exc=BadJSONError("bad json"))
    # Sanity-check the heuristic directly

    h = _heuristic_extract("I remember that Bob likes skiing.", assistant)
    assert h is not None, "Heuristic failed in isolation"

    # Make user similar enough to satisfy the similarity threshold
    await evaluate_and_store("I remember that Bob likes skiing.", assistant, store, llm)
    # heuristic should store
    assert len(store.added) == 1

    # Reset store and test negative heuristic
    store = DummyStore()
    assistant2 = "This is unrelated content."  # no 'I remember'
    llm2 = DummyLLM(exc=BadJSONError("bad json"))
    await evaluate_and_store("Something else", assistant2, store, llm2)
    assert len(store.added) == 0


@pytest.mark.asyncio
async def test_evaluate_and_store_performance():
    """Ensure the evaluation path completes in under 500ms with mocked LLM/store."""
    store = DummyStore()
    llm = DummyLLM(resp=(True, "Quick memory summary"))

    import time

    t0 = time.perf_counter()
    await evaluate_and_store("Quick check", "Quick check", store, llm)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"Evaluation took too long: {elapsed}s"
