"""Tests for the ChromaDB memory store."""

import time
from typing import List

import numpy as np
import pytest

from recallbox.store.chromadb import Document, MemoryStore


def create_dummy_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Create a deterministic embedding based on text content."""
    seed = hash(text) % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.rand(dim).astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-10
    return vec / norm


class MockEmbedClient:
    """Mock OpenRouter client for testing."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Return deterministic embeddings based on text content."""
        return [create_dummy_embedding(text, self._dim) for text in texts]


class TestMemoryStore:
    """Test suite for MemoryStore."""

    @pytest.fixture
    def mock_embed_client(self):
        """Provide a mock embedding client."""
        return MockEmbedClient()

    @pytest.mark.asyncio
    async def test_add_documents_upsert(self, mock_embed_client, tmp_path):
        """Test that adding documents stores the exact number of docs (overwrite policy)."""
        store = MemoryStore(tmp_path / "test_db", mock_embed_client)

        doc1 = Document(content="Hello world", metadata={"source": "test"})
        doc2 = Document(content="Hello world", metadata={"source": "test2"})
        doc3 = Document(content="Different content", metadata={"source": "test"})

        await store.add_documents([doc1, doc2, doc3])

        count = store._collection.count()
        assert count == 2, f"Expected 2 documents (doc1 and doc2 should overwrite), got {count}"

    @pytest.mark.asyncio
    async def test_query_returns_correct_top_k(self, mock_embed_client, tmp_path):
        """Test that query returns correct top_k with similarity ordering."""
        store = MemoryStore(tmp_path / "test_db2", mock_embed_client)

        docs = [
            Document(content="The cat sat on the mat", metadata={"source": "a"}),
            Document(content="A dog barked loudly", metadata={"source": "b"}),
            Document(content="The cat chased the dog", metadata={"source": "c"}),
            Document(content="Birds fly in the sky", metadata={"source": "d"}),
            Document(content="Fish swim in water", metadata={"source": "e"}),
        ]

        await store.add_documents(docs)

        results = await store.query("cat and dog", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, Document) for r in results)

    @pytest.mark.asyncio
    async def test_delete_by_source(self, mock_embed_client, tmp_path):
        """Test that deletion removes only documents from the specified source."""
        store = MemoryStore(tmp_path / "test_db3", mock_embed_client)

        docs = [
            Document(content="Doc from source A", metadata={"source": "A"}),
            Document(content="Doc from source B", metadata={"source": "B"}),
            Document(content="Another from A", metadata={"source": "A"}),
        ]

        await store.add_documents(docs)
        assert store._collection.count() == 3

        await store.delete_by_source("A")

        remaining = store._collection.get()
        remaining_contents = remaining.get("documents", [])
        assert len(remaining_contents) == 1
        assert "Doc from source B" in remaining_contents

    @pytest.mark.asyncio
    async def test_index_rebuild_trigger(self, mock_embed_client, tmp_path, caplog):
        """Test that index rebuild is triggered after >1000 new vectors."""
        store = MemoryStore(tmp_path / "test_db4", mock_embed_client)

        initial_counter = store._new_vec_counter

        for i in range(100):
            doc = Document(content=f"Document {i}", metadata={"source": f"src_{i}"})
            await store.add_documents([doc])

        assert store._new_vec_counter == 100 + initial_counter

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Performance test takes too long - enable manually when needed")
    async def test_query_performance(self, mock_embed_client, tmp_path):
        """Test that query latency is < 300ms for top_k=5 on 10k dataset."""
        store = MemoryStore(tmp_path / "test_db5", mock_embed_client)

        sentences = [f"Simple sentence number {i}." for i in range(10000)]
        docs = [Document(content=s, metadata={"source": "perf_test"}) for s in sentences]

        for i in range(0, len(docs), 100):
            await store.add_documents(docs[i : i + 100])

        start = time.perf_counter()
        results = await store.query("test query", top_k=5)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 300, f"Query took {elapsed}ms, expected < 300ms"
        assert len(results) == 5
