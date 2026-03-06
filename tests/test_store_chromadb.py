import hashlib
import pytest

from pathlib import Path

from recallbox.store.chromadb import MemoryStore, Document


class FakeCollection:
    def __init__(self):
        self.upserts = []
        self.adds = []
        self.queries = []
        self.deletes = []

    def upsert(self, ids, documents, metadatas):
        self.upserts.append((ids, documents, metadatas))

    def add(self, ids, documents, embeddings, metadatas):
        self.adds.append((ids, documents, embeddings, metadatas))

    def query(self, query_embeddings, n_results):
        self.queries.append((query_embeddings, n_results))
        # Return format: documents: [[...]], metadatas: [[...]]
        return {"documents": [["doc1"]], "metadatas": [[{"source": "s"}]]}

    def delete(self, where):
        self.deletes.append(where)


class FakeClient:
    def __init__(self, collection=None, persist_available=True):
        self._collection = collection or FakeCollection()
        self.persist_available = persist_available

    def get_or_create_collection(self, name=None):
        return self._collection

    def persist(self):
        if not self.persist_available:
            raise AttributeError


@pytest.fixture
def fake_chromadb(monkeypatch, tmp_path):
    coll = FakeCollection()

    def fake_persistent_client(path):
        return FakeClient(collection=coll)

    monkeypatch.setattr("recallbox.store.chromadb.chromadb.PersistentClient", fake_persistent_client)
    return coll


@pytest.mark.asyncio
async def test_add_documents_deduplicate(tmp_path, fake_chromadb):
    from recallbox.llm.client import OpenRouterClient

    # create dummy embed client but embed is not used by add_documents
    client = OpenRouterClient(
        api_key="x", embedding_model="e", chat_model="c", memory_prompt_path=str(tmp_path / "p"), base_url="https://api"
    )
    store = MemoryStore(tmp_path / "data", client)
    docs = [Document("a", {"source": "x"}), Document("a", {"source": "x"}), Document("b", {"source": "y"})]
    await store.add_documents(docs)
    # ensure upsert was called with 2 unique ids
    assert len(fake_chromadb.upserts) == 1
    ids, documents, metadatas = fake_chromadb.upserts[0]
    assert len(ids) == 2


@pytest.mark.asyncio
async def test_add_memory_and_query_and_delete(tmp_path, monkeypatch, fake_chromadb):
    # fake embed client that returns a vector
    class EC:
        async def embed(self, texts):
            # return numpy-like object with tolist
            class V:
                def __init__(self, arr):
                    self._arr = arr

                def tolist(self):
                    return list(self._arr)

            return [V([0.0, 1.0]) for _ in texts]

    ec = EC()
    store = MemoryStore(tmp_path / "data2", ec)
    await store.add_memory("hello", {"source": "s"})
    assert len(fake_chromadb.adds) == 1
    # query
    res = await store.query("hi", top_k=1)
    assert len(res) == 1
    assert res[0].content == "doc1"
    # delete
    await store.delete_by_source("s")
    assert fake_chromadb.deletes


def test_generate_id():
    from recallbox.store.chromadb import MemoryStore

    # Use dummy client
    class EC:
        async def embed(self, texts):
            return [[0.0]]

    ms = MemoryStore(Path("/tmp/testmem"), EC())
    assert ms._generate_id("foo") == hashlib.sha256(b"foo").hexdigest()
