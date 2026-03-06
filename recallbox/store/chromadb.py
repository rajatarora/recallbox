"""ChromaDB-backed memory store for embeddings.

This module provides a persistent vector store using ChromaDB, with support
for adding documents, querying by similarity, and deleting by source.

Usage:
    store = MemoryStore(Path("./data/memories"), embed_client)
    await store.add_documents([Document(content="Hello", metadata={"source": "test"})])
    results = await store.query("Hello world", top_k=5)
    await store.delete_by_source("test")
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, cast

import chromadb
# Settings import removed; using default client settings

from recallbox.llm.client import OpenRouterClient

logger = logging.getLogger(__name__)


class Document(tuple):
    """A document with content and metadata.

    Attributes:
        content: The text content of the document.
        metadata: A mapping of metadata key-value pairs.
    """

    __slots__ = ()

    def __new__(cls, content: str, metadata: Mapping[str, Any]) -> Document:
        return super().__new__(cls, (content, metadata))

    @property
    def content(self) -> str:
        return cast(str, self[0])

    @property
    def metadata(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self[1])


class MemoryStore:
    """A persistent ChromaDB-backed memory store for embeddings."""

    _COLLECTION_NAME = "memories"
    _REBUILD_THRESHOLD = 1000

    def __init__(
        self,
        persist_dir: Path,
        embed_client: OpenRouterClient,
    ) -> None:
        """Initialize the memory store.

        Args:
            persist_dir: Directory to persist the ChromaDB data.
            embed_client: Client for generating embeddings via OpenRouter.
        """
        self._persist_dir = persist_dir
        self._embed_client = embed_client
        self._new_vec_counter = 0

        persist_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(persist_dir), 0o700)

        # chromadb's client may not have precise stubs; cast to Any to satisfy mypy
        self._client = cast(Any, chromadb.PersistentClient(path=str(persist_dir)))
        # The collection API is dynamic; mypy would otherwise complain about attr-defined
        # Keep a narrow ignore only where necessary.
        # The chromadb client is dynamic; accept the runtime check without mypy ignore
        self._collection = self._client.get_or_create_collection(name=self._COLLECTION_NAME)

    async def add_documents(self, docs: List[Document]) -> None:
        """Add documents to the store.

        If a document with the same content already exists, it will be overwritten.

        Args:
            docs: List of documents to add.
        """
        if not docs:
            return

        seen_ids: set[str] = set()
        unique_docs: List[Document] = []
        unique_ids: List[str] = []

        for doc in docs:
            doc_id = self._generate_id(doc.content)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
                unique_ids.append(doc_id)

        contents = [doc.content for doc in unique_docs]
        metadatas: List[dict[str, Any]] = [dict(doc.metadata) for doc in unique_docs]

        await asyncio.to_thread(
            self._collection.upsert,
            ids=unique_ids,
            documents=contents,
            metadatas=metadatas,
        )

        self._new_vec_counter += len(unique_docs)
        await self._maybe_rebuild_index()

    async def add_memory(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        """Embed a single text memory and store it.

        This convenience method creates a deterministic ID from the text, obtains an
        embedding via the injected :class:`OpenRouterClient`, and stores the record in
        the Chroma collection.

        Args:
            text: The raw memory text to embed and store.
            metadata: Optional metadata mapping; defaults to an empty dict.
        """
        meta: Mapping[str, Any] = {} if metadata is None else metadata
        # Generate deterministic ID based on content
        doc_id = self._generate_id(text)
        # Obtain embedding vector
        vector = await self._embed_client.embed([text])
        embedding = vector[0].tolist()
        await asyncio.to_thread(
            self._collection.add,
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[dict(meta)],
        )
        self._new_vec_counter += 1
        await self._maybe_rebuild_index()

    async def query(self, query: str, top_k: int) -> List[Document]:
        """Query the store for similar documents.

        Args:
            query: The query string to search for.
            top_k: Number of results to return.

        Returns:
            List of documents sorted by similarity (highest first).
        """
        vectors = await self._embed_client.embed([query])
        query_vector = vectors[0]

        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
        )

        documents: List[Document] = []
        docs_list = results.get("documents")
        metadatas_list = results.get("metadatas")

        if docs_list is None or metadatas_list is None:
            return documents

        docs = docs_list[0] if docs_list else []
        metas = metadatas_list[0] if metadatas_list else []

        for content, metadata in zip(docs, metas):
            if content is not None:
                documents.append(Document(content=content, metadata=metadata or {}))

        return documents

    async def delete_by_source(self, source: str) -> None:
        """Delete all documents from a specific source.

        Args:
            source: The source identifier to delete.
        """
        await asyncio.to_thread(
            self._collection.delete,
            where={"source": source},
        )

    async def _maybe_rebuild_index(self) -> None:
        """Check if index rebuild is needed and trigger if threshold exceeded."""
        if self._new_vec_counter > self._REBUILD_THRESHOLD:
            self._new_vec_counter = 0
            logger.info(
                "Triggering background index rebuild",
                extra={"component": "memory"},
            )
            asyncio.create_task(self._rebuild_index())

    async def _rebuild_index(self) -> None:
        """Rebuild the HNSW index by triggering a persist operation."""
        try:
            await asyncio.to_thread(self._client.persist)
        except AttributeError:
            logger.debug("Persist method not available on this ChromaDB client version")

    def _generate_id(self, content: str) -> str:
        """Generate a deterministic ID based on document content.

        Args:
            content: The document content.

        Returns:
            A hex string SHA-256 hash of the content.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
