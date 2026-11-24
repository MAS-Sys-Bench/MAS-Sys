"""Long-term memory with embedding-based semantic retrieval.

Stores text entries as embedding vectors and retrieves the most relevant
entries via cosine similarity.  Uses the OpenAI ``text-embedding-3-small``
model by default, and pure-numpy vector operations — no heavy dependencies
such as FAISS or Chroma are required.

Supports:
- Batch embedding via ``store_batch()`` (single API call for multiple texts).
- Disk persistence via ``save()`` / ``load()`` (JSON metadata + .npy vectors).
- Configurable ``dimensions`` for models that support it (e.g. text-embedding-v3).
- Reads ``[memory]`` config section from config.toml when available.

Design references:
- MARBLE ``LongTermMemory``: embedding storage + cosine-similarity top-k retrieval.
- MetaGPT ``MemoryStorage``: FAISS-backed vector store (we use lighter numpy instead).
- ResearchTown ``LocalDatabaseClient``: numpy cosine search over Contriever embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, PrivateAttr

from app.config import config

try:
    from app.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MemoryEntry(BaseModel):
    """A single entry stored in long-term memory."""

    content: str
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Embedding vector (stored as list for JSON serialization).",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LongTermMemory(BaseModel):
    """Embedding-based long-term memory for agent context retrieval.

    Entries are stored with their embedding vectors.  Retrieval computes
    cosine similarity between a query embedding and all stored embeddings,
    returning the top-k most relevant entries.

    Usage example::

        ltm = LongTermMemory()
        await ltm.store("The capital of France is Paris.", metadata={"source": "user"})
        results = await ltm.retrieve("What is the capital of France?", top_k=3)
    """

    # configuration
    entries: List[MemoryEntry] = Field(default_factory=list)
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )

    class Config:
        arbitrary_types_allowed = True

    # cached numpy matrix (not serialized)
    _embedding_matrix: Optional[np.ndarray] = PrivateAttr(default=None)  # shape: (n_entries, dim)
    _client: Optional[AsyncOpenAI] = PrivateAttr(default=None)

    # public API

    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Compute embedding for *content* and store it."""
        if not content.strip():
            return

        embedding = await self._get_embedding(content)
        if embedding is None:
            logger.warning("LongTermMemory: failed to get embedding, storing without vector.")
            self.entries.append(MemoryEntry(content=content, metadata=metadata or {}))
            return

        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        self._embedding_matrix = None
        logger.debug(f"LongTermMemory: stored entry ({len(self.entries)} total).")

    async def store_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 10,
    ) -> int:
        """Embed and store multiple texts efficiently using batched API calls.

        Args:
            texts: List of texts to embed and store.
            metadatas: Optional list of metadata dicts (same length as texts).
            batch_size: Number of texts per API call (max varies by provider).

        Returns:
            Number of entries successfully stored.
        """
        if not texts:
            return 0
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("metadatas must have the same length as texts")

        stored = 0
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_metas = metadatas[start:start + batch_size] if metadatas else [{}] * len(batch_texts)

            embeddings = await self._get_embeddings_batch(batch_texts)

            for text, emb, meta in zip(batch_texts, embeddings, batch_metas):
                entry = MemoryEntry(
                    content=text,
                    embedding=emb,
                    metadata=meta or {},
                )
                self.entries.append(entry)
                stored += 1

        self._embedding_matrix = None
        logger.info(f"LongTermMemory: batch stored {stored} entries ({len(self.entries)} total).")
        return stored

    def save(self, path: str | Path) -> None:
        """Persist entries to disk (JSON metadata + .npy embedding matrix).

        Creates two files:
          - ``{path}.json`` — list of {content, metadata} dicts
          - ``{path}.npy`` — numpy array of shape (n, dim)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta_path = path.with_suffix(".json")
        npy_path = path.with_suffix(".npy")

        records = []
        vectors = []
        for e in self.entries:
            records.append({"content": e.content, "metadata": e.metadata})
            if e.embedding is not None:
                vectors.append(e.embedding)
            else:
                vectors.append([])

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)

        if vectors and any(v for v in vectors):
            mat = np.array([v for v in vectors if v], dtype=np.float32)
            np.save(npy_path, mat)
        logger.info(f"LongTermMemory: saved {len(records)} entries to {meta_path}")

    @classmethod
    def load(cls, path: str | Path) -> "LongTermMemory":
        """Load a previously saved LongTermMemory from disk.

        Args:
            path: Base path (without extension). Expects .json and .npy files.

        Returns:
            A new LongTermMemory instance with all entries restored.
        """
        path = Path(path)
        meta_path = path.with_suffix(".json")
        npy_path = path.with_suffix(".npy")

        if not meta_path.exists():
            raise FileNotFoundError(f"Memory metadata not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        mat = None
        if npy_path.exists():
            mat = np.load(npy_path)

        entries = []
        for idx, rec in enumerate(records):
            emb = mat[idx].tolist() if mat is not None and idx < len(mat) else None
            entries.append(MemoryEntry(
                content=rec["content"],
                embedding=emb,
                metadata=rec.get("metadata", {}),
            ))

        ltm = cls(entries=entries)
        logger.info(f"LongTermMemory: loaded {len(entries)} entries from {meta_path}")
        return ltm

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant entries for *query*.

        Args:
            query: The query text to match against stored entries.
            top_k: Maximum number of results to return.
            min_similarity: Minimum cosine similarity threshold (0.0–1.0).

        Returns:
            List of dicts with keys ``content``, ``metadata``, ``similarity``.
        """
        if not self.entries:
            return []

        query_embedding = await self._get_embedding(query)
        if query_embedding is None:
            logger.warning("LongTermMemory: failed to embed query; returning empty.")
            return []

        return self._search(query_embedding, top_k, min_similarity)

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Return all stored entries (without similarity scoring)."""
        return [
            {"content": e.content, "metadata": e.metadata}
            for e in self.entries
        ]

    @property
    def count(self) -> int:
        """Number of stored entries."""
        return len(self.entries)

    def clear(self) -> None:
        """Remove all entries."""
        self.entries.clear()
        self._embedding_matrix = None

    # internal helpers

    def _build_matrix(self) -> Optional[np.ndarray]:
        """(Re-)build the cached (n, dim) numpy matrix from stored embeddings."""
        vectors = [e.embedding for e in self.entries if e.embedding is not None]
        if not vectors:
            return None
        mat = np.array(vectors, dtype=np.float32)
        self._embedding_matrix = mat
        return mat

    def _search(
        self,
        query_vec: List[float],
        top_k: int,
        min_similarity: float,
    ) -> List[Dict[str, Any]]:
        """Cosine-similarity search using numpy.

        Steps:
        1. L2-normalize query vector and all entry vectors.
        2. Compute dot product (= cosine similarity for unit vectors).
        3. Return top-k entries above *min_similarity*.
        """
        mat = self._embedding_matrix
        if mat is None:
            mat = self._build_matrix()
        if mat is None:
            return []

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        # L2-normalize rows of the matrix
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        mat_normed = mat / norms

        similarities = mat_normed @ q  # shape: (n,)

        # Indices sorted by descending similarity
        sorted_idx = np.argsort(similarities)[::-1]

        results: List[Dict[str, Any]] = []
        for idx in sorted_idx[:top_k]:
            sim = float(similarities[idx])
            if sim < min_similarity:
                break
            entry = self.entries[idx]
            results.append(
                {
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "similarity": round(sim, 4),
                }
            )
        return results

    def _get_client(self) -> AsyncOpenAI:
        """Lazily create and cache the OpenAI client.

        Reads ``[memory]`` config section first and requires it to be configured.
        Also updates ``self.embedding_model`` from config if not explicitly set.
        """
        if self._client is not None:
            return self._client

        mem_cfg = config.memory
        if mem_cfg and mem_cfg.base_url:
            client = AsyncOpenAI(
                api_key=mem_cfg.api_key,
                base_url=mem_cfg.base_url,
            )
            if mem_cfg.model:
                self.embedding_model = mem_cfg.model
        else:
            raise ValueError(
                "LongTermMemory requires a configured [memory] section with at least "
                "'base_url' in config.toml; fallback to [llm] is no longer supported."
            )

        self._client = client
        return client

    def _get_dimensions(self) -> Optional[int]:
        """Return configured embedding dimensions, or None."""
        mem_cfg = config.memory
        if mem_cfg and mem_cfg.dimensions:
            return mem_cfg.dimensions
        return None

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Call OpenAI embeddings API for a single text."""
        try:
            client = self._get_client()
            kwargs: Dict[str, Any] = {
                "model": self.embedding_model,
                "input": text.strip(),
            }
            dims = self._get_dimensions()
            if dims:
                kwargs["dimensions"] = dims
            response = await client.embeddings.create(**kwargs)
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"LongTermMemory: embedding API call failed: {e}")
            return None

    async def _get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Call OpenAI embeddings API for multiple texts in a single request."""
        try:
            client = self._get_client()
            cleaned = [t.strip() for t in texts]
            kwargs: Dict[str, Any] = {
                "model": self.embedding_model,
                "input": cleaned,
            }
            dims = self._get_dimensions()
            if dims:
                kwargs["dimensions"] = dims
            response = await client.embeddings.create(**kwargs)
            # Response data is sorted by index
            result: List[Optional[List[float]]] = [None] * len(texts)
            for item in response.data:
                result[item.index] = item.embedding
            return result
        except Exception as e:
            logger.warning(f"LongTermMemory: batch embedding API call failed: {e}, falling back to individual calls")
            # Fallback: try each text individually so one failure doesn't lose the whole batch
            results: List[Optional[List[float]]] = []
            for t in texts:
                emb = await self._get_embedding(t)
                results.append(emb)
            return results
