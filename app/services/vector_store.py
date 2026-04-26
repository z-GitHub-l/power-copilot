from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Protocol

try:
    import chromadb
    from chromadb.errors import NotFoundError
except Exception:  # pragma: no cover - optional dependency
    chromadb = None

    class NotFoundError(Exception):
        pass

from app.config import Settings
from app.schemas import DocumentChunk, SourceChunk

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Replaceable embedding interface for future real embedding services."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class SimpleHashEmbeddingProvider:
    """
    Lightweight local embedding stub.

    This keeps the MVP fully local and runnable on Windows.
    It can be replaced later without changing the VectorStore API.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            weight = 1.0 + min(len(token), 12) / 12.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens: list[str] = []
        latin_buffer: list[str] = []

        for char in lowered:
            if "\u4e00" <= char <= "\u9fff":
                if latin_buffer:
                    tokens.extend(re.findall(r"[a-z0-9_]+", "".join(latin_buffer)))
                    latin_buffer = []
                tokens.append(char)
            else:
                latin_buffer.append(char)

        if latin_buffer:
            tokens.extend(re.findall(r"[a-z0-9_]+", "".join(latin_buffer)))

        return tokens


class VectorStore:
    def __init__(
        self,
        settings: Settings,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings
        self.persist_dir = Path(self.settings.chroma_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_provider = embedding_provider or SimpleHashEmbeddingProvider()
        self.collection = None
        self.client = None
        self.fallback_store_path = self.persist_dir / "fallback_store.json"
        self._fallback_records: list[dict[str, Any]] = []
        self._use_chroma = False

        self._initialize_backend()

    def upsert_documents(self, chunks: list[dict]) -> int:
        normalized_chunks = self._normalize_chunks(chunks)
        if not normalized_chunks:
            return 0

        if not self._use_chroma:
            return self._fallback_upsert_documents(normalized_chunks)

        self._delete_existing_sources({chunk["source"] for chunk in normalized_chunks})
        collection = self._ensure_collection()

        ids = [chunk["id"] for chunk in normalized_chunks]
        documents = [chunk["text"] for chunk in normalized_chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in normalized_chunks
        ]
        embeddings = self.embedding_provider.embed_documents(documents)

        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except NotFoundError:
            collection = self._refresh_collection()
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        return len(normalized_chunks)

    def similarity_search(self, query: str, top_k: int = 4) -> list[dict]:
        results = self._query_collection(query=query, top_k=top_k)
        return [
            {
                "text": item["text"],
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "score": item["score"],
            }
            for item in results
        ]

    def reset_collection(self) -> None:
        if not self._use_chroma:
            self._fallback_records = []
            self._save_fallback_records()
            return

        try:
            self.client.delete_collection(name=self.settings.collection_name)
        except Exception:
            logger.debug(
                "Collection did not exist when resetting: %s",
                self.settings.collection_name,
            )

        self.collection = self._get_or_create_collection()

    def list_indexed_sources(self) -> set[str]:
        if not self._use_chroma:
            return {
                str(record.get("source"))
                for record in self._fallback_records
                if record.get("source")
            }

        collection = self._ensure_collection()
        try:
            total = collection.count()
        except NotFoundError:
            collection = self._refresh_collection()
            total = collection.count()

        if total == 0:
            return set()

        data = collection.get(limit=total, include=["metadatas"])
        metadatas = data.get("metadatas", []) or []
        return {
            str(metadata.get("source"))
            for metadata in metadatas
            if metadata and metadata.get("source")
        }

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        payload: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            chunk_index = chunk.metadata.get("chunk_index", index)
            payload.append(
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": int(chunk_index),
                }
            )
        return self.upsert_documents(payload)

    def query(self, query_text: str, top_k: int = 4) -> list[SourceChunk]:
        results = self._query_collection(query=query_text, top_k=top_k)
        return [
            SourceChunk(
                source=item["source"],
                chunk_id=item["id"],
                score=item["score"],
                content=item["text"],
            )
            for item in results
        ]

    def _initialize_backend(self) -> None:
        if chromadb is None:
            logger.info("chromadb is not installed, using JSON fallback vector store.")
            self._load_fallback_records()
            return

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self._get_or_create_collection()
            self._use_chroma = True
        except Exception as exc:  # pragma: no cover - runtime safety fallback
            logger.warning(
                "Failed to initialize chromadb, falling back to JSON storage: %s",
                exc,
            )
            self.client = None
            self.collection = None
            self._load_fallback_records()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _refresh_collection(self):
        self.collection = self._get_or_create_collection()
        return self.collection

    def _ensure_collection(self):
        try:
            self.collection.count()
        except NotFoundError:
            self.collection = self._get_or_create_collection()
        return self.collection

    def _normalize_chunks(self, chunks: list[dict]) -> list[dict[str, Any]]:
        normalized_chunks: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = str(chunk.get("id", "")).strip()
            text = str(chunk.get("text", "")).strip()
            source = str(chunk.get("source", "")).strip()

            if not chunk_id or not text or not source:
                logger.warning("Skip invalid chunk, required fields missing: %s", chunk)
                continue

            try:
                chunk_index = int(chunk.get("chunk_index", 0))
            except (TypeError, ValueError):
                logger.warning("Invalid chunk_index, fallback to 0: %s", chunk)
                chunk_index = 0

            normalized_chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source": source,
                    "chunk_index": chunk_index,
                }
            )

        return normalized_chunks

    def _delete_existing_sources(self, sources: set[str]) -> None:
        collection = self._ensure_collection()
        for source in sources:
            try:
                existing = collection.get(where={"source": source})
            except NotFoundError:
                collection = self._refresh_collection()
                existing = collection.get(where={"source": source})

            existing_ids = existing.get("ids", []) or []
            if existing_ids:
                collection.delete(ids=existing_ids)

    def _query_collection(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        if not self._use_chroma:
            return self._fallback_query_collection(query=query, top_k=top_k)

        collection = self._ensure_collection()
        try:
            total = collection.count()
        except NotFoundError:
            collection = self._refresh_collection()
            total = collection.count()

        if total == 0:
            return []

        limit = min(max(int(top_k), 1), total)
        query_embedding = self.embedding_provider.embed_query(query)

        try:
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
        except NotFoundError:
            collection = self._refresh_collection()
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[dict[str, Any]] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            results.append(
                {
                    "id": str(chunk_id),
                    "text": str(document),
                    "source": str(metadata.get("source", "unknown")),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "score": round(float(distance), 4),
                }
            )

        return results

    def _fallback_upsert_documents(self, chunks: list[dict[str, Any]]) -> int:
        sources = {chunk["source"] for chunk in chunks}
        if sources:
            self._fallback_records = [
                record
                for record in self._fallback_records
                if str(record.get("source")) not in sources
            ]

        embeddings = self.embedding_provider.embed_documents([chunk["text"] for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            self._fallback_records.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "embedding": embedding,
                }
            )

        self._save_fallback_records()
        return len(chunks)

    def _fallback_query_collection(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._fallback_records:
            return []

        limit = min(max(int(top_k), 1), len(self._fallback_records))
        query_embedding = self.embedding_provider.embed_query(query)

        scored_records: list[dict[str, Any]] = []
        for record in self._fallback_records:
            embedding = record.get("embedding")
            if not isinstance(embedding, list):
                continue

            similarity = self._dot_product(query_embedding, embedding)
            distance = round(1.0 - similarity, 4)
            scored_records.append(
                {
                    "id": str(record.get("id", "")),
                    "text": str(record.get("text", "")),
                    "source": str(record.get("source", "unknown")),
                    "chunk_index": int(record.get("chunk_index", 0)),
                    "score": distance,
                }
            )

        scored_records.sort(key=lambda item: item["score"])
        return scored_records[:limit]

    def _load_fallback_records(self) -> None:
        self._fallback_records = []
        if not self.fallback_store_path.exists():
            return

        try:
            payload = json.loads(self.fallback_store_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load fallback vector store data: %s", exc)
            return

        records = payload.get("records", []) if isinstance(payload, dict) else []
        if not isinstance(records, list):
            logger.warning("Fallback vector store data is invalid.")
            return

        self._fallback_records = [
            record for record in records if isinstance(record, dict) and record.get("id")
        ]

    def _save_fallback_records(self) -> None:
        payload = {
            "collection_name": self.settings.collection_name,
            "records": self._fallback_records,
        }
        self.fallback_store_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _dot_product(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            size = min(len(left), len(right))
            return sum(left[index] * right[index] for index in range(size))
        return sum(left_value * right_value for left_value, right_value in zip(left, right))
