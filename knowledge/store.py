"""Índice vectorial con ChromaDB: indexar, borrar, buscar y detectar embedders distintos.

La colección guarda en su metadata el modelo y la dimensión de los embeddings. Abrirla con
otro embedder lanza `EmbeddingMismatch`: mezclar vectores de modelos distintos da
distancias sin sentido.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chromadb

if TYPE_CHECKING:
    from knowledge.chunking import Chunk
    from knowledge.embeddings import EmbeddingProvider
    from knowledge.ingest import Document

STORE_PATH = ".chroma"
COLLECTION = "docs_operador"


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    texto: str
    metadatos: dict
    distancia: float


class EmbeddingMismatch(Exception):
    """La colección se creó con otro modelo o dimensión de embeddings."""


class VectorStore:
    def __init__(self, client, collection, embedder: EmbeddingProvider) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def index_document(self, doc: Document, chunks: list[Chunk]) -> int:
        """Borra el documento y escribe sus chunks (S36). Regresa cuántos escribió."""
        self.delete_document(doc.doc_id)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents(chunks)
        size = self.client.get_max_batch_size()
        for start in range(0, len(chunks), size):
            part = chunks[start : start + size]
            self.collection.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[start : start + size],
                documents=[c.texto for c in part],
                metadatas=[c.metadatos for c in part],
            )
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def search(self, pregunta: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        """Los `k` chunks más cercanos a la pregunta (S37), de menor a mayor distancia."""
        r = self.collection.query(
            query_embeddings=[self.embedder.embed_query(pregunta)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(chunk_id, texto, metadatos, distancia)
            for chunk_id, texto, metadatos, distancia in zip(
                r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0], strict=True
            )
        ]

    def stored_hash(self, doc_id: str) -> str | None:
        """`hash` del documento indexado, o `None` si no está en el índice."""
        r = self.collection.get(where={"doc_id": doc_id}, limit=1, include=["metadatas"])
        return r["metadatas"][0].get("hash") if r["metadatas"] else None

    def count(self) -> int:
        return self.collection.count()


def open_store(
    embedder: EmbeddingProvider, path: str = STORE_PATH, name: str = COLLECTION
) -> VectorStore:
    """Abre (o crea) la colección con distancia coseno (S36).

    Si ya existía con otro `embedding_model` o `dim`, lanza `EmbeddingMismatch`.
    """
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(
        name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": embedder.model, "dim": embedder.dim},
        embedding_function=None,  # Los embeddings los calculamos nosotros, nunca Chroma.
    )
    stored = collection.metadata or {}
    if stored.get("embedding_model") != embedder.model or stored.get("dim") != embedder.dim:
        raise EmbeddingMismatch(
            f"El índice {name!r} en {path} se creó con {stored.get('embedding_model')} "
            f"(dim {stored.get('dim')}) y lo abriste con {embedder.model} (dim {embedder.dim}). "
            "Usa el mismo embedder o reconstruye el índice con "
            "`uv run scripts/ingest_docs.py data/docs --rebuild`."
        )
    return VectorStore(client, collection, embedder)


def delete_store(path: str = STORE_PATH, name: str = COLLECTION) -> None:
    """Borra la colección completa (para `ingest_docs.py --rebuild`)."""
    client = chromadb.PersistentClient(path=path)
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)


def build_where(tipo: str | None = None, proveedor: str | None = None) -> dict | None:
    filtros = []
    if tipo is not None:
        filtros.append({"tipo_doc": tipo})
    if proveedor is not None:
        filtros.append({"proveedor": proveedor})
    if not filtros:
        return None
    if len(filtros) == 1:
        return filtros[0]
    return {"$and": filtros}  # Chroma no acepta dos llaves en el mismo dict.
