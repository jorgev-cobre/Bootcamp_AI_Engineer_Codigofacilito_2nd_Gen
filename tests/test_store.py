from __future__ import annotations

import pytest

from knowledge.chunking import Chunk
from knowledge.embeddings import FakeEmbedder
from knowledge.ingest import Document
from knowledge.store import EmbeddingMismatch, Hit, build_where, open_store


def document(doc_id: str, textos: list[str], **metadatos) -> tuple[Document, list[Chunk]]:
    base = {"fuente": doc_id, "tipo_doc": "politica", "titulo": doc_id, "hash": str(len(textos))}
    base.update(metadatos)
    doc = Document(doc_id, "\n\n".join(textos), base)
    chunks = [
        Chunk(f"{doc_id}#{n}", texto, {**base, "doc_id": doc_id, "n": n})
        for n, texto in enumerate(textos)
    ]
    return doc, chunks


@pytest.fixture
def store(tmp_path):
    return open_store(FakeEmbedder(dim=64), path=str(tmp_path / "chroma"))


def test_collection_uses_cosine(store):
    assert store.collection.configuration["hnsw"]["space"] == "cosine"
    assert store.collection.metadata == {"embedding_model": "fake", "dim": 64}


def test_reindexing_a_shorter_document_leaves_no_orphans(store):
    store.index_document(*document("politicas/sla.md", ["uno", "dos", "tres", "cuatro"]))
    store.index_document(*document("politicas/sla.md", ["uno corregido"]))

    stored = store.collection.get(where={"doc_id": "politicas/sla.md"})
    assert stored["ids"] == ["politicas/sla.md#0"]
    assert stored["documents"] == ["uno corregido"]


def test_delete_document_and_stored_hash(store):
    store.index_document(*document("politicas/sla.md", ["uno", "dos"]))
    store.index_document(*document("faq/accesos.txt", ["contraseña"], tipo_doc="faq"))
    assert store.stored_hash("politicas/sla.md") == "2"

    store.delete_document("politicas/sla.md")

    assert store.stored_hash("politicas/sla.md") is None
    assert store.count() == 1


@pytest.mark.parametrize(
    "other", [FakeEmbedder(dim=32), type("Otro", (FakeEmbedder,), {"model": "otro"})(dim=64)]
)
def test_opening_with_another_embedder_raises(tmp_path, other):
    open_store(FakeEmbedder(dim=64), path=str(tmp_path / "chroma"))
    with pytest.raises(EmbeddingMismatch, match="--rebuild"):
        open_store(other, path=str(tmp_path / "chroma"))


def test_search_returns_hits_sorted_by_distance(store):
    store.index_document(
        *document(
            "runbooks/reinicio-colas.md",
            ["Reinicia la cola de pedidos del almacén", "Renueva el certificado del sitio"],
        )
    )
    store.index_document(*document("faq/accesos.txt", ["Restablece tu contraseña"], tipo_doc="faq"))

    hits = store.search("¿cómo reinicio la cola de pedidos?", k=3)

    assert all(isinstance(h, Hit) for h in hits)
    assert hits[0].chunk_id == "runbooks/reinicio-colas.md#0"
    assert [h.distancia for h in hits] == sorted(h.distancia for h in hits)


def test_search_with_where_filters(store):
    store.index_document(*document("politicas/sla.md", ["cola de pedidos"]))
    store.index_document(*document("faq/accesos.txt", ["cola de pedidos"], tipo_doc="faq"))

    hits = store.search("cola de pedidos", k=5, where={"tipo_doc": "faq"})

    assert [h.metadatos["fuente"] for h in hits] == ["faq/accesos.txt"]


def test_build_where_without_filters_is_none():
    assert build_where() is None


def test_build_where_is_the_guided_practice():
    with pytest.raises(NotImplementedError):
        build_where(tipo="politica")
