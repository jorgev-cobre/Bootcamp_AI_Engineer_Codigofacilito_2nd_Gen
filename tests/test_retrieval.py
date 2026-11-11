from __future__ import annotations

import pytest

from knowledge.embeddings import FakeEmbedder
from knowledge.retrieval import Retriever, rrf
from knowledge.store import open_store
from tests.test_lexical import CORPUS
from tests.test_store import document

PREGUNTA = "¿Qué dice el ticket T-1042?"
TIPOS = {"faq": "faq", "politicas": "politica", "runbooks": "runbook"}


@pytest.fixture
def store(tmp_path):
    # Dimensión grande: con el embedder falso, menos colisiones entre palabras.
    store = open_store(FakeEmbedder(dim=1024), path=str(tmp_path / "chroma"))
    textos: dict[str, list[str]] = {}
    for chunk_id, texto in CORPUS.items():
        textos.setdefault(chunk_id.split("#")[0], []).append(texto)
    for doc_id, chunks in textos.items():
        tipo = TIPOS[doc_id.split("/")[0]]
        store.index_document(*document(doc_id, chunks, titulo="", tipo_doc=tipo))
    return store


def test_rrf_rewards_chunks_that_appear_in_both_lists():
    fusion = rrf([["a", "b", "c"], ["c", "a"]])

    assert [chunk_id for chunk_id, _ in fusion] == ["a", "c", "b"]
    assert fusion[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_vector_mode_is_the_class_6_search(store):
    vector = Retriever(store, modo="vector").search(PREGUNTA, k=3)
    clase6 = store.search(PREGUNTA, k=3)

    assert [h.chunk_id for h in vector] == [h.chunk_id for h in clase6]
    assert [h.distancia for h in vector] == pytest.approx([h.distancia for h in clase6])


def test_hybrid_brings_the_id_that_vector_search_misses(store):
    vector = Retriever(store, modo="vector").search(PREGUNTA, k=3)
    hibrida = Retriever(store, modo="hibrida").search(PREGUNTA, k=3)

    assert "runbooks/caida-pagos.md#0" not in [h.chunk_id for h in vector]
    assert "runbooks/caida-pagos.md#0" in [h.chunk_id for h in hibrida]
    # Los que llegaron por BM25 también traen su distancia a la pregunta.
    assert all(0 <= h.distancia <= 2 for h in hibrida)


def test_hybrid_respects_where_in_both_searches(store):
    hits = Retriever(store, modo="hibrida").search("ticket", k=5, where={"tipo_doc": "faq"})

    assert {h.metadatos["fuente"] for h in hits} == {"faq/soporte-general.txt"}
