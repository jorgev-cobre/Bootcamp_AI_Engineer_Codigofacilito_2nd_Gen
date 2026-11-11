from __future__ import annotations

import pytest

from knowledge.rerank import rerank
from knowledge.retrieval import Retriever
from knowledge.store import Hit
from tests.test_rewrite import StubStore, client_for, reply

HITS = [
    Hit("faq/soporte-general.txt#0", "Responde al correo del ticket.", {"fuente": "faq"}, 0.30),
    Hit("politicas/escalamiento.md#2", "Escalar a Guardia N2.", {"fuente": "esc"}, 0.35),
    Hit("contratos/pagosred.pdf#2", "4 horas hábiles.", {"fuente": "pagosred", "pagina": 3}, 0.40),
    Hit("faq/accesos.txt#0", "Restablece tu contraseña.", {"fuente": "accesos"}, 0.50),
]


def juicios(*relevancias: int) -> dict:
    return {"juicios": [{"doc": n, "relevancia": r} for n, r in enumerate(relevancias, start=1)]}


def test_rerank_orders_by_relevance_and_drops_the_irrelevant(logger, fake_sleep):
    provider, client = client_for([reply(**juicios(1, 3, 2, 0))], logger, fake_sleep)

    hits = rerank(client, "¿Cuándo escalo con PagosRed?", HITS, k=5)

    assert [h.chunk_id for h in hits] == ["politicas/escalamiento.md#2", "contratos/pagosred.pdf#2"]
    assert [h.relevancia for h in hits] == [3, 2]
    prompt = provider.calls[0]["messages"][-1]["content"]
    assert "[3] pagosred · p. 3" in prompt


def test_ties_keep_the_search_order_and_missing_judgments_count_as_zero(logger, fake_sleep):
    payload = {"juicios": [{"doc": 3, "relevancia": 2}, {"doc": 1, "relevancia": 2}]}
    _, client = client_for([reply(**payload)], logger, fake_sleep)

    hits = rerank(client, "pregunta", HITS, k=5)

    assert [h.chunk_id for h in hits] == ["faq/soporte-general.txt#0", "contratos/pagosred.pdf#2"]


def test_out_of_range_document_is_repaired_once(logger, fake_sleep):
    malo = {"juicios": [{"doc": 7, "relevancia": 3}]}
    provider, client = client_for([reply(**malo), reply(**juicios(3, 0, 0, 0))], logger, fake_sleep)

    hits = rerank(client, "pregunta", HITS, k=2)

    assert [h.chunk_id for h in hits] == ["faq/soporte-general.txt#0"]
    assert len(provider.calls) == 2
    assert "fuera de rango" in provider.calls[1]["messages"][-1]["content"]


def test_no_candidates_means_no_llm_call(logger, fake_sleep):
    provider, client = client_for([], logger, fake_sleep)

    assert rerank(client, "pregunta", []) == []
    assert provider.calls == []


def test_the_retriever_reranks_its_candidates(logger, fake_sleep):
    _, client = client_for([reply(**juicios(1, 3, 2, 0))], logger, fake_sleep)
    retriever = Retriever(StubStore(HITS), modo="vector", client=client, rerank=True)

    hits = retriever.search("¿Cuándo escalo con PagosRed?", k=5)

    assert [h.chunk_id for h in hits] == ["politicas/escalamiento.md#2", "contratos/pagosred.pdf#2"]
    assert [h.relevancia for h in hits] == [3, 2]


def test_reranking_needs_a_client():
    with pytest.raises(ValueError, match="LLMClient"):
        Retriever(StubStore(HITS), modo="vector", rerank=True)
