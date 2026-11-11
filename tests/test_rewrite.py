from __future__ import annotations

import json

import pytest

from core.llm_client import LLMClient
from knowledge.retrieval import Retriever
from knowledge.rewrite import rewrite_query
from knowledge.store import Hit
from tests.conftest import FakeProvider, make_response

HITS = [
    Hit("politicas/sla.md#5", "PagosRed: 4 horas hábiles.", {"fuente": "politicas/sla.md"}, 0.30),
    Hit("contratos/pagosred.pdf#2", "Tiempo de respuesta.", {"fuente": "pagosred"}, 0.35),
]


class StubStore:
    """`search` y `get_hits` sin Chroma: siempre los mismos hits, y anota cada consulta."""

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.consultas: list[str] = []

    def search(self, pregunta: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        self.consultas.append(pregunta)
        return self.hits[:k]

    def get_hits(self, chunk_ids: list[str], pregunta: str) -> list[Hit]:
        por_id = {h.chunk_id: h for h in self.hits}
        return [por_id[chunk_id] for chunk_id in chunk_ids]


def client_for(script, logger, fake_sleep):
    provider = FakeProvider("gemini", script)
    return provider, LLMClient([provider], logger, sleep=fake_sleep)


def reply(**payload):
    return make_response("gemini", json.dumps(payload, ensure_ascii=False))


def test_rewrite_drops_the_original_question_and_duplicates(logger, fake_sleep):
    consultas = [
        "tiempo de respuesta PagosRed",
        "¿Cuánto tardan los de pagos en contestar?",
        "PagosRed tiempo de respuesta",
        "penalización por incumplir el tiempo de respuesta",
        "contrato PagosRed cláusula de niveles de servicio",
    ]
    _, client = client_for([reply(consultas=consultas)], logger, fake_sleep)

    salida = rewrite_query(client, "¿Cuánto tardan los de pagos en contestar?", n=3)

    assert salida == [
        "tiempo de respuesta PagosRed",
        "penalización por incumplir el tiempo de respuesta",
        "contrato PagosRed cláusula de niveles de servicio",
    ]


def test_every_rewritten_query_is_searched(logger, fake_sleep):
    pregunta = "¿Cuánto tardan los de pagos en contestar?"
    provider, client = client_for(
        [reply(consultas=["tiempo de respuesta PagosRed"])], logger, fake_sleep
    )
    store = StubStore(HITS)
    retriever = Retriever(store, modo="vector", client=client, reescribir=True)

    hits = retriever.search(pregunta, k=2)

    assert store.consultas == [pregunta, "tiempo de respuesta PagosRed"]
    assert retriever.consultas == store.consultas
    assert [h.chunk_id for h in hits] == ["politicas/sla.md#5", "contratos/pagosred.pdf#2"]
    assert len(provider.calls) == 1


def test_rewriting_needs_a_client():
    with pytest.raises(ValueError, match="LLMClient"):
        Retriever(StubStore(HITS), modo="vector", reescribir=True)
