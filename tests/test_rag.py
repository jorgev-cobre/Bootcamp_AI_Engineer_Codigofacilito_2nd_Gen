from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from contracts.structured_output import StructuredOutputError
from core.llm_client import LLMClient
from knowledge.rag import NO_ENCONTRADO, RagAnswer, answer, rag_messages
from knowledge.store import Hit
from tests.conftest import FakeProvider, make_response, read_events

HITS = [
    Hit(
        "politicas/escalamiento.md#2",
        "## Prioridad alta\nSi el proveedor no responde en 4 horas, escalar a Guardia N2.",
        {"fuente": "politicas/escalamiento.md", "seccion": "Escalamiento > Prioridad alta"},
        0.25,
    ),
    Hit(
        "contratos/pagosred.pdf#2",
        "Tiempo de respuesta: 4 horas hábiles.",
        {"fuente": "contratos/pagosred.pdf", "pagina": 3},
        0.35,
    ),
    Hit("faq/accesos.txt#0", "Restablece tu contraseña.", {"fuente": "faq/accesos.txt"}, 0.9),
]
QUESTION = "¿Cuál es el tiempo de respuesta de PagosRed según su contrato?"


class StubStore:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.searches: list[dict] = []

    def search(self, pregunta: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        self.searches.append({"pregunta": pregunta, "k": k, "where": where})
        return self.hits[:k]


def reply(**payload):
    return make_response("gemini", json.dumps(payload, ensure_ascii=False))


def client_for(script, logger, fake_sleep):
    provider = FakeProvider("gemini", script)
    return provider, LLMClient([provider], logger, sleep=fake_sleep)


def test_no_hits_under_threshold_does_not_call_the_llm(logger, fake_sleep):
    provider, client = client_for([], logger, fake_sleep)

    result = answer(client, StubStore([HITS[2]]), "¿Cuántos días de vacaciones tengo?")

    assert result.encontrado is False
    assert result.respuesta == NO_ENCONTRADO
    assert provider.calls == []


def test_valid_citations_build_sources(logger, fake_sleep):
    provider, client = client_for(
        [reply(encontrado=True, respuesta="Son 4 horas hábiles [2].", citas=[2])],
        logger,
        fake_sleep,
    )

    result = answer(client, StubStore(HITS), QUESTION, k=3)

    assert result.encontrado is True
    assert result.respuesta == "Son 4 horas hábiles [2]."
    assert result.fuentes == [
        {
            "n": 2,
            "etiqueta": "contratos/pagosred.pdf · p. 3",
            "chunk_id": "contratos/pagosred.pdf#2",
            "fuente": "contratos/pagosred.pdf",
            "pagina": 3,
        }
    ]
    assert result.chunk_ids == ["politicas/escalamiento.md#2", "contratos/pagosred.pdf#2"]
    assert len(provider.calls) == 1


def test_citation_out_of_range_is_repaired_once_then_fails(logger, fake_sleep):
    bad = reply(encontrado=True, respuesta="Dato [7].", citas=[7])
    provider, client = client_for([bad, bad], logger, fake_sleep)

    with pytest.raises(StructuredOutputError, match="citas fuera de rango"):
        answer(client, StubStore(HITS), QUESTION)

    assert len(provider.calls) == 2  # la llamada original y una sola reparación
    assert "citas fuera de rango: [7]" in provider.calls[1]["messages"][-1]["content"]


def test_found_without_citations_is_invalid():
    with pytest.raises(ValidationError, match="si encontrado=true, cita al menos un documento"):
        RagAnswer.model_validate({"encontrado": True, "respuesta": "Sí.", "citas": []})


def test_model_says_not_found(logger, fake_sleep):
    _, client = client_for(
        [reply(encontrado=False, respuesta="No está en los documentos.", citas=[])],
        logger,
        fake_sleep,
    )

    result = answer(client, StubStore(HITS), "¿Quién fundó Acme?")

    assert result.encontrado is False
    assert result.respuesta == "No encontré eso en los documentos de operaciones."
    assert result.fuentes == []


def test_documents_block_labels_markdown_and_pdf():
    system, user = rag_messages(QUESTION, HITS[:2])

    assert system["role"] == "system" and "<documentos>" in system["content"]
    assert (
        "[1] politicas/escalamiento.md › Escalamiento > Prioridad alta\n## Prioridad alta"
        in (user["content"])
    )
    assert "[2] contratos/pagosred.pdf · p. 3\nTiempo de respuesta" in user["content"]
    assert user["content"].rstrip().endswith(QUESTION)


def test_log_has_rag_fields(logger, log_path, fake_sleep):
    _, client = client_for(
        [reply(encontrado=True, respuesta="Escala a N2 [1].", citas=[1])], logger, fake_sleep
    )

    answer(client, StubStore(HITS), "¿Cuándo escalo?", k=5)

    [event] = read_events(log_path)
    assert event["rag_k"] == 5
    assert event["rag_hits"] == 2
    assert event["rag_chunk_ids"] == ["politicas/escalamiento.md#2", "contratos/pagosred.pdf#2"]
    assert event["rag_encontrado"] is True
    assert event["prompt_name"] == "rag_answer"
    assert "Prioridad alta" not in json.dumps(event)  # nunca el texto de los chunks
