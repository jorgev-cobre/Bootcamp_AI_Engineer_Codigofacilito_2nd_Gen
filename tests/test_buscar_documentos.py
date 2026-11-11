from __future__ import annotations

import json

import pytest

from agents.tools import ToolRunner
from core.llm_client import ToolCall
from knowledge.embeddings import FakeEmbedder
from knowledge.retrieval import Retriever
from knowledge.store import open_store
from tests.test_store import document
from tools.documentos import TEXTO_MAX, buscar_documentos_tool


@pytest.fixture
def retriever(tmp_path):
    store = open_store(FakeEmbedder(dim=256), path=str(tmp_path / "chroma"))
    store.index_document(
        *document(
            "politicas/escalamiento.md",
            ["Si el proveedor no responde en 4 horas, escalar a Guardia N2."],
            seccion="Escalamiento > Prioridad alta",
        )
    )
    store.index_document(
        *document(
            "contratos/pagosred.pdf",
            ["Tiempo de respuesta de PagosRed: 4 horas hábiles."],
            tipo_doc="contrato",
            proveedor="PagosRed",
            pagina=3,
        )
    )
    store.index_document(
        *document(
            "contratos/nubehost.pdf",
            ["Tiempo de respuesta de NubeHost: 2 horas hábiles."],
            tipo_doc="contrato",
            proveedor="NubeHost",
            pagina=2,
        )
    )
    store.index_document(
        *document("faq/inventario.txt", ["Inventario anual. " + "Detalle. " * 200], tipo_doc="faq")
    )
    return Retriever(store, modo="hibrida")


def run(runner: ToolRunner, arguments: dict) -> dict:
    message = runner.execute(ToolCall("paso-1", "buscar_documentos", json.dumps(arguments)))
    return json.loads(message["content"])


def test_returns_fragments_with_their_ref(retriever):
    runner = ToolRunner([buscar_documentos_tool(retriever)], rol="lectura")

    salida = run(runner, {"consulta": "tiempo de respuesta de PagosRed"})

    refs = [r["ref"] for r in salida["resultados"]]
    assert refs[0] == "contratos/pagosred.pdf · p. 3"
    assert len(refs) == 3
    assert runner.eventos[-1]["tool_permiso"] == "lectura"


def test_long_fragments_are_trimmed(retriever):
    runner = ToolRunner([buscar_documentos_tool(retriever)])

    salida = run(runner, {"consulta": "inventario anual"})

    primero = salida["resultados"][0]
    assert primero["ref"] == "faq/inventario.txt"
    assert len(primero["texto"]) == TEXTO_MAX and primero["texto"].endswith("…")


def test_the_agent_chooses_the_filters(retriever):
    runner = ToolRunner([buscar_documentos_tool(retriever)])

    salida = run(
        runner, {"consulta": "tiempo de respuesta", "tipo": "contrato", "proveedor": "PagosRed"}
    )

    assert [r["ref"] for r in salida["resultados"]] == ["contratos/pagosred.pdf · p. 3"]


def test_unknown_tipo_is_a_validation_error(retriever):
    runner = ToolRunner([buscar_documentos_tool(retriever)])

    salida = run(runner, {"consulta": "tiempo de respuesta", "tipo": "manual"})

    assert salida["error"]["tipo"] == "validacion"
    assert "tipo" in salida["error"]["mensaje"]


def test_short_query_is_a_validation_error(retriever):
    runner = ToolRunner([buscar_documentos_tool(retriever)])

    salida = run(runner, {"consulta": "x"})

    assert salida["error"]["tipo"] == "validacion"


class SinResultados:
    def search(self, pregunta: str, k: int = 5, where: dict | None = None) -> list:
        return []


def test_no_results_explain_what_to_do():
    runner = ToolRunner([buscar_documentos_tool(SinResultados())])

    salida = run(runner, {"consulta": "vacaciones del personal"})

    assert salida["resultados"] == []
    assert "otras palabras" in salida["nota"]
