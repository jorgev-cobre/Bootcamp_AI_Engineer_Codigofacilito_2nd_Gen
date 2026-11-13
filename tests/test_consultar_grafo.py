from __future__ import annotations

import json

import pytest

from agents.tools import ToolRunner
from core.llm_client import ToolCall
from knowledge.graph_build import build_graph, load_chunks
from main import parse_args
from scripts.eval_retrieval import ROOT
from tools.grafo import consultar_grafo_tool


@pytest.fixture(scope="module")
def runner() -> ToolRunner:
    kg = build_graph(load_chunks(ROOT / "data" / "docs"))
    return ToolRunner([consultar_grafo_tool(kg)], rol="lectura")


def run(runner: ToolRunner, arguments: dict) -> dict:
    message = runner.execute(ToolCall("paso-1", "consultar_grafo", json.dumps(arguments)))
    return json.loads(message["content"])


def test_a_ticket_brings_its_provider_and_where_to_read(runner):
    salida = run(runner, {"entidad": "T-1043"})

    assert salida["entidades"] == ["T-1043"]
    assert "T-1043 depende_de NubeHost [tickets_db.json]" in salida["hechos"][:2]
    assert salida["documentos"][0].startswith("runbooks/portal-proveedores.md")
    assert any(ref.startswith("contratos/nubehost.pdf") for ref in salida["documentos"])
    assert runner.eventos[-1]["tool_permiso"] == "lectura"


def test_an_unknown_entity_gets_a_hint_instead_of_an_error(runner):
    salida = run(runner, {"entidad": "Acme Pagos"})
    assert salida["hechos"] == [] and "ID de ticket" in salida["nota"]


def test_the_cli_only_adds_the_graph_to_an_agent():
    assert parse_args(["--agent", "react", "--grafo", "¿Y el T-1043?"]).grafo
    with pytest.raises(SystemExit):
        parse_args(["--grafo", "¿Y el T-1043?"])
    assert parse_args(["--rag", "¿Y el T-1043?", "--modo", "hibrida+grafo"]).modo == "hibrida+grafo"
