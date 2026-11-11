"""Reto de la Clase 5: crear_ticket idempotente, local y vía MCP. Sin red."""

from __future__ import annotations

import json

import anyio
import pytest
from mcp import Client

import tools.tickets
from agents.tools import POLITICAS, ToolRunner
from contracts.tool_schemas import CREAR_TICKET, BuscarTicketArgs, CrearTicketArgs
from core.llm_client import ToolCall
from tools.tickets import buscar_ticket, crear_ticket, idempotency_key

ARGS = CrearTicketArgs(
    asunto="Falla en pagos",
    descripcion="El portal rechaza tarjetas desde las 9:00",
    prioridad="alta",
)


class DB:
    """Vista mínima de var/tickets_nuevos.json para los asserts."""

    def contar(self) -> int:
        return len(tools.tickets._nuevos()["tickets"])


@pytest.fixture
def db_vacia() -> DB:
    db = DB()
    assert db.contar() == 0  # conftest aísla var/ en tmp_path
    return db


def test_mismo_request_crea_un_solo_ticket(db_vacia):
    a = crear_ticket(ARGS, request_id="req-123")
    b = crear_ticket(ARGS, request_id="req-123")
    assert a["id"] == b["id"] == "T-2001"
    assert a["duplicado_evitado"] is False
    assert b["duplicado_evitado"] is True
    assert db_vacia.contar() == 1


def test_otro_request_crea_otro_ticket(db_vacia):
    a = crear_ticket(ARGS, request_id="req-123")
    b = crear_ticket(ARGS, request_id="req-456")
    assert a["id"] != b["id"]
    assert db_vacia.contar() == 2


def test_la_llave_ignora_mayusculas_acentos_y_espacios():
    k1 = idempotency_key("req-1", "Falla en pagos", "El portal rechaza tarjetas")
    k2 = idempotency_key("req-1", "  falla en  PAGOS ", "El portal rechazá tarjetas")
    assert k1 == k2
    assert k1 != idempotency_key("req-2", "Falla en pagos", "El portal rechaza tarjetas")


def test_buscar_ticket_encuentra_el_ticket_nuevo():
    creado = crear_ticket(ARGS, request_id="req-123")
    encontrado = buscar_ticket(BuscarTicketArgs(ticket_id=creado["id"]))
    assert encontrado["asunto"] == "Falla en pagos"
    assert encontrado["estado"] == "abierto"


def test_politica_es_escritura_idempotente():
    politica = POLITICAS["crear_ticket"]
    assert politica.permiso == "escritura"
    assert politica.idempotente is True


def test_el_runner_inyecta_el_request_id(db_vacia):
    runner = ToolRunner([CREAR_TICKET], rol="operador", request_id="req-cli")
    call = ToolCall("1", "crear_ticket", ARGS.model_dump_json())
    a = json.loads(runner.execute(call)["content"])
    b = json.loads(runner.execute(call)["content"])
    assert a["id"] == b["id"]
    assert b["duplicado_evitado"] is True
    assert db_vacia.contar() == 1


def test_el_modelo_no_puede_mandar_la_llave():
    # El schema que ve el modelo no tiene request_id ni idempotency_key.
    props = CREAR_TICKET.to_openai()["function"]["parameters"]["properties"]
    assert set(props) == {"asunto", "descripcion", "prioridad"}


def test_rol_lectura_no_puede_crear(db_vacia):
    runner = ToolRunner([CREAR_TICKET], rol="lectura")
    call = ToolCall("1", "crear_ticket", ARGS.model_dump_json())
    result = json.loads(runner.execute(call)["content"])
    assert result["error"]["tipo"] == "permiso"
    assert db_vacia.contar() == 0


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_dos_llamadas_un_ticket(db_vacia, monkeypatch):
    from operador_mcp import server

    monkeypatch.setattr(server.runner, "rol", "operador")
    payload = ARGS.model_dump()
    with anyio.fail_after(20):
        async with Client(server.mcp) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            a = await client.call_tool("crear_ticket", payload)
            b = await client.call_tool("crear_ticket", payload)
    annotations = tools["crear_ticket"].annotations
    assert annotations.read_only_hint is False
    assert annotations.destructive_hint is False
    assert annotations.idempotent_hint is True
    assert a.structured_content["id"] == b.structured_content["id"]
    assert b.structured_content["duplicado_evitado"] is True
    assert db_vacia.contar() == 1
