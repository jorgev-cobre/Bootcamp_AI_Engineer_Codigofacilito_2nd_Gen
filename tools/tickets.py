"""Herramientas de tickets sobre data/tickets_db.json.

`data/tickets_db.json` es de solo lectura. Los cambios de estado van a un overlay,
`var/tickets_estado.json` (`{ticket_id: estado}`), que `buscar_ticket` aplica al leer.
Los tickets creados por el agente van a `var/tickets_nuevos.json`, con IDs `T-2xxx`.
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contracts.tool_schemas import (
        ActualizarEstadoArgs,
        BuscarTicketArgs,
        CrearTicketArgs,
        ListarTicketsPorProveedorArgs,
    )

# Rutas fijas: nunca se construyen a partir de los argumentos del modelo.
TICKETS_DB = Path(__file__).resolve().parents[1] / "data" / "tickets_db.json"
ESTADOS_PATH = Path(__file__).resolve().parents[1] / "var" / "tickets_estado.json"
NUEVOS_PATH = Path(__file__).resolve().parents[1] / "var" / "tickets_nuevos.json"
PRIMER_ID_NUEVO = 2001
HISTORY_LIMIT = 3
LISTA_LIMITE = 10


@lru_cache(maxsize=1)
def _tickets() -> dict[str, dict]:
    tickets = json.loads(TICKETS_DB.read_text(encoding="utf-8"))
    return {ticket["id"]: ticket for ticket in tickets}


def buscar_ticket(args: BuscarTicketArgs) -> dict:
    """Estado, equipo, proveedor y últimas entradas del historial de un ticket."""
    ticket = _tickets().get(args.ticket_id) or _nuevos()["tickets"].get(args.ticket_id)
    if ticket is None:
        return {"error": f"ticket {args.ticket_id} no existe"}
    return {
        "id": ticket["id"],
        "asunto": ticket["asunto"],
        "estado": _estados().get(ticket["id"], ticket["estado"]),
        "equipo": ticket["equipo"],
        "prioridad": ticket["prioridad"],
        "proveedor": ticket["proveedor"],
        "historial": ticket["historial"][-HISTORY_LIMIT:],
    }


def _estados() -> dict[str, str]:
    """Overlay de estados. Se lee en cada llamada: otro proceso (el server MCP) lo escribe."""
    if not ESTADOS_PATH.exists():
        return {}
    return json.loads(ESTADOS_PATH.read_text(encoding="utf-8"))


def _guardar_estados(estados: dict[str, str]) -> None:
    """Escritura atómica: un archivo temporal y `os.replace`, nunca un JSON a medias."""
    ESTADOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ESTADOS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(estados, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, ESTADOS_PATH)


def actualizar_estado(args: ActualizarEstadoArgs) -> dict:
    """Fija el estado de un ticket. Idempotente: repetirla con el mismo estado no cambia nada."""
    from contracts.tool_schemas import tool_error  # Import local: tool_schemas importa este módulo.

    ticket = _tickets().get(args.ticket_id)
    if ticket is None:
        return tool_error(
            "negocio",
            f"ticket {args.ticket_id} no existe",
            sugerencia="Confirma el ID con el usuario antes de reintentar.",
        )
    estados = _estados()
    anterior = estados.get(args.ticket_id, ticket["estado"])
    cambio = anterior != args.estado
    if cambio:
        estados[args.ticket_id] = args.estado
        _guardar_estados(estados)
    return {"id": args.ticket_id, "estado": args.estado, "cambio": cambio}


def _normalizar(texto: str) -> str:
    """Minúsculas, sin acentos ni espacios alrededor: "  PAGOSRED " == "pagosred"."""
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_acentos.lower().strip()


def listar_tickets_por_proveedor(args: ListarTicketsPorProveedorArgs) -> dict:
    """Tickets de un proveedor (máximo 10) y el total encontrado. Solo lectura."""
    buscado = _normalizar(args.nombre)
    estados = _estados()
    encontrados = [
        ticket
        for ticket in _tickets().values()
        if ticket["proveedor"] and _normalizar(ticket["proveedor"]) == buscado
    ]
    return {
        "proveedor": args.nombre,
        "total": len(encontrados),
        "tickets": [
            {
                "id": ticket["id"],
                "asunto": ticket["asunto"],
                "estado": estados.get(ticket["id"], ticket["estado"]),
            }
            for ticket in encontrados[:LISTA_LIMITE]
        ],
    }


# --- Reto de la Clase 5: crear_ticket idempotente -------------------------------------


def _nuevos() -> dict:
    """`{"tickets": {id: ticket}, "llaves": {llave: id}}`. Se lee en cada llamada."""
    if not NUEVOS_PATH.exists():
        return {"tickets": {}, "llaves": {}}
    return json.loads(NUEVOS_PATH.read_text(encoding="utf-8"))


def _guardar_nuevos(data: dict) -> None:
    """Misma escritura atómica que el overlay de estados."""
    NUEVOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = NUEVOS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, NUEVOS_PATH)


def idempotency_key(request_id: str, asunto: str, descripcion: str) -> str:
    """Llave estable: el mismo request con el mismo contenido da la misma llave.

    El contenido se normaliza (minúsculas, sin acentos, espacios colapsados) para que
    "Falla en pagos" y "  falla en  PAGOS" cuenten como el mismo ticket.
    """

    def norm(texto: str) -> str:
        return " ".join(_normalizar(texto).split())

    base = f"{request_id}|{norm(asunto)}|{norm(descripcion)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def crear_ticket(args: CrearTicketArgs, *, request_id: str) -> dict:
    """Crea un ticket `T-2xxx`. Idempotente: con la misma llave regresa el ticket que ya existe."""
    llave = idempotency_key(request_id, args.asunto, args.descripcion)
    data = _nuevos()
    previo_id = data["llaves"].get(llave)
    if previo_id:
        previo = data["tickets"][previo_id]
        return {**_resumen(previo), "duplicado_evitado": True}

    ticket_id = f"T-{PRIMER_ID_NUEVO + len(data['tickets'])}"
    ticket = {
        "id": ticket_id,
        "asunto": args.asunto,
        "descripcion": args.descripcion,
        "estado": "abierto",
        "equipo": "Mesa de ayuda",
        "prioridad": args.prioridad,
        "proveedor": None,
        "historial": [
            {"fecha": date.today().isoformat(), "nota": "Creado por el Agente Operador."}
        ],
    }
    data["tickets"][ticket_id] = ticket
    data["llaves"][llave] = ticket_id
    _guardar_nuevos(data)
    return {**_resumen(ticket), "duplicado_evitado": False}


def _resumen(ticket: dict) -> dict:
    campos = ("id", "asunto", "prioridad", "estado")
    return {campo: ticket[campo] for campo in campos}
