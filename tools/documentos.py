"""`buscar_documentos`: el agente decide cuándo buscar en los documentos, con qué palabras
y cuántas veces (Clase 7).

En `--rag` la búsqueda es un paso fijo antes de responder. Como herramienta, el agente
puede consultar el ticket, después la política y el contrato, y volver a buscar con
otras palabras si no encontró el dato.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from contracts.tool_schemas import BuscarDocumentosArgs, ToolSpec
from knowledge.rag import etiqueta

if TYPE_CHECKING:
    from knowledge.retrieval import Retriever

K = 3  # Fragmentos por búsqueda; si no alcanzan, el agente busca otra vez.
TEXTO_MAX = 1_000  # Caracteres por fragmento: la observación completa cabe en el agente.


def _recortar(texto: str) -> str:
    texto = " ".join(texto.split())
    return texto if len(texto) <= TEXTO_MAX else texto[: TEXTO_MAX - 1] + "…"


def buscar_documentos(args: BuscarDocumentosArgs, *, retriever: Retriever) -> dict:
    # TODO(clase-7): práctica guiada. Pasa los filtros que eligió el agente:
    # where=build_where(args.tipo, args.proveedor).
    hits = retriever.search(args.consulta, k=K)
    if not hits:
        return {
            "resultados": [],
            "nota": "Sin resultados. Prueba con otras palabras o responde que no está.",
        }
    return {
        "resultados": [{"ref": etiqueta(h.metadatos), "texto": _recortar(h.texto)} for h in hits]
    }


def buscar_documentos_tool(retriever: Retriever) -> ToolSpec:
    """La herramienta lista para el agente, ligada a un `Retriever` ya abierto."""
    return ToolSpec(
        name="buscar_documentos",
        description=(
            "Busca en los documentos de operaciones (runbooks, políticas, contratos con "
            "proveedores y preguntas frecuentes) y regresa los 3 fragmentos más relevantes, "
            "cada uno con su ref. Úsala para reglas, procedimientos, SLA, penalizaciones y "
            "cláusulas; para el estado de un ticket usa buscar_ticket. Si no encontraste el "
            "dato, busca otra vez con otras palabras. Cita la ref de cada dato que uses, "
            "entre corchetes."
        ),
        args_model=BuscarDocumentosArgs,
        handler=partial(buscar_documentos, retriever=retriever),
    )
