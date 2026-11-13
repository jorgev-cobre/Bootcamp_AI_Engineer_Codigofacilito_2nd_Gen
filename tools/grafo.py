"""`consultar_grafo`: el agente pregunta por una entidad y recibe sus relaciones y los
documentos donde aparece (Clase 8).

`buscar_documentos` trae texto parecido a la consulta; `consultar_grafo` sigue relaciones:
del ticket a su proveedor y del proveedor a su contrato. Con las dos, el agente averigua
primero de qué proveedor se trata y después busca la cláusula con el filtro `proveedor`.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from contracts.tool_schemas import ConsultarGrafoArgs, ToolSpec
from knowledge.graph_build import menciones
from knowledge.graph_search import hechos, rank_chunks

if TYPE_CHECKING:
    from knowledge.graph import KnowledgeGraph

LIMITE_HECHOS = 6
LIMITE_DOCUMENTOS = 5
SIN_ENTIDAD = "No está en el grafo. Usa un ID de ticket, un proveedor, un equipo o una guardia."


def consultar_grafo(args: ConsultarGrafoArgs, *, kg: KnowledgeGraph) -> dict:
    semillas = menciones(kg, args.entidad)
    if not semillas:
        return {
            "hechos": [],
            "documentos": [],
            "nota": SIN_ENTIDAD,
        }
    return {
        "entidades": [kg.nombre(nodo) for nodo in semillas],
        "hechos": hechos(kg, args.entidad, limite=LIMITE_HECHOS),
        "documentos": [
            kg.g.nodes[f"chunk:{chunk_id}"]["ref"]
            for chunk_id, _ in rank_chunks(kg, args.entidad, k=LIMITE_DOCUMENTOS)
        ],
    }


def consultar_grafo_tool(kg: KnowledgeGraph) -> ToolSpec:
    """La herramienta lista para el agente, ligada a un grafo ya cargado."""
    return ToolSpec(
        name="consultar_grafo",
        description=(
            "Consulta el grafo de conocimiento de operaciones. Dado un ticket (T-1043), un "
            "proveedor, un equipo, una guardia (N2) o un servicio, regresa sus relaciones "
            "con su fuente (por ejemplo, de qué proveedor depende un ticket) y los documentos "
            "donde aparece. Úsala para saber cómo se conectan las cosas; después usa "
            "buscar_documentos, con el filtro de proveedor si aplica, para leer la cláusula. "
            "Cita las fuentes entre corchetes."
        ),
        args_model=ConsultarGrafoArgs,
        handler=partial(consultar_grafo, kg=kg),
    )
