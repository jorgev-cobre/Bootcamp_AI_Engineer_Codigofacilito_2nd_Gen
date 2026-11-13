"""Construir el grafo sin LLM: datos estructurados y menciones por reglas (Clase 8).

    tickets_db.json   → ticket —depende_de→ proveedor, ticket —atendido_por→ equipo
    proveedores.json  → un nodo por proveedor, con su servicio, SLA y estado
    data/docs         → un nodo por chunk; chunk —menciona→ entidad si el texto la nombra,
                        y chunk —contrato_de→ proveedor en las páginas de un contrato

Las reglas solo encuentran lo que ya conocen: IDs de ticket, nombres de proveedores y
equipos, y guardias N1 a N3. Lo demás (servicios, quién hospeda qué) lo extrae el LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from knowledge.chunking import CHUNK_TOKENS, chunk_file
from knowledge.graph import KnowledgeGraph, clave, nodo_id
from knowledge.ingest import PROVEEDORES_PATH, corpus_files, load_path
from knowledge.rag import etiqueta

if TYPE_CHECKING:
    from knowledge.chunking import Chunk

TICKETS_PATH = Path(__file__).resolve().parents[1] / "data" / "tickets_db.json"
GUARDIAS = ("N1", "N2", "N3")  # Los niveles de politicas/guardias.md.

_TICKET = re.compile(r"\bT-\d{4}\b")
_GUARDIA = re.compile(r"\bN([123])\b")  # "Guardia N2", "N3 decide"


def from_datos(kg: KnowledgeGraph, tickets: list[dict], proveedores: list[dict]) -> None:
    """Nodos y relaciones que ya están en los datos estructurados."""
    for guardia in GUARDIAS:
        kg.add_entidad("guardia", guardia)
    for p in proveedores:
        kg.add_entidad("proveedor", p["nombre"], servicio=p["servicio"], sla_horas=p["sla_horas"])
    for t in tickets:
        ticket = kg.add_entidad(
            "ticket", t["id"], asunto=t["asunto"], prioridad=t["prioridad"], estado=t["estado"]
        )
        equipo = kg.add_entidad("equipo", t["equipo"])
        kg.add_relacion(ticket, "atendido_por", equipo, "tickets_db.json")
        if t["proveedor"]:
            kg.add_relacion(
                ticket, "depende_de", nodo_id("proveedor", t["proveedor"]), "tickets_db.json"
            )


def menciones(kg: KnowledgeGraph, texto: str) -> list[str]:
    """Entidades del grafo que nombra el texto, sin repetir y en orden de tipo."""
    encontradas = [nodo_id("ticket", t) for t in _TICKET.findall(texto)]
    encontradas += [nodo_id("guardia", f"N{n}") for n in _GUARDIA.findall(texto)]
    encontradas = [nodo for nodo in encontradas if nodo in kg.g]
    en_texto = clave(texto)
    for tipo in ("proveedor", "equipo", "servicio"):
        encontradas += [n for n in kg.entidades(tipo) if clave(kg.nombre(n)) in en_texto]
    return list(dict.fromkeys(encontradas))


def add_chunk(kg: KnowledgeGraph, chunk: Chunk) -> str:
    """El nodo del chunk, con su texto y su ref para citar."""
    nodo = kg.add_entidad(
        "chunk",
        chunk.chunk_id,
        texto=chunk.texto,
        ref=etiqueta(chunk.metadatos),
        fuente=chunk.metadatos["fuente"],
    )
    if chunk.metadatos.get("proveedor"):
        # Las páginas de un contrato dicen «el Proveedor»; el nombre está en los metadatos.
        proveedor = nodo_id("proveedor", chunk.metadatos["proveedor"])
        kg.add_relacion(nodo, "contrato_de", proveedor, chunk.chunk_id)
    return nodo


def add_menciones(kg: KnowledgeGraph, nodo: str, texto: str) -> None:
    for entidad in menciones(kg, texto):
        kg.add_relacion(nodo, "menciona", entidad, kg.nombre(nodo))


def load_chunks(root: Path, max_tokens: int = CHUNK_TOKENS) -> list[Chunk]:
    """Los mismos chunks que indexa ingest_docs.py: los ids coinciden con los del índice."""
    chunks: list[Chunk] = []
    for archivo in corpus_files(root):
        chunks += chunk_file(load_path(archivo, root), max_tokens=max_tokens)
    return chunks


def build_graph(chunks: list[Chunk]) -> KnowledgeGraph:
    """Datos estructurados, un nodo por chunk y menciones por reglas."""
    kg = KnowledgeGraph()
    tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    proveedores = json.loads(PROVEEDORES_PATH.read_text(encoding="utf-8"))
    from_datos(kg, tickets, proveedores)
    nodos = [add_chunk(kg, chunk) for chunk in chunks]
    for nodo, chunk in zip(nodos, chunks, strict=True):
        add_menciones(kg, nodo, chunk.texto)
    return kg
