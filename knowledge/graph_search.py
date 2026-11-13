"""Consultar el grafo: de la pregunta a las semillas y de las semillas a los chunks (Clase 8).

    pregunta → semillas: las entidades que nombra (las mismas reglas que con los chunks)
             → PageRank personalizado desde las semillas → chunks con más probabilidad

Es la idea de HippoRAG: una caminata aleatoria que en cada paso sigue una arista con
probabilidad α o regresa a una semilla con probabilidad 1 − α. Los chunks cerca de varias
semillas, o a dos saltos por el proveedor de un ticket, juntan más probabilidad.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx

from knowledge.graph_build import menciones
from knowledge.lexical import matches

if TYPE_CHECKING:
    from knowledge.graph import KnowledgeGraph

ALPHA = 0.85  # Probabilidad de seguir caminando; con 1 − α se regresa a una semilla.
ITERACIONES = 30  # 0.85 ** 30 ≈ 0.008: suficiente para ordenar.
POR_DOCUMENTO = 2  # Chunks por documento: el grafo sirve para juntar documentos distintos.


def ppr(
    g: nx.Graph, semillas: list[str], alpha: float = ALPHA, iteraciones: int = ITERACIONES
) -> dict[str, float]:
    """PageRank personalizado por iteración de potencias. Las probabilidades suman 1."""
    reinicio = {s: 1 / len(semillas) for s in semillas}
    p = dict(reinicio)
    for _ in range(iteraciones):
        nuevo: dict[str, float] = defaultdict(float)
        for nodo, masa in p.items():
            vecinos = list(g.neighbors(nodo)) or list(reinicio)  # Sin vecinos: a las semillas.
            for vecino in vecinos:
                nuevo[vecino] += alpha * masa / len(vecinos)
        for semilla, peso in reinicio.items():
            nuevo[semilla] += (1 - alpha) * peso
        p = nuevo
    return dict(p)


def caminata(kg: KnowledgeGraph, pregunta: str) -> dict[str, float]:
    """Probabilidad de cada nodo desde las entidades de la pregunta ({} si no nombra ninguna).

    Las relaciones tienen dirección, pero la caminata las recorre en los dos sentidos: del
    ticket a su proveedor y del proveedor a las páginas de su contrato.
    """
    semillas = menciones(kg, pregunta)
    if not semillas:
        return {}
    return ppr(kg.g.to_undirected(as_view=True), semillas)


def rank_chunks(
    kg: KnowledgeGraph,
    pregunta: str,
    k: int = 20,
    where: dict | None = None,
    por_documento: int = POR_DOCUMENTO,
) -> list[tuple[str, float]]:
    """Hasta `k` chunks, de más a menos probables, como (chunk_id, probabilidad).

    `where` filtra igual que en Chroma y en BM25. Sin el tope por documento, un runbook
    largo que nombra al ticket en cada sección se queda con el top-5 y el contrato del
    proveedor no entra.
    """
    puntajes = caminata(kg, pregunta)
    nodos = [n for n in puntajes if kg.g.nodes[n]["tipo"] == "chunk"]
    nodos = [n for n in nodos if matches(kg.g.nodes[n]["metadatos"], where)]
    elegidos, por_fuente = [], defaultdict(int)
    for nodo in sorted(nodos, key=lambda n: (-puntajes[n], n)):
        chunk_id = kg.nombre(nodo)
        fuente = chunk_id.split("#")[0]
        if por_fuente[fuente] < por_documento:
            por_fuente[fuente] += 1
            elegidos.append((chunk_id, puntajes[nodo]))
    return elegidos[:k]


def _fuente(kg: KnowledgeGraph, fuente: str) -> str:
    nodo = f"chunk:{fuente}"
    return kg.g.nodes[nodo]["ref"] if nodo in kg.g else fuente


def _etiqueta(kg: KnowledgeGraph, nodo: str) -> str:
    """El nombre de una entidad o la ref de un chunk."""
    return kg.g.nodes[nodo].get("ref") or kg.nombre(nodo)


def hecho_entre(kg: KnowledgeGraph, u: str, v: str) -> str:
    """La relación entre dos nodos vecinos, en la dirección en que existe, con sus fuentes."""
    if not kg.g.has_edge(u, v):
        u, v = v, u
    datos = next(iter(kg.g.get_edge_data(u, v).values()))
    hecho = f"{_etiqueta(kg, u)} {datos['relacion']} {_etiqueta(kg, v)}"
    if datos["relacion"] == "menciona":  # El chunk que menciona ya es la fuente.
        return hecho
    return f"{hecho} [{', '.join(_fuente(kg, f) for f in datos['fuentes'])}]"


def hechos(kg: KnowledgeGraph, pregunta: str, limite: int = 10) -> list[str]:
    """Relaciones entre entidades, de la más cercana a la más lejana, con sus fuentes.

    «T-1043 depende_de NubeHost [tickets_db.json]». Una relación vale lo que su extremo
    menos probable: así los 21 tickets del equipo TI no inundan la lista.
    """
    puntajes = caminata(kg, pregunta)
    candidatas = [
        (min(puntajes.get(u, 0), puntajes.get(v, 0)), u, v, datos)
        for u, v, datos in kg.g.edges(data=True)
        if "chunk" not in (kg.g.nodes[u]["tipo"], kg.g.nodes[v]["tipo"])
        and u in puntajes
        and v in puntajes
    ]
    candidatas.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [hecho_entre(kg, u, v) for _, u, v, _ in candidatas[:limite]]


def comunidades(kg: KnowledgeGraph, semilla: int = 42) -> list[set[str]]:
    """Grupos de nodos más conectados entre sí que con el resto (Louvain), del más grande
    al más chico, sin los chunks que quedaron solos.

    GraphRAG de Microsoft usa Leiden jerárquico y le pide al LLM un reporte de cada grupo;
    la búsqueda global responde con esos reportes. Aquí solo se calculan los grupos.
    """
    grupos = nx.community.louvain_communities(nx.Graph(kg.g), seed=semilla)
    return sorted((g for g in grupos if len(g) > 1), key=lambda g: (-len(g), sorted(g)))


def ruta(kg: KnowledgeGraph, origen: str, destino: str) -> list[str]:
    """El camino más corto entre dos entidades, un hecho por salto ([] si no hay camino).

    Explica por qué dos cosas están conectadas, con una fuente para citar en cada salto.
    """
    inicio, fin = menciones(kg, origen), menciones(kg, destino)
    if not inicio or not fin:
        return []
    try:
        nodos = nx.shortest_path(kg.g.to_undirected(as_view=True), inicio[0], fin[0])
    except nx.NetworkXNoPath:
        return []
    return [hecho_entre(kg, u, v) for u, v in zip(nodos, nodos[1:], strict=False)]
