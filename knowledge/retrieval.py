"""Retrieval híbrido: vectores y BM25 fusionados con RRF, más reescritura y re-ranking.

    pregunta → [reescritura] → por consulta: top-N vectorial, top-N BM25 y [top-N del
             grafo] → RRF → [rerank] → top-k

El grafo (Clase 8) es un ranking más: con `modo="grafo"` es el único, y con `grafo=` en
otro modo se fusiona con los demás.

`Retriever.search` tiene la misma firma que `VectorStore.search`: `answer()` de la
Clase 6 lo usa sin cambios, y la herramienta `buscar_documentos` del agente también.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from knowledge.graph_search import rank_chunks
from knowledge.lexical import BM25Index
from knowledge.rerank import rerank
from knowledge.rewrite import rewrite_query

if TYPE_CHECKING:
    from core.llm_client import LLMClient
    from knowledge.graph import KnowledgeGraph
    from knowledge.store import Hit, VectorStore

Modo = Literal["vector", "bm25", "hibrida", "grafo"]
MODOS: tuple[Modo, ...] = ("vector", "bm25", "hibrida", "grafo")
RRF_K = 60  # La constante del artículo original de RRF (Cormack et al., 2009).
CANDIDATOS = 20  # Cuántos trae cada búsqueda y cuántos califica el reranker.


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: cada lista le suma 1 / (k + posición) a cada chunk.

    Solo usa posiciones, así que no importa que la distancia coseno y el puntaje BM25
    estén en escalas distintas. Un chunk que aparece arriba en varias listas sube.
    """
    puntajes: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for posicion, chunk_id in enumerate(ranking, start=1):
            puntajes[chunk_id] += 1 / (k + posicion)
    return sorted(puntajes.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        *,
        modo: Modo = "hibrida",
        client: LLMClient | None = None,
        reescribir: bool = False,
        rerank: bool = False,
        bm25: BM25Index | None = None,
        candidatos: int = CANDIDATOS,
        grafo: KnowledgeGraph | None = None,
    ) -> None:
        if (reescribir or rerank) and client is None:
            raise ValueError("reescribir y rerank necesitan un LLMClient")
        if modo == "grafo" and grafo is None:
            raise ValueError("el modo grafo necesita un KnowledgeGraph")
        self.store = store
        self.modo = modo
        self.client = client
        self.con_reescritura = reescribir
        self.con_rerank = rerank
        self.candidatos = candidatos
        self.grafo = grafo
        self.bm25 = bm25 or (BM25Index.from_store(store) if modo in ("bm25", "hibrida") else None)
        self.consultas: list[str] = []  # Las de la última búsqueda, para mostrarlas.

    def search(self, pregunta: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        self.consultas = [pregunta]
        if self.con_reescritura:
            self.consultas += rewrite_query(self.client, pregunta)

        rankings: list[list[str]] = []
        for consulta in self.consultas:
            if self.modo in ("vector", "hibrida"):
                hits = self.store.search(consulta, k=self.candidatos, where=where)
                rankings.append([h.chunk_id for h in hits])
            if self.modo in ("bm25", "hibrida"):
                lex = self.bm25.search(consulta, k=self.candidatos, where=where)
                rankings.append([h.chunk_id for h in lex])
            if self.grafo is not None:
                nodos = rank_chunks(self.grafo, consulta, k=self.candidatos, where=where)
                rankings.append([chunk_id for chunk_id, _ in nodos])

        ids = [chunk_id for chunk_id, _ in rrf(rankings)][: self.candidatos]
        hits = self.store.get_hits(ids, pregunta)
        if self.con_rerank:
            return rerank(self.client, pregunta, hits, k=k)
        return hits[:k]
