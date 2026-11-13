"""Grafo de conocimiento del Agente Operador, con NetworkX (Clase 8).

Nodos: entidades (ticket, proveedor, equipo, guardia, servicio) y los chunks de data/docs.
Aristas: relaciones dirigidas con nombre. Cada relación guarda sus `fuentes` (de dónde
salió: tickets_db.json, proveedores.json o un chunk) para poder citar cada hecho.

Es un grafo de propiedades: nodos y aristas llevan atributos, el mismo modelo que usa
Neo4j. NetworkX lo tiene en memoria; aquí lo guardamos como JSON en var/grafo.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from knowledge.lexical import tokenize

GRAPH_PATH = Path("var") / "grafo.json"
TIPOS = ("ticket", "proveedor", "equipo", "guardia", "servicio", "chunk")


def nodo_id(tipo: str, nombre: str) -> str:
    """`ticket:T-1043`, `proveedor:NubeHost`, `chunk:runbooks/portal-proveedores.md#0`."""
    return f"{tipo}:{nombre}"


def clave(texto: str) -> str:
    """Forma comparable de un nombre: «Transportes Sierra» → « transportes sierra »."""
    return f" {' '.join(tokenize(texto))} "


class KnowledgeGraph:
    def __init__(self, g: nx.MultiDiGraph | None = None) -> None:
        self.g = g if g is not None else nx.MultiDiGraph()

    def add_entidad(self, tipo: str, nombre: str, **atributos) -> str:
        """Crea el nodo o, si ya existe, le agrega los atributos. Regresa su id."""
        if tipo not in TIPOS:
            raise ValueError(f"tipo desconocido: {tipo}")
        nodo = nodo_id(tipo, nombre)
        if nodo in self.g:
            self.g.nodes[nodo].update(atributos)
        else:
            self.g.add_node(nodo, tipo=tipo, nombre=nombre, **atributos)
        return nodo

    def add_relacion(self, origen: str, relacion: str, destino: str, fuente: str) -> None:
        """Una arista por (origen, relación, destino). Si ya existe, solo suma la fuente."""
        for nodo in (origen, destino):
            if nodo not in self.g:
                raise KeyError(f"nodo desconocido: {nodo}")
        if self.g.has_edge(origen, destino, key=relacion):
            fuentes = self.g.edges[origen, destino, relacion]["fuentes"]
            if fuente not in fuentes:
                fuentes.append(fuente)
        else:
            self.g.add_edge(origen, destino, key=relacion, relacion=relacion, fuentes=[fuente])

    def entidades(self, tipo: str) -> list[str]:
        return sorted(n for n, datos in self.g.nodes(data=True) if datos["tipo"] == tipo)

    def nombre(self, nodo: str) -> str:
        return self.g.nodes[nodo]["nombre"]

    def buscar(self, tipo: str, nombre: str) -> str | None:
        """El nodo de ese tipo con el mismo nombre, sin importar mayúsculas ni acentos."""
        buscado = clave(nombre)
        return next((n for n in self.entidades(tipo) if clave(self.nombre(n)) == buscado), None)

    def resumen(self) -> dict[str, int]:
        """Nodos por tipo y número de relaciones."""
        conteo = Counter(datos["tipo"] for _, datos in self.g.nodes(data=True))
        return {**{tipo: conteo[tipo] for tipo in TIPOS}, "relaciones": self.g.number_of_edges()}

    def save(self, path: Path | str = GRAPH_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self.g, edges="edges")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str = GRAPH_PATH) -> KnowledgeGraph:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(json_graph.node_link_graph(data, edges="edges"))
