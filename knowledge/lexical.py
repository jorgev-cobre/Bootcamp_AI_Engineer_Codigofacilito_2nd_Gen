"""Búsqueda léxica con BM25: encuentra las palabras exactas que los embeddings diluyen.

Un embedding compara significado. Un ID como "T-1042", un correo o un número de cláusula
casi no tienen significado, y la búsqueda vectorial los confunde con cualquier otro ID.
BM25 premia los términos raros del corpus (IDF) y normaliza por la longitud del chunk.

El índice se arma en memoria con los chunks que ya están en Chroma: no hay una segunda
copia en disco que mantener al día.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge.store import VectorStore

K1 = 1.5  # Saturación: la décima aparición de un término suma mucho menos que la primera.
B = 0.75  # Cuánto se castiga a un chunk largo (0 = nada, 1 = proporcional a su largo).

# "T-1042", "99.5" y "mesa@nubehost.example" quedan como un solo token.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-@.][a-z0-9]+)*")
# Ya sin acentos, porque `tokenize` los quita antes de comparar.
STOPWORDS = frozenset(
    "a al como con cual cuales cuando cuanto cuantos de del desde donde el en entre es esa ese "
    "eso esta este esto fue hay hasta la las le les lo los mas me mi mis muy no nos o para pero "
    "por que quien quienes se ser si sin sobre son su sus te un una uno unos unas y ya".split()
)


def tokenize(texto: str) -> list[str]:
    """Minúsculas, sin acentos, sin stopwords. "¿Qué pasó con el T-1042?" → [paso, t-1042]."""
    plano = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return [t for t in _TOKEN.findall(plano) if t not in STOPWORDS]


def matches(metadatos: dict, where: dict | None) -> bool:
    """Aplica los dos formatos que arma `build_where`: `{campo: valor}` y `{"$and": [...]}`."""
    if not where:
        return True
    if "$and" in where:
        return all(matches(metadatos, w) for w in where["$and"])
    return all(metadatos.get(campo) == valor for campo, valor in where.items())


@dataclass(frozen=True)
class LexHit:
    chunk_id: str
    puntaje: float


class BM25Index:
    def __init__(self, textos: dict[str, str], metadatos: dict[str, dict] | None = None) -> None:
        self.metadatos = metadatos or {}
        self.tf = {chunk_id: Counter(tokenize(texto)) for chunk_id, texto in textos.items()}
        self.largo = {chunk_id: sum(tf.values()) for chunk_id, tf in self.tf.items()}
        self.promedio = sum(self.largo.values()) / max(len(self.largo), 1)
        # df: en cuántos chunks aparece cada término.
        self.df = Counter(termino for tf in self.tf.values() for termino in tf)

    @classmethod
    def from_store(cls, store: VectorStore) -> BM25Index:
        """Lee todos los chunks del índice. Título y sección cuentan como texto del chunk."""
        r = store.collection.get(include=["documents", "metadatas"])
        textos = {
            chunk_id: f"{meta.get('titulo', '')} {meta.get('seccion', '')} {texto}"
            for chunk_id, texto, meta in zip(r["ids"], r["documents"], r["metadatas"], strict=True)
        }
        return cls(textos, dict(zip(r["ids"], r["metadatas"], strict=True)))

    def idf(self, termino: str) -> float:
        n, df = len(self.tf), self.df[termino]
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, terminos: set[str], chunk_id: str) -> float:
        tf = self.tf[chunk_id]
        norma = K1 * (1 - B + B * self.largo[chunk_id] / self.promedio)
        return sum(self.idf(t) * tf[t] * (K1 + 1) / (tf[t] + norma) for t in terminos if t in tf)

    def search(self, pregunta: str, k: int = 5, where: dict | None = None) -> list[LexHit]:
        """Los `k` chunks con mayor puntaje BM25. Un chunk sin términos en común no entra."""
        terminos = set(tokenize(pregunta))
        hits = [
            LexHit(chunk_id, self.score(terminos, chunk_id))
            for chunk_id in self.tf
            if matches(self.metadatos.get(chunk_id, {}), where)
        ]
        hits = [h for h in hits if h.puntaje > 0]
        return sorted(hits, key=lambda h: (-h.puntaje, h.chunk_id))[:k]
