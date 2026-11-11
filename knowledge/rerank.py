"""Re-ranking con el LLM: de 20 candidatos parecidos a los pocos que sí responden.

La búsqueda (vectorial, BM25 o híbrida) es barata pero gruesa: ordena por parecido, no
por si el fragmento responde la pregunta. El reranker lee la pregunta junto con cada
candidato y le pone una calificación de 0 a 3. Un cross-encoder hace lo mismo con un
modelo chico; aquí usamos el LLM que ya tenemos, con salida estructurada (Clase 3).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from contracts.structured_output import generate_structured
from knowledge.rag import COMPANY, PROMPTS_DIR, etiqueta
from prompting import PromptKit

if TYPE_CHECKING:
    from core.llm_client import LLMClient
    from knowledge.store import Hit

MIN_RELEVANCIA = 2  # 3 responde, 2 aporta un dato, 1 solo trata el tema, 0 nada.
TEXTO_MAX = 1_000  # Caracteres por candidato en el prompt: 20 candidatos ≈ 5,000 tokens.


class Juicio(BaseModel):
    doc: int
    relevancia: int = Field(ge=0, le=3)


class Reranking(BaseModel):
    juicios: list[Juicio]

    @model_validator(mode="after")
    def _docs_en_rango(self, info: ValidationInfo) -> Reranking:
        n = (info.context or {}).get("n")
        fuera = [j.doc for j in self.juicios if n is not None and not 1 <= j.doc <= n]
        if fuera:
            raise ValueError(f"documentos fuera de rango: {fuera}; solo existen del 1 al {n}")
        return self


def calificar(client: LLMClient, pregunta: str, hits: list[Hit]) -> dict[int, int]:
    """Una llamada al LLM: número de candidato (desde 1) → relevancia de 0 a 3."""
    prompt = PromptKit(PROMPTS_DIR).get("rerank")
    fragmentos = "\n\n".join(
        f"[{n}] {etiqueta(hit.metadatos)}\n{hit.texto[:TEXTO_MAX]}"
        for n, hit in enumerate(hits, start=1)
    )
    salida = generate_structured(
        client,
        prompt.render(company=COMPANY, fragmentos=fragmentos, pregunta=pregunta),
        Reranking,
        max_repairs=1,
        log_extra={"prompt_name": prompt.name, "prompt_version": prompt.version},
        validation_context={"n": len(hits)},
        **prompt.model_params,
    )
    return {j.doc: j.relevancia for j in salida.juicios}


def rerank(
    client: LLMClient,
    pregunta: str,
    hits: list[Hit],
    k: int = 5,
    minimo: int = MIN_RELEVANCIA,
) -> list[Hit]:
    """Califica `hits` y regresa hasta `k`, de mayor a menor relevancia, con `minimo` o más.

    Un candidato sin calificación cuenta como 0. Con la misma calificación gana el que
    venía antes: el orden de la búsqueda desempata.
    """
    if not hits:
        return []
    relevancia = calificar(client, pregunta, hits)
    orden = sorted(range(1, len(hits) + 1), key=lambda n: (-relevancia.get(n, 0), n))
    elegidos = [n for n in orden if relevancia.get(n, 0) >= minimo][:k]
    return [replace(hits[n - 1], relevancia=relevancia[n]) for n in elegidos]
