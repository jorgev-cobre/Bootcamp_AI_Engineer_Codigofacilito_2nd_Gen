"""RAG: buscar en el índice, responder solo con los documentos y citar cada dato (S42).

Sin hits bajo el umbral no se llama al LLM. La respuesta es un `RagAnswer` validado con
`generate_structured` (Clase 3): una cita fuera de rango es una salida inválida más y se
repara una vez con el mismo mecanismo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from contracts.structured_output import generate_structured
from prompting import PromptKit

if TYPE_CHECKING:
    from core.llm_client import LLMClient
    from knowledge.store import Hit, VectorStore

UMBRAL = 0.6  # distancia coseno máxima; ilustrativo, se calibra con el golden set
NO_ENCONTRADO = "No encontré eso en los documentos de operaciones."
COMPANY = "Acme Operaciones"
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
MAX_REPAIRS = 1


class CitaInvalida(ValueError):
    """El modelo citó un documento que no está en el contexto."""


class RagAnswer(BaseModel):
    encontrado: bool
    respuesta: str
    citas: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _citas_validas(self, info: ValidationInfo) -> RagAnswer:
        if self.encontrado and not self.citas:
            raise ValueError("si encontrado=true, cita al menos un documento")
        n_hits = (info.context or {}).get("n_hits")
        if n_hits is not None:
            fuera = [c for c in self.citas if not 1 <= c <= n_hits]
            if fuera:
                raise CitaInvalida(
                    f"citas fuera de rango: {fuera}; solo existen los documentos 1 a {n_hits}"
                )
        return self


@dataclass(frozen=True)
class RagResult:
    encontrado: bool
    respuesta: str
    fuentes: list[dict] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    @classmethod
    def no_encontrado(cls, pregunta: str) -> RagResult:
        return cls(encontrado=False, respuesta=NO_ENCONTRADO, fuentes=[], chunk_ids=[])


def etiqueta(metadatos: dict) -> str:
    """`{fuente} › {seccion}` (Markdown), `{fuente} · p. {pagina}` (PDF) o `{fuente}`."""
    fuente = metadatos.get("fuente", "?")
    if metadatos.get("seccion"):
        return f"{fuente} › {metadatos['seccion']}"
    if metadatos.get("pagina"):
        return f"{fuente} · p. {metadatos['pagina']}"
    return fuente


def rag_messages(pregunta: str, hits: list[Hit]) -> list[dict]:
    """System del prompt `rag_answer`, el bloque `<documentos>` numerado y la pregunta (S40)."""
    documentos = "\n\n".join(
        f"[{n}] {etiqueta(hit.metadatos)}\n{hit.texto}" for n, hit in enumerate(hits, start=1)
    )
    prompt = PromptKit(PROMPTS_DIR).get("rag_answer")
    return prompt.render(company=COMPANY, documentos=documentos, pregunta=pregunta)


def answer(
    client: LLMClient,
    store: VectorStore,
    pregunta: str,
    k: int = 5,
    where: dict | None = None,
    umbral: float = UMBRAL,
) -> RagResult:
    """Busca, filtra por `umbral`, responde con citas y arma las fuentes (S42)."""
    hits = [h for h in store.search(pregunta, k=k, where=where) if h.distancia <= umbral]
    if not hits:
        return RagResult.no_encontrado(pregunta)

    prompt = PromptKit(PROMPTS_DIR).get("rag_answer")
    chunk_ids = [h.chunk_id for h in hits]
    log_extra = {
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "rag_k": k,
        "rag_hits": len(hits),
        "rag_chunk_ids": chunk_ids,
    }
    salida = generate_structured(
        client,
        rag_messages(pregunta, hits),
        RagAnswer,
        max_repairs=MAX_REPAIRS,
        log_extra=log_extra,
        validation_context={"n_hits": len(hits)},
        **prompt.model_params,
    )
    if client.last_request_id:
        client.logger.annotate(client.last_request_id, rag_encontrado=salida.encontrado)
    if not salida.encontrado:
        return RagResult.no_encontrado(pregunta)

    fuentes = []
    for n in dict.fromkeys(salida.citas):  # Sin repetidos, en el orden en que se citaron.
        hit = hits[n - 1]
        fuente = {"n": n, "etiqueta": etiqueta(hit.metadatos), "chunk_id": hit.chunk_id}
        fuente.update(
            {
                key: hit.metadatos[key]
                for key in ("fuente", "seccion", "pagina")
                if key in hit.metadatos
            }
        )
        fuentes.append(fuente)
    return RagResult(True, salida.respuesta, fuentes, chunk_ids)
