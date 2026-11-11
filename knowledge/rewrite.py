"""Reescritura de la pregunta (multi-query): varias consultas encuentran más que una.

El usuario pregunta como habla ("¿cuánto se tardan los de pagos en contestar?") y el
documento está escrito en otro registro ("tiempo de respuesta de PagosRed"). El LLM
convierte la pregunta en consultas con el vocabulario de los documentos; cada consulta
aporta su propio ranking a la fusión.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from contracts.structured_output import generate_structured
from knowledge.lexical import tokenize
from knowledge.rag import COMPANY, PROMPTS_DIR
from prompting import PromptKit

if TYPE_CHECKING:
    from core.llm_client import LLMClient

MAX_CONSULTAS = 3


class Reescritura(BaseModel):
    consultas: list[str] = Field(min_length=1)


def _clave(texto: str) -> str:
    return " ".join(sorted(set(tokenize(texto))))


def rewrite_query(client: LLMClient, pregunta: str, n: int = MAX_CONSULTAS) -> list[str]:
    """Hasta `n` consultas nuevas: distintas de la pregunta y distintas entre sí."""
    prompt = PromptKit(PROMPTS_DIR).get("query_rewrite")
    salida = generate_structured(
        client,
        prompt.render(company=COMPANY, pregunta=pregunta, n=n),
        Reescritura,
        max_repairs=1,
        log_extra={"prompt_name": prompt.name, "prompt_version": prompt.version},
        **prompt.model_params,
    )
    vistas = {_clave(pregunta)}
    consultas = []
    for consulta in salida.consultas:
        clave = _clave(consulta)
        if clave and clave not in vistas:
            vistas.add(clave)
            consultas.append(consulta.strip())
    return consultas[:n]
