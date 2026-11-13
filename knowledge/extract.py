"""Extracción de entidades y relaciones con el LLM (Clase 8).

Las reglas solo encuentran nombres que ya conocen. El LLM lee cada chunk y regresa las
entidades que nombra y las relaciones que afirma, con un esquema cerrado: solo estos
tipos y estas relaciones, cada una entre los tipos que le tocan. Después, la resolución
de entidades decide a qué nodo del grafo corresponde cada nombre extraído.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from contracts.structured_output import generate_structured
from knowledge.graph import KnowledgeGraph, nodo_id
from knowledge.rag import COMPANY, PROMPTS_DIR
from prompting import PromptKit

if TYPE_CHECKING:
    from core.llm_client import LLMClient

CACHE_PATH = Path(".cache") / "extracciones.json"
RELACIONES = {  # relación → (tipo de origen, tipo de destino)
    "depende_de": ("ticket", "proveedor"),
    "afecta": ("ticket", "servicio"),
    "presta": ("proveedor", "servicio"),
    "responsable_de": ("equipo", "servicio"),
    "escala_a": ("guardia", "guardia"),
}

_GUARDIA = re.compile(r"N[123]")
_ARTICULO = re.compile(r"^(el|la|los|las)\s+")


class Entidad(BaseModel):
    nombre: str = Field(min_length=1)
    tipo: Literal["ticket", "proveedor", "equipo", "guardia", "servicio"]


class Relacion(BaseModel):
    origen: str
    relacion: Literal["depende_de", "afecta", "presta", "responsable_de", "escala_a"]
    destino: str


class Extraccion(BaseModel):
    entidades: list[Entidad]
    relaciones: list[Relacion]

    @model_validator(mode="after")
    def _tipos_de_cada_relacion(self) -> Extraccion:
        tipos = {e.nombre: e.tipo for e in self.entidades}
        errores = []
        for r in self.relaciones:
            esperado = RELACIONES[r.relacion]
            if (tipos.get(r.origen), tipos.get(r.destino)) != esperado:
                errores.append(
                    f"{r.origen} —{r.relacion}→ {r.destino} debe ir de un {esperado[0]} a un "
                    f"{esperado[1]} declarados en entidades"
                )
        if errores:
            raise ValueError("; ".join(errores))
        return self


class ExtractionCache:
    """Extracciones ya pagadas, por prompt y texto: reconstruir el grafo no repite llamadas."""

    def __init__(self, path: Path | str = CACHE_PATH) -> None:
        self.path = Path(path)
        self.datos = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.aciertos = 0
        self.nuevas = 0

    @staticmethod
    def key(prompt: str, version: int, texto: str) -> str:
        return hashlib.sha256(f"{prompt}|{version}|{texto}".encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        valor = self.datos.get(key)
        self.aciertos += valor is not None
        return valor

    def put(self, key: str, valor: dict) -> None:
        self.datos[key] = valor
        self.nuevas += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.datos, ensure_ascii=False), encoding="utf-8")


def extraer(
    client: LLMClient, kg: KnowledgeGraph, texto: str, cache: ExtractionCache | None = None
) -> Extraccion:
    """Una llamada al LLM por chunk (o ninguna, si ya está en la caché)."""
    prompt = PromptKit(PROMPTS_DIR).get("graph_extract")
    clave_cache = ExtractionCache.key(prompt.name, prompt.version, texto)
    if cache is not None and (guardada := cache.get(clave_cache)) is not None:
        return Extraccion.model_validate(guardada)
    conocidos = {
        tipo: ", ".join(kg.nombre(n) for n in kg.entidades(tipo))
        for tipo in ("proveedor", "equipo")
    }
    salida = generate_structured(
        client,
        prompt.render(
            company=COMPANY,
            proveedores=conocidos["proveedor"],
            equipos=conocidos["equipo"],
            texto=texto,
        ),
        Extraccion,
        max_repairs=1,
        log_extra={"prompt_name": prompt.name, "prompt_version": prompt.version},
        **prompt.model_params,
    )
    if cache is not None:
        cache.put(clave_cache, salida.model_dump())
    return salida


def resolver(kg: KnowledgeGraph, entidad: Entidad) -> str | None:
    """El nodo del grafo que corresponde a una entidad extraída (resolución de entidades).

    Tickets, proveedores, equipos y guardias deben existir: un T-9999 o un proveedor que no
    está en proveedores.json se descarta. Un servicio nuevo se crea con su nombre en
    minúsculas y sin artículo, y los nombres equivalentes caen en el mismo nodo.
    """
    if entidad.tipo == "ticket":
        nodo = nodo_id("ticket", entidad.nombre.strip().upper())
        return nodo if nodo in kg.g else None
    if entidad.tipo == "guardia":
        nivel = _GUARDIA.search(entidad.nombre.upper())
        return nodo_id("guardia", nivel.group()) if nivel else None
    existente = kg.buscar(entidad.tipo, entidad.nombre)
    if existente is not None or entidad.tipo != "servicio":
        return existente
    return kg.add_entidad("servicio", _ARTICULO.sub("", " ".join(entidad.nombre.lower().split())))


def add_extraccion(kg: KnowledgeGraph, chunk: str, extraccion: Extraccion) -> None:
    """Menciones y relaciones del chunk, cada una con el chunk como fuente."""
    fuente = kg.nombre(chunk)
    nodos = {}
    for entidad in extraccion.entidades:
        nodo = resolver(kg, entidad)
        if nodo is not None:
            nodos[entidad.nombre] = nodo
            kg.add_relacion(chunk, "menciona", nodo, fuente)
    for r in extraccion.relaciones:
        if r.origen in nodos and r.destino in nodos:
            kg.add_relacion(nodos[r.origen], r.relacion, nodos[r.destino], fuente)
