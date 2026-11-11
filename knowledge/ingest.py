"""Ingesta: archivos de data/docs → `Document` con texto limpio y metadatos.

Un `Document` por archivo (Markdown y TXT) o por página (PDF). El `doc_id` es la ruta
relativa a la raíz del corpus, siempre con "/": es estable entre ejecuciones y sistemas
y es la llave para borrar y reindexar un archivo completo.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

PROVEEDORES_PATH = Path(__file__).resolve().parents[1] / "data" / "proveedores.json"
PAGE_BREAK = "\f"  # Separador de páginas dentro del texto completo de un PDF.
REPEATED_LINE_RATIO = 0.6  # Una línea en ≥ 60% de las páginas es encabezado o pie.
EDGE_LINES = 2  # Líneas del borde de cada página donde se buscan "Página N de M".

_HYPHENATED = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True)
class Document:
    doc_id: str  # ruta relativa a data/docs, con "/" (estable entre ejecuciones y sistemas)
    texto: str
    metadatos: dict  # fuente, tipo_doc, titulo, pagina?, proveedor?, hash


class IngestError(Exception):
    """El archivo no se puede ingerir: formato no soportado, PDF ilegible, etc."""


def clean_text(texto: str) -> str:
    """Limpia el texto de un archivo completo con tres reglas, en este orden.

    1. Une palabras partidas con guion al final de línea ("res-\\npuesta" → "respuesta").
    2. Quita las líneas que se repiten idénticas en ≥ 60% de las páginas (encabezados y
       pies). Las páginas van separadas por `\\f`. En las dos primeras y dos últimas
       líneas de cada página los números no cuentan, para que "Página 1 de 4" y
       "Página 2 de 4" sean la misma línea. Con una sola página no quita nada.
    3. Colapsa espacios y saltos de línea múltiples.

    Normaliza a Unicode NFC antes de todo.
    """
    texto = unicodedata.normalize("NFC", texto)
    texto = _HYPHENATED.sub(r"\1\2", texto)

    pages = texto.split(PAGE_BREAK)
    if len(pages) > 1:
        threshold = REPEATED_LINE_RATIO * len(pages)
        exact = Counter(
            line for page in pages for line in {_line_key(x) for x in page.splitlines()}
        )
        repeated = {line for line, n in exact.items() if line and n >= threshold}
        # "Página 1 de 4" y "Página 2 de 4" solo difieren en números: se comparan sin ellos,
        # pero solo en los bordes de la página, donde viven encabezados y pies.
        edges = Counter(
            key for page in pages for key in {_DIGITS.sub("#", x) for x in _edge_lines(page)}
        )
        numbered = {key for key, n in edges.items() if key and n >= threshold}
        pages = [
            "\n".join(
                line
                for index, line in enumerate(page.splitlines())
                if _line_key(line) not in repeated
                and not (_is_edge(page, index) and _DIGITS.sub("#", _line_key(line)) in numbered)
            )
            for page in pages
        ]

    cleaned = []
    for page in pages:
        lines = [_SPACES.sub(" ", line).strip() for line in page.splitlines()]
        cleaned.append(_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip())
    return PAGE_BREAK.join(cleaned)


def _line_key(line: str) -> str:
    return _SPACES.sub(" ", line).strip()


def _edge_lines(page: str) -> list[str]:
    lines = page.splitlines()
    return [_line_key(line) for index, line in enumerate(lines) if _is_edge(page, index)]


def _is_edge(page: str, index: int) -> bool:
    """Las dos primeras y las dos últimas líneas de la página."""
    total = len(page.splitlines())
    return index < EDGE_LINES or index >= total - EDGE_LINES


def _hash(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _tipo_doc(doc_id: str) -> str:
    """Singular de la carpeta: runbooks → runbook, politicas → politica, etc."""
    carpeta = doc_id.split("/", 1)[0] if "/" in doc_id else ""
    return carpeta.removesuffix("s") if carpeta else "documento"


def _readable_name(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").capitalize()


def _slug(texto: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def _proveedor(path: Path) -> str:
    """Nombre del proveedor con el casing de proveedores.json, a partir del archivo."""
    nombres = [p["nombre"] for p in json.loads(PROVEEDORES_PATH.read_text(encoding="utf-8"))]
    by_slug = {_slug(nombre): nombre for nombre in nombres}
    nombre = by_slug.get(_slug(path.stem))
    if nombre is None:
        raise IngestError(f"proveedor desconocido en el nombre del contrato: {path.name}")
    return nombre


def _metadatos(doc_id: str, path: Path, texto: str, titulo: str) -> dict:
    metadatos = {
        "fuente": doc_id,
        "tipo_doc": _tipo_doc(doc_id),
        "titulo": titulo,
        "hash": _hash(texto),
    }
    if metadatos["tipo_doc"] == "contrato":
        metadatos["proveedor"] = _proveedor(path)
    return metadatos


def load_markdown(path: Path, doc_id: str) -> list[Document]:
    texto = clean_text(path.read_text(encoding="utf-8"))
    titulo = next(
        (line[2:].strip() for line in texto.splitlines() if line.startswith("# ")),
        _readable_name(path),
    )
    return [Document(doc_id, texto, _metadatos(doc_id, path, texto, titulo))]


def load_text(path: Path, doc_id: str) -> list[Document]:
    texto = clean_text(path.read_text(encoding="utf-8"))
    return [Document(doc_id, texto, _metadatos(doc_id, path, texto, _readable_name(path)))]


def load_pdf(path: Path, doc_id: str) -> list[Document]:
    """Un `Document` por página (`pagina` desde 1); todas comparten `doc_id` y `hash`."""
    try:
        reader = PdfReader(path)
        raw = PAGE_BREAK.join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise IngestError(f"no se pudo leer el PDF {doc_id}: {e}") from e
    texto = clean_text(raw)
    titulo = path.stem
    if _tipo_doc(doc_id) == "contrato":
        titulo = f"Contrato de servicio {_proveedor(path)}"
    base = _metadatos(doc_id, path, texto.replace(PAGE_BREAK, "\n\n"), titulo)
    return [
        Document(doc_id, page, {**base, "pagina": number})
        for number, page in enumerate(texto.split(PAGE_BREAK), start=1)
        if page.strip()
    ]


LOADERS: dict[str, Callable[[Path, str], list[Document]]] = {
    ".md": load_markdown,
    ".txt": load_text,
    ".pdf": load_pdf,
}


def load_path(path: Path, root: Path) -> list[Document]:
    """Carga un archivo; `doc_id` es su ruta relativa a `root` con "/"."""
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise IngestError(f"formato no soportado: {path.suffix.lower() or path.name}")
    doc_id = path.relative_to(root).as_posix()
    return loader(path, doc_id)


def corpus_files(root: Path) -> list[Path]:
    """Archivos del corpus en orden estable (ordenados por ruta)."""
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))


def load_dir(root: Path) -> list[Document]:
    """Todos los documentos de `root`, recorridos en orden estable."""
    return [doc for path in corpus_files(root) for doc in load_path(path, root)]
