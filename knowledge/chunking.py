"""Chunking: `Document` → `Chunk` con tres estrategias, elegidas por tipo de documento.

Los tokens se estiman con `estimate_tokens` de la Clase 2 (`ceil(caracteres / 4)`): es una
estimación, pero es la misma que usa el `ContextManager`, así que los presupuestos cuadran.
Ningún chunk supera `max_tokens` y, para un mismo archivo, la salida es determinista.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from context.manager import estimate_tokens
from knowledge.ingest import Document

SEPARATORS = ("\n\n", "\n", ". ")
FIXED_OVERLAP_RATIO = 0.15  # 60 tokens de traslape con chunks de 400.

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{doc_id}#{n}" con n desde 0
    texto: str
    metadatos: dict  # los del Document + doc_id, seccion?, n


def _chunks(doc: Document, pieces: list[tuple[str, str | None]], start: int = 0) -> list[Chunk]:
    chunks = []
    for offset, (texto, seccion) in enumerate(pieces):
        n = start + offset
        metadatos = {**doc.metadatos, "doc_id": doc.doc_id, "n": n}
        if seccion:
            metadatos["seccion"] = seccion
        chunks.append(Chunk(f"{doc.doc_id}#{n}", texto, metadatos))
    return chunks


def _split_tokens(texto: str, max_tokens: int) -> list[str]:
    """Último recurso: corta por palabras sin pasar de `max_tokens` (o por caracteres)."""
    pieces, current = [], ""
    for word in texto.split():
        while estimate_tokens(word) > max_tokens:  # Una "palabra" más larga que el límite.
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[: max_tokens * 4])
            word = word[max_tokens * 4 :]
        candidate = f"{current} {word}" if current else word
        if estimate_tokens(candidate) > max_tokens:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def split_recursivo(texto: str, max_tokens: int) -> list[str]:
    """Parte por `\\n\\n` → `\\n` → `. ` y, solo como último recurso, por tokens.

    Junta piezas vecinas mientras quepan en `max_tokens`, así que respeta párrafos y
    oraciones siempre que puede.
    """
    texto = texto.strip()
    if not texto:
        return []
    if estimate_tokens(texto) <= max_tokens:
        return [texto]
    for sep in SEPARATORS:
        if sep not in texto:
            continue
        parts = [p for p in texto.split(sep) if p.strip()]
        result, current = [], ""
        for part in parts:
            # Al partir por ". " el punto se queda con su oración.
            piece = part + "." if sep == ". " and not part.endswith(".") else part
            candidate = f"{current}{sep if sep != '. ' else ' '}{piece}" if current else piece
            if estimate_tokens(candidate) <= max_tokens:
                current = candidate
                continue
            if current:
                result.append(current.strip())
            if estimate_tokens(piece) <= max_tokens:
                current = piece
            else:
                result.extend(split_recursivo(piece, max_tokens))
                current = ""
        if current.strip():
            result.append(current.strip())
        return result
    return _split_tokens(texto, max_tokens)


def split_por_encabezados(texto: str) -> list[tuple[str, str]]:
    """Secciones de un Markdown: `[(ruta, cuerpo)]`.

    `ruta` son los encabezados vigentes unidos con " > ", sin `#`
    ("Escalamiento > Prioridad alta"). `cuerpo` empieza con su línea de encabezado.
    """
    sections: list[tuple[str, list[str]]] = []
    path: list[tuple[int, str]] = []
    for line in texto.splitlines():
        match = _HEADING.match(line)
        if match:
            level, title = len(match.group(1)), match.group(2)
            path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, title)]
            sections.append((" > ".join(t for _, t in path), [line]))
        elif sections:
            sections[-1][1].append(line)
        else:
            sections.append(("", [line]))
    return [
        (ruta, "\n".join(lines).strip()) for ruta, lines in sections if "\n".join(lines).strip()
    ]


def chunk_fixed(doc: Document, max_tokens: int = 400, overlap: int = 60) -> list[Chunk]:
    """Ventanas de `max_tokens` con `overlap` tokens de traslape. No mira la estructura."""
    words = doc.texto.split()
    pieces: list[str] = []
    start = 0
    while start < len(words):
        end = start
        while end < len(words) and estimate_tokens(" ".join(words[start : end + 1])) <= max_tokens:
            end += 1
        if end == start:  # Una palabra más larga que el límite.
            pieces.extend(_split_tokens(words[start], max_tokens))
            start += 1
            continue
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        back = end
        while back > start + 1 and estimate_tokens(" ".join(words[back - 1 : end])) <= overlap:
            back -= 1
        start = back if back < end else end
    return _chunks(doc, [(p, None) for p in pieces])


def chunk_recursive(doc: Document, max_tokens: int = 400) -> list[Chunk]:
    """Respeta párrafos y oraciones: `split_recursivo` sobre el texto completo."""
    return _chunks(doc, [(p, None) for p in split_recursivo(doc.texto, max_tokens)])


def _markdown_pieces(doc: Document, max_tokens: int) -> list[tuple[str, str | None]]:
    pieces: list[tuple[str, str | None]] = []
    for ruta, cuerpo in split_por_encabezados(doc.texto):
        heading, _, body = cuerpo.partition("\n")
        if not _HEADING.match(heading):
            heading, body = "", cuerpo
        if not body.strip():
            continue  # Encabezado sin texto propio: su contenido está en las subsecciones.
        if estimate_tokens(cuerpo) <= max_tokens:
            pieces.append((cuerpo, ruta or None))
            continue
        # Sección larga: se parte y cada pedazo conserva su encabezado al inicio.
        budget = max_tokens - estimate_tokens(heading + "\n") if heading else max_tokens
        for part in split_recursivo(body, budget):
            pieces.append((f"{heading}\n{part}" if heading else part, ruta or None))
    return pieces


def chunk_markdown(doc: Document, max_tokens: int = 400) -> list[Chunk]:
    """Un chunk por sección; `seccion` = ruta de encabezados (S23)."""
    return _chunks(doc, _markdown_pieces(doc, max_tokens))


def chunk_document(doc: Document, max_tokens: int = 400) -> list[Chunk]:
    """Elige la estrategia por `tipo_doc`: runbook y politica → Markdown; contrato →
    recursiva; faq → fija (con 15% de traslape)."""
    tipo = doc.metadatos.get("tipo_doc")
    if tipo in ("runbook", "politica"):
        return chunk_markdown(doc, max_tokens)
    if tipo == "contrato":
        return chunk_recursive(doc, max_tokens)
    return chunk_fixed(doc, max_tokens, overlap=round(max_tokens * FIXED_OVERLAP_RATIO))


def chunk_file(docs: list[Document], max_tokens: int = 400) -> list[Chunk]:
    """Chunks de todos los `Document` de un archivo (las páginas de un PDF), con `n`
    continuo entre páginas para que `chunk_id` sea único por archivo."""
    chunks: list[Chunk] = []
    for doc in docs:
        for chunk in chunk_document(doc, max_tokens):
            n = len(chunks)
            chunks.append(Chunk(f"{doc.doc_id}#{n}", chunk.texto, {**chunk.metadatos, "n": n}))
    return chunks
