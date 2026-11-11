from __future__ import annotations

from itertools import groupby
from pathlib import Path

import pytest

from context.manager import estimate_tokens
from knowledge.chunking import (
    CHUNK_TOKENS,
    chunk_document,
    chunk_file,
    chunk_fixed,
    chunk_markdown,
    chunk_recursive,
    split_por_encabezados,
    split_recursivo,
)
from knowledge.ingest import Document, load_dir, load_path

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"
RULE = (
    "Si el proveedor no responde en 4 horas, escalar a Guardia N2 y avisar al dueño del servicio."
)


@pytest.fixture(scope="module")
def corpus() -> list[Document]:
    return load_dir(DOCS)


@pytest.fixture(scope="module")
def escalamiento() -> Document:
    return load_path(DOCS / "politicas" / "escalamiento.md", DOCS)[0]


def test_split_por_encabezados_builds_heading_path():
    texto = "# Guía\nIntro\n## Uno\nA\n### Uno punto uno\nB\n## Dos\nC"
    assert split_por_encabezados(texto) == [
        ("Guía", "# Guía\nIntro"),
        ("Guía > Uno", "## Uno\nA"),
        ("Guía > Uno > Uno punto uno", "### Uno punto uno\nB"),
        ("Guía > Dos", "## Dos\nC"),
    ]


def test_chunk_markdown_keeps_section_path_and_heading(escalamiento):
    chunks = chunk_markdown(escalamiento)
    alta = next(c for c in chunks if c.metadatos.get("seccion") == "Escalamiento > Prioridad alta")
    assert alta.texto.startswith("## Prioridad alta\n")
    assert alta.metadatos["doc_id"] == "politicas/escalamiento.md"


def test_escalation_rule_whole_with_markdown_and_split_with_fixed(escalamiento):
    """El caso de S22 sobre el archivo real: el corte fijo separa la regla de su sección."""
    [md] = [c for c in chunk_markdown(escalamiento) if RULE in c.texto]
    assert md.texto.startswith("## Prioridad alta")

    fixed = chunk_fixed(escalamiento, max_tokens=400, overlap=60)
    with_heading = [c for c in fixed if "## Prioridad alta" in c.texto]
    with_rule = [c for c in fixed if RULE in c.texto]
    assert with_heading and with_rule
    assert not any(RULE in c.texto for c in with_heading)


@pytest.mark.parametrize("max_tokens", [100, 200, 400])
def test_no_chunk_exceeds_max_tokens(corpus, max_tokens):
    for doc in corpus:
        for strategy in (chunk_fixed, chunk_recursive, chunk_markdown):
            for chunk in strategy(doc, max_tokens):
                assert estimate_tokens(chunk.texto) <= max_tokens, (doc.doc_id, strategy.__name__)


def test_chunk_fixed_overlaps(escalamiento):
    first, second = chunk_fixed(escalamiento, max_tokens=200, overlap=40)[:2]
    assert " ".join(first.texto.split()[-5:]) in second.texto  # las últimas palabras se repiten


def test_split_recursivo_prefers_paragraphs():
    parrafos = ["uno " * 30, "dos " * 30, "tres " * 30]
    assert split_recursivo("\n\n".join(parrafos), max_tokens=40) == [p.strip() for p in parrafos]


def test_ids_are_deterministic(corpus):
    first = [(c.chunk_id, c.texto) for d in corpus for c in chunk_document(d)]
    second = [(c.chunk_id, c.texto) for d in load_dir(DOCS) for c in chunk_document(d)]
    assert first == second


def test_chunk_document_picks_strategy_by_tipo_doc(corpus):
    # Contra CHUNK_TOKENS y no contra 400: la decisión del reto puede cambiar el tamaño.
    by_tipo = {d.metadatos["tipo_doc"]: d for d in corpus}
    n = CHUNK_TOKENS
    assert chunk_document(by_tipo["runbook"]) == chunk_markdown(by_tipo["runbook"], n)
    assert chunk_document(by_tipo["politica"]) == chunk_markdown(by_tipo["politica"], n)
    assert chunk_document(by_tipo["contrato"]) == chunk_recursive(by_tipo["contrato"], n)
    traslape = round(n * 0.15)
    assert chunk_document(by_tipo["faq"]) == chunk_fixed(by_tipo["faq"], n, overlap=traslape)


def test_chunk_file_numbers_pages_continuously(corpus):
    for doc_id, docs in groupby(corpus, key=lambda d: d.doc_id):
        chunks = chunk_file(list(docs))
        assert [c.metadatos["n"] for c in chunks] == list(range(len(chunks)))
        assert [c.chunk_id for c in chunks] == [f"{doc_id}#{n}" for n in range(len(chunks))]
    pagosred = chunk_file([d for d in corpus if d.doc_id == "contratos/pagosred.pdf"])
    assert {c.metadatos["pagina"] for c in pagosred} == {1, 2, 3, 4}
