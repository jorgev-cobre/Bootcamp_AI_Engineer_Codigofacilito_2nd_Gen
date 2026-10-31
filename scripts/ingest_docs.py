"""Ingesta idempotente de documentos al índice vectorial (S18).

    uv run scripts/ingest_docs.py data/docs
    uv run scripts/ingest_docs.py data/docs --embedder fake
    uv run scripts/ingest_docs.py data/docs --rebuild

Por archivo: si no está en el índice, lo indexa; si está con el mismo `hash`, lo salta;
si cambió, lo borra y lo reindexa. Un error en un archivo no detiene a los demás.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar los paquetes del repo al correr como script.

from knowledge.chunking import chunk_file  # noqa: E402
from knowledge.embeddings import make_embedder  # noqa: E402
from knowledge.ingest import corpus_files, load_path  # noqa: E402
from knowledge.store import (  # noqa: E402
    COLLECTION,
    STORE_PATH,
    EmbeddingMismatch,
    delete_store,
    open_store,
)


def plural(n: int, singular: str, plural_form: str) -> str:
    return f"{n} {singular if n == 1 else plural_form}"


def summary(
    archivos: int, nuevos: int, modificados: int, sin_cambios: int, chunks: int, errores: int
) -> str:
    return " · ".join(
        [
            f"ingesta: {plural(archivos, 'archivo', 'archivos')}",
            plural(nuevos, "nuevo", "nuevos"),
            plural(modificados, "modificado", "modificados"),
            f"{sin_cambios} sin cambios",
            plural(chunks, "chunk", "chunks"),
            plural(errores, "error", "errores"),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Indexa los documentos de operaciones.")
    parser.add_argument("root", type=Path, help="Carpeta del corpus, por ejemplo data/docs.")
    parser.add_argument("--embedder", choices=("gemini", "fake"), default="gemini")
    parser.add_argument("--rebuild", action="store_true", help="Borra el índice y reindexa todo.")
    parser.add_argument("--store", default=STORE_PATH, help=f"Carpeta del índice ({STORE_PATH}).")
    parser.add_argument("--collection", default=COLLECTION, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"Error: no existe la carpeta {args.root}", file=sys.stderr)
        return 1
    if args.rebuild:
        delete_store(args.store, args.collection)
    try:
        store = open_store(make_embedder(args.embedder), path=args.store, name=args.collection)
    except (EmbeddingMismatch, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    files = corpus_files(args.root)
    nuevos = modificados = sin_cambios = chunks = errores = 0
    for path in files:
        try:
            docs = load_path(path, args.root)
            if not docs:
                continue
            doc = docs[0]
            stored = store.stored_hash(doc.doc_id)
            if stored == doc.metadatos["hash"]:
                sin_cambios += 1
                continue
            chunks += store.index_document(doc, chunk_file(docs))
            if stored is None:
                nuevos += 1
            else:
                modificados += 1
        except Exception as e:  # Un archivo con problemas no detiene la ingesta.
            errores += 1
            print(f"Error en {path.relative_to(args.root)}: {e}", file=sys.stderr)

    print(summary(len(files), nuevos, modificados, sin_cambios, chunks, errores))
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
