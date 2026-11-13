"""Construye el grafo de conocimiento y lo guarda en var/grafo.json (Clase 8).

    uv run scripts/build_graph.py
    uv run scripts/build_graph.py --llm
    uv run scripts/build_graph.py --docs data/docs --out var/grafo.json

Sin --llm: datos de tickets_db.json y proveedores.json, y menciones por reglas en los
chunks de data/docs (los mismos que indexa ingest_docs.py). Con --llm, además, una
llamada al LLM por chunk para extraer servicios y relaciones; las extracciones se guardan
en .cache/extracciones.json y no se vuelven a pagar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar los paquetes del repo al correr como script.

from knowledge.extract import CACHE_PATH, ExtractionCache  # noqa: E402
from knowledge.graph import GRAPH_PATH  # noqa: E402
from knowledge.graph_build import build_graph, load_chunks  # noqa: E402


def describe(resumen: dict[str, int]) -> str:
    nodos = ", ".join(f"{n} {tipo}" for tipo, n in resumen.items() if tipo != "relaciones")
    return f"{nodos} · {resumen['relaciones']} relaciones"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Construye el grafo de conocimiento.")
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "docs")
    parser.add_argument("--out", type=Path, default=GRAPH_PATH)
    parser.add_argument("--llm", action="store_true", help="Extrae servicios y relaciones.")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    client = cache = None
    if args.llm:
        from scripts.eval_retrieval import build_llm_client

        client, cache = build_llm_client(), ExtractionCache(args.cache)
    kg = build_graph(load_chunks(args.docs), client, cache)
    kg.save(args.out)
    print(f"Grafo: {describe(kg.resumen())} → {args.out}")
    if cache is not None:
        print(f"Extracción con LLM: {cache.nuevas} llamadas, {cache.aciertos} de la caché")
    return 0


if __name__ == "__main__":
    sys.exit(main())
