"""Construye el grafo de conocimiento y lo guarda en var/grafo.json (Clase 8).

    uv run scripts/build_graph.py
    uv run scripts/build_graph.py --docs data/docs --out var/grafo.json

Sin LLM: datos de tickets_db.json y proveedores.json, y menciones por reglas en los chunks
de data/docs. Los chunks son los mismos que indexa ingest_docs.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar los paquetes del repo al correr como script.

from knowledge.graph import GRAPH_PATH  # noqa: E402
from knowledge.graph_build import build_graph, load_chunks  # noqa: E402


def describe(resumen: dict[str, int]) -> str:
    nodos = ", ".join(f"{n} {tipo}" for tipo, n in resumen.items() if tipo != "relaciones")
    return f"{nodos} · {resumen['relaciones']} relaciones"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Construye el grafo de conocimiento.")
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "docs")
    parser.add_argument("--out", type=Path, default=GRAPH_PATH)
    args = parser.parse_args(argv)

    kg = build_graph(load_chunks(args.docs))
    kg.save(args.out)
    print(f"Grafo: {describe(kg.resumen())} → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
