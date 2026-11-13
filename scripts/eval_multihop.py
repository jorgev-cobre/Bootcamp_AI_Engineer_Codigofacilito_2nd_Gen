"""recall@k de las fuentes en preguntas de varios saltos (reto de la Clase 7).

    uv run scripts/eval_multihop.py
    uv run scripts/eval_multihop.py --modos vector hibrida hibrida+rerank --detalle
    uv run scripts/eval_multihop.py --embedder fake --report reports/multihop.md

Cada pregunta de data/golden/multihop_v1.jsonl necesita 2 o más documentos. Por modo:
`recall@k` es la fracción de las `fuentes_esperadas` que llegan al top-k, promediada
entre preguntas, y `completas` cuenta las preguntas en las que llegan todas. Los modos
son los de eval_retrieval.py; los que tienen `+` llaman al LLM una vez por pregunta y
por paso.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar los paquetes del repo al correr como script.

from knowledge.embeddings import make_embedder  # noqa: E402
from knowledge.store import COLLECTION, STORE_PATH, EmbeddingMismatch, open_store  # noqa: E402
from scripts.eval_retrieval import (  # noqa: E402
    MODOS,
    build_llm_client,
    load_golden,
    make_retriever,
)

DEFAULT_GOLDEN = ROOT / "data" / "golden" / "multihop_v1.jsonl"
K = 5


def recall(recuperadas: list[str], esperadas: list[str]) -> float:
    """Fracción de las fuentes esperadas que aparecen en las recuperadas."""
    return len(set(esperadas) & set(recuperadas)) / len(esperadas)


def faltantes(recuperadas: list[str], esperadas: list[str]) -> list[str]:
    return [fuente for fuente in esperadas if fuente not in recuperadas]


def evaluate(retriever, golden: list[dict], k: int) -> dict[str, list[str]]:
    """id de la pregunta → fuentes del top-k, en orden."""
    return {
        q["id"]: [h.metadatos["fuente"] for h in retriever.search(q["pregunta"], k=k)]
        for q in golden
    }


def format_table(resultados: dict[str, dict[str, list[str]]], golden: list[dict], k: int) -> str:
    filas = [f"| modo | recall@{k} | completas |", "|---|---|---|"]
    for modo, recuperadas in resultados.items():
        valores = [recall(recuperadas[q["id"]], q["fuentes_esperadas"]) for q in golden]
        completas = sum(valor == 1 for valor in valores)
        filas.append(f"| {modo} | {sum(valores) / len(golden):.2f} | {completas}/{len(golden)} |")
    return "\n".join(filas)


def format_detail(resultados: dict[str, dict[str, list[str]]], golden: list[dict]) -> str:
    """Qué fuente esperada no llegó, por pregunta y modo (— = llegaron todas)."""
    modos = list(resultados)
    filas = ["| id | " + " | ".join(modos) + " |", "|---" * (len(modos) + 1) + "|"]
    for q in golden:
        celdas = [
            ", ".join(faltantes(resultados[m][q["id"]], q["fuentes_esperadas"])) or "—"
            for m in modos
        ]
        filas.append(f"| {q['id']} | " + " | ".join(celdas) + " |")
    return "\n".join(filas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mide recall@k en preguntas de varios saltos.")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--embedder", choices=("gemini", "fake"), default="gemini")
    parser.add_argument("--modos", nargs="+", choices=MODOS, default=["vector", "hibrida"])
    parser.add_argument("--detalle", action="store_true", help="Fuentes que faltan, por pregunta.")
    parser.add_argument("--report", type=Path, help="Además, escribe las tablas en este archivo.")
    parser.add_argument("--store", default=STORE_PATH, help=f"Carpeta del índice ({STORE_PATH}).")
    parser.add_argument("--collection", default=COLLECTION, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        store = open_store(make_embedder(args.embedder), path=args.store, name=args.collection)
    except (EmbeddingMismatch, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if store.count() == 0:
        print(
            "El índice está vacío: corre `uv run scripts/ingest_docs.py data/docs`.",
            file=sys.stderr,
        )
        return 2

    modos = list(dict.fromkeys(args.modos))
    client = build_llm_client() if any("+" in modo for modo in modos) else None
    golden = load_golden(args.golden)
    resultados = {
        modo: evaluate(make_retriever(store, modo, client), golden, args.k) for modo in modos
    }

    titulo = (
        f"recall@{args.k} sobre {args.golden.name} "
        f"({len(golden)} preguntas, embedder {args.embedder})"
    )
    salida = [titulo, "", format_table(resultados, golden, args.k)]
    if args.detalle or args.report:
        salida += ["", "Fuentes esperadas que no llegaron al top-k:", ""]
        salida.append(format_detail(resultados, golden))
    texto = "\n".join(salida)
    print(texto)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"# Preguntas de varios saltos\n\n{texto}\n", encoding="utf-8")
        print(f"\nReporte: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
