"""hit@k del retrieval sobre el golden set, sin LLM.

    uv run scripts/eval_retrieval.py
    uv run scripts/eval_retrieval.py --k 1 3 5 --embedder fake

Una pregunta cuenta como acierto en k si algún hit del top-k viene de su
`fuente_esperada`. Para las preguntas fuera del corpus imprime la distancia del primer
hit: sirve para calibrar `UMBRAL` en knowledge/rag.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar los paquetes del repo al correr como script.

from knowledge.embeddings import make_embedder  # noqa: E402
from knowledge.rag import UMBRAL  # noqa: E402
from knowledge.store import COLLECTION, STORE_PATH, EmbeddingMismatch, open_store  # noqa: E402

DEFAULT_GOLDEN = ROOT / "data" / "golden" / "rag_v1.jsonl"


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def hit_at_k(fuentes: list[str], esperada: str, k: int) -> bool:
    return esperada in fuentes[:k]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mide hit@k del retrieval.")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--embedder", choices=("gemini", "fake"), default="gemini")
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

    golden = load_golden(args.golden)
    ks = sorted(set(args.k))
    k_max = ks[-1]
    preguntas = [q for q in golden if q.get("fuente_esperada")]
    fuera = [q for q in golden if q.get("fuera_de_corpus")]

    recuperadas: dict[str, list[str]] = {}
    for q in preguntas:
        hits = store.search(q["pregunta"], k=k_max)
        recuperadas[q["id"]] = [h.metadatos["fuente"] for h in hits]

    print(f"k · hit@k ({len(preguntas)} preguntas, embedder {args.embedder})")
    for k in ks:
        aciertos = sum(hit_at_k(recuperadas[q["id"]], q["fuente_esperada"], k) for q in preguntas)
        print(f"{k} · {aciertos / len(preguntas):.2f} ({aciertos}/{len(preguntas)})")

    falladas = [
        q for q in preguntas if not hit_at_k(recuperadas[q["id"]], q["fuente_esperada"], k_max)
    ]
    print(f"\nFalladas en k={k_max}: {len(falladas)}")
    for q in falladas:
        print(f"  {q['id']} {q['pregunta']}")
        fuentes = ", ".join(recuperadas[q["id"]])
        print(f"     esperada: {q['fuente_esperada']} · recuperadas: {fuentes}")

    if fuera:
        print(f"\nFuera del corpus (UMBRAL actual: {UMBRAL}):")
        for q in fuera:
            [primero] = store.search(q["pregunta"], k=1) or [None]
            if primero is None:
                print(f"  {q['id']} {q['pregunta']} → sin hits")
                continue
            veredicto = "se descarta" if primero.distancia > UMBRAL else "pasaría el umbral"
            print(
                f"  {q['id']} {q['pregunta']} → primer hit {primero.metadatos['fuente']} "
                f"a distancia {primero.distancia:.3f} ({veredicto})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
