"""hit@k del retrieval sobre el golden set, por modo de búsqueda.

    uv run scripts/eval_retrieval.py
    uv run scripts/eval_retrieval.py --k 1 3 5 --embedder fake
    uv run scripts/eval_retrieval.py --modos vector bm25 hibrida --detalle
    uv run scripts/eval_retrieval.py --modos hibrida hibrida+rerank

Una pregunta cuenta como acierto en k si algún hit del top-k viene de su
`fuente_esperada`. Para las preguntas fuera del corpus imprime el primer hit de cada modo
y su distancia: sirve para calibrar `UMBRAL` en knowledge/rag.py. Los modos con
`reescritura` o `rerank` llaman al LLM (una llamada por pregunta y por paso).
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
from knowledge.retrieval import Retriever  # noqa: E402
from knowledge.store import COLLECTION, STORE_PATH, EmbeddingMismatch, open_store  # noqa: E402

DEFAULT_GOLDEN = ROOT / "data" / "golden" / "rag_v3.jsonl"
MODOS = (
    "vector",
    "bm25",
    "hibrida",
    "hibrida+reescritura",
    "hibrida+rerank",
    "hibrida+reescritura+rerank",
)


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def hit_at_k(fuentes: list[str], esperada: str, k: int) -> bool:
    return esperada in fuentes[:k]


def posicion(fuentes: list[str], esperada: str) -> int | None:
    """Posición (desde 1) del primer hit de la fuente esperada, o None si no llegó."""
    return fuentes.index(esperada) + 1 if esperada in fuentes else None


def make_retriever(store, modo: str, client=None) -> Retriever:
    base, *pasos = modo.split("+")
    return Retriever(
        store,
        modo=base,
        client=client,
        reescribir="reescritura" in pasos,
        rerank="rerank" in pasos,
    )


def build_llm_client():
    from core.config import get_settings
    from core.llm_client import LLMClient

    return LLMClient.from_settings(get_settings())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mide hit@k del retrieval.")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--embedder", choices=("gemini", "fake"), default="gemini")
    parser.add_argument("--modos", nargs="+", choices=MODOS, default=["vector"])
    parser.add_argument("--detalle", action="store_true", help="Posición por pregunta y modo.")
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
    ks = sorted(set(args.k))
    k_max = ks[-1]
    preguntas = [q for q in golden if q.get("fuente_esperada")]
    fuera = [q for q in golden if q.get("fuera_de_corpus")]

    # recuperadas[modo][id] = fuentes del top-k_max, en orden
    recuperadas: dict[str, dict[str, list[str]]] = {}
    primeros: dict[str, dict[str, str]] = {}
    for modo in modos:
        retriever = make_retriever(store, modo, client)
        recuperadas[modo] = {
            q["id"]: [h.metadatos["fuente"] for h in retriever.search(q["pregunta"], k=k_max)]
            for q in preguntas
        }
        primeros[modo] = {}
        for q in fuera:
            hits = retriever.search(q["pregunta"], k=1)
            if not hits:
                primeros[modo][q["id"]] = "sin hits"
                continue
            primero = f"{hits[0].metadatos['fuente']} a {hits[0].distancia:.3f}"
            if modo == "vector":  # El umbral de distancia solo filtra en modo vector.
                primero += " (se descarta)" if hits[0].distancia > UMBRAL else " (pasa el umbral)"
            primeros[modo][q["id"]] = primero

    ancho = max(len(m) for m in modos)
    print(f"hit@k sobre {args.golden.name} ({len(preguntas)} preguntas, embedder {args.embedder})")
    print(f"\n{'modo'.ljust(ancho)}  " + "  ".join(f"hit@{k}" for k in ks))
    for modo in modos:
        valores = []
        for k in ks:
            aciertos = sum(
                hit_at_k(recuperadas[modo][q["id"]], q["fuente_esperada"], k) for q in preguntas
            )
            valores.append(f"{aciertos / len(preguntas):.2f}".ljust(len(f"hit@{k}")))
        print((f"{modo.ljust(ancho)}  " + "  ".join(valores)).rstrip())

    print(f"\nFalladas en k={k_max}:")
    for modo in modos:
        falladas = [
            q["id"]
            for q in preguntas
            if not hit_at_k(recuperadas[modo][q["id"]], q["fuente_esperada"], k_max)
        ]
        print(f"  {modo.ljust(ancho)}  {', '.join(falladas) or '—'}")

    if args.detalle:
        print(f"\nPosición de la fuente esperada (— = fuera del top-{k_max}):")
        print("  id   " + "  ".join(m.ljust(len(m)) for m in modos) + "  pregunta")
        for q in preguntas:
            celdas = [
                str(posicion(recuperadas[m][q["id"]], q["fuente_esperada"]) or "—").ljust(len(m))
                for m in modos
            ]
            print(f"  {q['id']}  " + "  ".join(celdas) + f"  {q['pregunta']}")

    if fuera:
        print(f"\nFuera del corpus, primer hit (UMBRAL actual: {UMBRAL}):")
        for q in fuera:
            print(f"  {q['id']} {q['pregunta']}")
            for modo in modos:
                print(f"     {modo.ljust(ancho)}  {primeros[modo][q['id']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
