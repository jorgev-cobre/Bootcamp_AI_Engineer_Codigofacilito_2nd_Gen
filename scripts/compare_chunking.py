"""Compara tamaños de chunk con el golden set: hit@k y distancias, sin LLM.

    uv run scripts/compare_chunking.py
    uv run scripts/compare_chunking.py --sizes 200 400 800 --embedder fake
    uv run scripts/compare_chunking.py --report reports/chunking.md

Cada tamaño vive en su propia colección del índice (`docs_operador_c200`, ...), así que la
colección principal no se toca. Un chunk que sale igual con dos tamaños tiene el mismo
texto, y su embedding sale de la caché: solo se pagan los chunks nuevos.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar los paquetes del repo al correr como script.

from knowledge.chunking import CHUNK_TOKENS, chunk_file  # noqa: E402
from knowledge.embeddings import make_embedder  # noqa: E402
from knowledge.ingest import corpus_files, load_path  # noqa: E402
from knowledge.rag import UMBRAL  # noqa: E402
from knowledge.store import (  # noqa: E402
    COLLECTION,
    STORE_PATH,
    EmbeddingMismatch,
    VectorStore,
    delete_store,
    open_store,
)
from scripts.eval_retrieval import hit_at_k, load_golden  # noqa: E402

DEFAULT_GOLDEN = ROOT / "data" / "golden" / "rag_v2.jsonl"
SIZES = (200, 400, 800)
KS = (1, 3, 5)


@dataclass(frozen=True)
class Resultado:
    size: int
    chunks: int
    total: int  # preguntas con fuente esperada
    aciertos: dict[int, int]  # k → preguntas con su fuente en el top-k
    peor_dentro: float  # mayor distancia del primer hit entre las preguntas del corpus
    fuera: dict[str, float]  # id → distancia del primer hit de cada pregunta fuera del corpus

    def hit(self, k: int) -> float:
        return self.aciertos[k] / self.total


def collection_name(size: int) -> str:
    return f"{COLLECTION}_c{size}"


def index_corpus(store: VectorStore, root: Path, size: int) -> int:
    """Indexa `root` con chunks de `size` tokens, saltando lo que no cambió."""
    for path in corpus_files(root):
        docs = load_path(path, root)
        if docs and store.stored_hash(docs[0].doc_id) != docs[0].metadatos["hash"]:
            store.index_document(docs[0], chunk_file(docs, max_tokens=size))
    return store.count()


def evaluate(store: VectorStore, golden: list[dict], size: int, chunks: int) -> Resultado:
    preguntas = [q for q in golden if q.get("fuente_esperada")]
    aciertos = dict.fromkeys(KS, 0)
    primeras = []
    for q in preguntas:
        hits = store.search(q["pregunta"], k=max(KS))
        fuentes = [h.metadatos["fuente"] for h in hits]
        for k in KS:
            aciertos[k] += hit_at_k(fuentes, q["fuente_esperada"], k)
        primeras.append(hits[0].distancia)
    fuera = {
        q["id"]: store.search(q["pregunta"], k=1)[0].distancia
        for q in golden
        if q.get("fuera_de_corpus")
    }
    return Resultado(size, chunks, len(preguntas), aciertos, max(primeras), fuera)


def recommend(resultados: list[Resultado], actual: int = CHUNK_TOKENS) -> Resultado:
    """Gana el mejor hit@5, que es lo que `--rag` le manda al LLM (k=5). Desempatan hit@1,
    luego hit@3 y, al final, quedarse con el tamaño actual."""
    return max(
        resultados, key=lambda r: (r.aciertos[5], r.aciertos[1], r.aciertos[3], r.size == actual)
    )


def suggest_threshold(resultado: Resultado) -> float | None:
    """Punto medio entre la peor distancia dentro del corpus y la mejor fuera de él.

    Regresa `None` si se traslapan: entonces la distancia sola no separa las preguntas y
    el umbral no se mueve.
    """
    if not resultado.fuera:
        return None
    mejor_fuera = min(resultado.fuera.values())
    if resultado.peor_dentro >= mejor_fuera:
        return None
    return round((resultado.peor_dentro + mejor_fuera) / 2, 2)


def format_table(resultados: list[Resultado]) -> str:
    ids = sorted({qid for r in resultados for qid in r.fuera})
    header = ["tamaño", "chunks", *(f"hit@{k}" for k in KS), "peor dentro", *ids]
    rows = [
        [
            str(r.size),
            str(r.chunks),
            *(f"{r.hit(k):.2f}" for k in KS),
            f"{r.peor_dentro:.3f}",
            *(f"{r.fuera[qid]:.3f}" for qid in ids),
        ]
        for r in resultados
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def decision(resultados: list[Resultado]) -> tuple[Resultado, float | None, str]:
    elegido = recommend(resultados)
    umbral = suggest_threshold(elegido)
    otros = "; ".join(
        f"{r.size} → hit@5 {r.hit(5):.2f}, hit@1 {r.hit(1):.2f}"
        for r in resultados
        if r is not elegido
    )
    texto = (
        f"Tamaño por defecto: {elegido.size} tokens, con hit@5 {elegido.hit(5):.2f} "
        f"({elegido.aciertos[5]}/{elegido.total}) y hit@1 {elegido.hit(1):.2f} "
        f"({elegido.aciertos[1]}/{elegido.total}). Los otros: {otros}. Regla: gana el mejor "
        "hit@5, porque `--rag` le manda 5 chunks al LLM; desempatan hit@1, luego hit@3 y, al "
        f"final, el tamaño actual ({CHUNK_TOKENS})."
    )
    if umbral is None:
        texto += (
            f" UMBRAL: se queda en {UMBRAL}. La peor distancia dentro del corpus "
            f"({elegido.peor_dentro:.3f}) no queda por debajo de la mejor fuera del corpus "
            f"({min(elegido.fuera.values(), default=float('nan')):.3f}): la distancia sola no "
            "separa, y esas preguntas las resuelve el LLM con encontrado=false."
        )
    else:
        texto += (
            f" UMBRAL: {umbral}, el punto medio entre la peor distancia dentro del corpus "
            f"({elegido.peor_dentro:.3f}) y la mejor fuera de él "
            f"({min(elegido.fuera.values()):.3f})."
        )
    return elegido, umbral, texto


def report(resultados: list[Resultado], golden: Path, embedder: str) -> str:
    _, _, texto = decision(resultados)
    return (
        "# Tamaño de chunk y umbral\n\n"
        f"Generado con `uv run scripts/compare_chunking.py` sobre `{golden.name}` "
        f"con el embedder `{embedder}`. Las columnas `q…` son la distancia del primer hit de "
        "cada pregunta fuera del corpus; `peor dentro` es la mayor distancia del primer hit "
        "entre las preguntas del corpus.\n\n"
        f"{format_table(resultados)}\n\n"
        "## Decisión\n\n"
        f"{texto}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara tamaños de chunk con el golden set.")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--embedder", choices=("gemini", "fake"), default="gemini")
    parser.add_argument("--store", default=STORE_PATH, help=f"Carpeta del índice ({STORE_PATH}).")
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "docs")
    parser.add_argument("--rebuild", action="store_true", help="Reindexa cada tamaño desde cero.")
    parser.add_argument("--report", type=Path, help="Escribe el reporte en Markdown aquí.")
    args = parser.parse_args(argv)

    golden = load_golden(args.golden)
    embedder = make_embedder(args.embedder)
    resultados = []
    for size in sorted(set(args.sizes)):
        if args.rebuild:
            delete_store(args.store, collection_name(size))
        try:
            store = open_store(embedder, path=args.store, name=collection_name(size))
        except EmbeddingMismatch as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        chunks = index_corpus(store, args.docs, size)
        resultados.append(evaluate(store, golden, size, chunks))

    print(format_table(resultados))
    elegido, umbral, texto = decision(resultados)
    print(f"\n{texto}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report(resultados, args.golden, args.embedder), encoding="utf-8")
        print(f"\nReporte: {args.report}")
    if elegido.size != CHUNK_TOKENS or (umbral is not None and umbral != UMBRAL):
        print(
            f"Para aplicarlo: CHUNK_TOKENS = {elegido.size} en knowledge/chunking.py, "
            f"UMBRAL = {umbral if umbral is not None else UMBRAL} en knowledge/rag.py y "
            "`uv run scripts/ingest_docs.py data/docs --rebuild`."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
