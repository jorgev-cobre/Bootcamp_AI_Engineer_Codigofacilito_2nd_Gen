from __future__ import annotations

from scripts.eval_retrieval import ROOT, make_retriever, posicion, usa_llm
from scripts.eval_retrieval import main as eval_main
from scripts.ingest_docs import main as ingest_main


def test_position_of_the_expected_source():
    assert posicion(["faq/a.txt", "politicas/b.md", "faq/a.txt"], "faq/a.txt") == 1
    assert posicion(["politicas/b.md"], "faq/a.txt") is None


def test_mode_names_become_retriever_options():
    retriever = make_retriever(store=None, modo="vector")  # vector no lee el índice BM25
    assert (retriever.modo, retriever.con_reescritura, retriever.con_rerank) == (
        "vector",
        False,
        False,
    )


def test_the_graph_is_one_more_step_of_the_mode_name():
    grafo = object()
    assert make_retriever(None, "vector+grafo", grafo=grafo).grafo is grafo
    assert make_retriever(None, "vector", grafo=grafo).grafo is None
    assert usa_llm("hibrida+grafo+rerank") and not usa_llm("hibrida+grafo")


def test_compares_modes_on_the_golden_set(tmp_path, capsys):
    store = str(tmp_path / "chroma")
    assert ingest_main([str(ROOT / "data" / "docs"), "--embedder", "fake", "--store", store]) == 0
    capsys.readouterr()

    args = ["--embedder", "fake", "--store", store, "--modos", "vector", "bm25", "hibrida"]
    assert eval_main([*args, "--detalle"]) == 0

    out = capsys.readouterr().out
    assert "hit@k sobre rag_v3.jsonl (20 preguntas, embedder fake)" in out
    assert "\nvector " in out and "\nbm25 " in out and "\nhibrida " in out
    assert "Posición de la fuente esperada" in out
    assert "q16 ¿Cuál es el presupuesto de viáticos" in out
