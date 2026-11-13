from __future__ import annotations

from scripts.build_graph import main as build_main
from scripts.eval_multihop import DEFAULT_GOLDEN, faltantes, recall
from scripts.eval_multihop import main as multihop_main
from scripts.eval_retrieval import ROOT, load_golden
from scripts.ingest_docs import main as ingest_main


def test_recall_counts_each_expected_source_once():
    recuperadas = ["faq/a.txt", "faq/a.txt", "politicas/b.md"]
    assert recall(recuperadas, ["faq/a.txt", "contratos/c.pdf"]) == 0.5
    assert recall(recuperadas, ["faq/a.txt", "politicas/b.md"]) == 1.0


def test_missing_sources_keep_the_golden_order():
    esperadas = ["runbooks/x.md", "contratos/y.pdf", "politicas/z.md"]
    assert faltantes(["politicas/z.md"], esperadas) == ["runbooks/x.md", "contratos/y.pdf"]


def test_every_question_needs_two_existing_documents():
    golden = load_golden(DEFAULT_GOLDEN)
    assert len(golden) == 6
    for q in golden:
        assert len(q["fuentes_esperadas"]) >= 2, q["id"]
        for fuente in q["fuentes_esperadas"]:
            assert (ROOT / "data" / "docs" / fuente).exists(), fuente


def test_compares_modes_and_writes_the_report(tmp_path, capsys):
    store = str(tmp_path / "chroma")
    assert ingest_main([str(ROOT / "data" / "docs"), "--embedder", "fake", "--store", store]) == 0
    capsys.readouterr()

    report = tmp_path / "reports" / "multihop.md"
    args = ["--embedder", "fake", "--store", store, "--modos", "vector", "bm25", "hibrida"]
    assert multihop_main([*args, "--report", str(report)]) == 0

    out = capsys.readouterr().out
    assert "recall@5 sobre multihop_v1.jsonl (6 preguntas, embedder fake)" in out
    assert "| vector |" in out and "| bm25 |" in out and "| hibrida |" in out
    assert "| m6 |" in out
    assert report.read_text(encoding="utf-8").startswith("# Preguntas de varios saltos")


def test_graph_modes_read_the_graph_file(tmp_path, capsys):
    store = str(tmp_path / "chroma")
    assert ingest_main([str(ROOT / "data" / "docs"), "--embedder", "fake", "--store", store]) == 0
    grafo = tmp_path / "grafo.json"
    args = ["--embedder", "fake", "--store", store, "--grafo", str(grafo), "--modos", "grafo"]
    assert multihop_main(args) == 2
    assert "scripts/build_graph.py" in capsys.readouterr().err

    assert build_main(["--out", str(grafo)]) == 0
    assert multihop_main([*args[:-1], "bm25", "grafo", "bm25+grafo"]) == 0
    out = capsys.readouterr().out
    assert "| grafo | 0.67 | 4/6 |" in out
    assert "| bm25 | 0.75 | 3/6 |" in out
