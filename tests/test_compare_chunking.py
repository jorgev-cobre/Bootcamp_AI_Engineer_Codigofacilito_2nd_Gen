from __future__ import annotations

from scripts.compare_chunking import Resultado, collection_name, main, recommend, suggest_threshold


def resultado(size: int, a1: int, a3: int, a5: int, peor: float = 0.4, fuera: float = 0.8):
    return Resultado(size, 10, 14, {1: a1, 3: a3, 5: a5}, peor, {"q11": fuera})


def test_best_hit5_wins_and_hit1_breaks_ties():
    r200, r400, r800 = (
        resultado(200, 9, 12, 14),
        resultado(400, 8, 13, 14),
        resultado(800, 7, 13, 13),
    )
    assert recommend([r200, r400, r800]).size == 200


def test_full_tie_keeps_the_current_size():
    r200, r400, r800 = (
        resultado(200, 9, 12, 14),
        resultado(400, 9, 12, 14),
        resultado(800, 9, 12, 14),
    )
    assert recommend([r200, r400, r800], actual=400).size == 400


def test_threshold_is_the_midpoint_when_distances_separate():
    assert suggest_threshold(resultado(400, 9, 12, 14, peor=0.52, fuera=0.72)) == 0.62


def test_no_threshold_when_distances_overlap():
    assert suggest_threshold(resultado(400, 9, 12, 14, peor=0.75, fuera=0.71)) is None


def test_each_size_gets_its_collection_and_a_report(tmp_path, capsys):
    report = tmp_path / "chunking.md"
    args = ["--embedder", "fake", "--store", str(tmp_path / "chroma"), "--sizes", "200", "800"]

    assert main([*args, "--report", str(report)]) == 0

    out = capsys.readouterr().out
    assert "| tamaño | chunks | hit@1 | hit@3 | hit@5 | peor dentro | q11 | q16 |" in out
    assert "| 200 |" in out and "| 800 |" in out
    assert collection_name(200) == "docs_operador_c200"
    texto = report.read_text(encoding="utf-8")
    assert texto.startswith("# Tamaño de chunk y umbral")
    assert "## Decisión" in texto
