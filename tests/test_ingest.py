from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from knowledge.ingest import IngestError, clean_text, load_dir, load_path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "docs"
HEADER = "Acme Operaciones · Contrato de servicio · Confidencial"


def make_pdf(path: Path, pages: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=letter)
    for number, lines in enumerate(pages, start=1):
        pdf.drawString(72, 740, HEADER)
        pdf.drawString(72, 40, f"Página {number} de {len(pages)}")
        y = 700
        for line in lines:
            pdf.drawString(72, y, line)
            y -= 15
        pdf.showPage()
    pdf.save()
    return path


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pdf_gives_one_document_per_page(tmp_path):
    make_pdf(
        tmp_path / "contratos" / "pagosred.pdf",
        [
            ["Cláusula 1. Objeto del contra-", "to de servicio."],
            ["Cláusula 2. Vigencia."],
            ["Fin."],
        ],
    )

    docs = load_path(tmp_path / "contratos" / "pagosred.pdf", tmp_path)

    assert [d.metadatos["pagina"] for d in docs] == [1, 2, 3]
    assert {d.doc_id for d in docs} == {"contratos/pagosred.pdf"}
    assert len({d.metadatos["hash"] for d in docs}) == 1  # hash del archivo completo
    assert docs[0].texto == "Cláusula 1. Objeto del contrato de servicio."
    assert all(HEADER not in d.texto and "Página" not in d.texto for d in docs)


def test_clean_text_joins_hyphenated_words():
    assert clean_text("tiempo de res-\npuesta") == "tiempo de respuesta"


def test_clean_text_removes_repeated_headers_and_footers():
    pages = [
        f"{HEADER}\nSección {letra}\nTexto de la página {n}\nCierre {letra}\nPágina {n} de 3"
        for n, letra in zip((1, 2, 3), "ABC", strict=True)
    ]
    cleaned = clean_text("\f".join(pages)).split("\f")
    assert cleaned[0] == "Sección A\nTexto de la página 1\nCierre A"
    assert all(HEADER not in page and "de 3" not in page for page in cleaned)


def test_clean_text_keeps_lines_of_a_single_page():
    assert clean_text(f"{HEADER}\nTexto") == f"{HEADER}\nTexto"


def test_clean_text_collapses_spaces_and_blank_lines_and_normalizes_nfc():
    decomposed = "Camio\u0301n"  # "Camión" con el acento como carácter aparte.
    assert (
        clean_text(f"  {decomposed}   lleno \n\n\n\n otra  línea ") == "Camión lleno\n\notra línea"
    )


def test_metadata_from_folder_and_provider_only_in_contracts():
    by_id = {d.doc_id: d.metadatos for d in load_dir(DOCS)}

    assert by_id["runbooks/caida-pagos.md"]["tipo_doc"] == "runbook"
    assert by_id["politicas/escalamiento.md"]["tipo_doc"] == "politica"
    assert by_id["contratos/pagosred.pdf"]["tipo_doc"] == "contrato"
    assert by_id["faq/accesos.txt"]["tipo_doc"] == "faq"
    assert by_id["politicas/escalamiento.md"]["titulo"] == "Escalamiento"
    assert by_id["contratos/transportes-sierra.pdf"]["titulo"] == (
        "Contrato de servicio Transportes Sierra"
    )
    assert by_id["contratos/transportes-sierra.pdf"]["proveedor"] == "Transportes Sierra"
    assert all("proveedor" not in m for i, m in by_id.items() if not i.startswith("contratos/"))


def test_no_metadata_value_is_none():
    for doc in load_dir(DOCS):
        assert None not in doc.metadatos.values(), doc.doc_id


def test_hash_changes_with_text_but_not_with_spacing(tmp_path):
    path = tmp_path / "runbooks" / "prueba.md"
    path.parent.mkdir()

    def hash_of(texto: str) -> str:
        path.write_text(texto, encoding="utf-8")
        return load_path(path, tmp_path)[0].metadatos["hash"]

    original = hash_of("# Prueba\n\nReinicia la cola de pedidos.")
    assert hash_of("# Prueba\n\n\n\nReinicia   la cola de pedidos.  ") == original
    assert hash_of("# Prueba\n\nReinicia la cola de facturas.") != original


def test_unsupported_format_raises(tmp_path):
    path = tmp_path / "faq" / "notas.docx"
    path.parent.mkdir()
    path.write_bytes(b"")
    with pytest.raises(IngestError, match=r"formato no soportado: \.docx"):
        load_path(path, tmp_path)


def test_ingest_docs_is_idempotent(tmp_path, capsys):
    corpus = tmp_path / "docs"
    shutil.copytree(DOCS, corpus)
    ingest = load_script("ingest_docs")
    args = [str(corpus), "--embedder", "fake", "--store", str(tmp_path / "chroma")]

    assert ingest.main(args) == 0
    first = capsys.readouterr().out.strip()
    assert ingest.main(args) == 0
    second = capsys.readouterr().out.strip()
    with (corpus / "faq" / "accesos.txt").open("a", encoding="utf-8") as f:
        f.write("Una línea nueva al final.\n")
    assert ingest.main(args) == 0
    third = capsys.readouterr().out.strip()

    assert first.startswith("ingesta: 12 archivos · 12 nuevos · 0 modificados · 0 sin cambios · ")
    assert first.endswith(" · 0 errores")
    assert second == (
        "ingesta: 12 archivos · 0 nuevos · 0 modificados · 12 sin cambios · 0 chunks · 0 errores"
    )
    assert "· 1 modificado · 11 sin cambios ·" in third


def test_ingest_docs_counts_errors_and_keeps_going(tmp_path, capsys):
    corpus = tmp_path / "docs"
    shutil.copytree(DOCS / "faq", corpus / "faq")
    (corpus / "faq" / "roto.docx").write_bytes(b"")
    ingest = load_script("ingest_docs")

    code = ingest.main([str(corpus), "--embedder", "fake", "--store", str(tmp_path / "chroma")])

    out = capsys.readouterr()
    assert code == 1
    assert "4 archivos · 3 nuevos" in out.out and out.out.strip().endswith("· 1 error")
    assert "formato no soportado: .docx" in out.err
