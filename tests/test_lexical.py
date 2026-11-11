from __future__ import annotations

import pytest

from knowledge.embeddings import FakeEmbedder
from knowledge.lexical import BM25Index, matches, tokenize
from knowledge.store import open_store
from tests.test_store import document

CORPUS = {
    "faq/soporte-general.txt#0": "¿Qué pasó con mi ticket? Responde al correo del ticket.",
    "faq/soporte-general.txt#1": "Abre un ticket con la mesa de ayuda; el ticket empieza con T-.",
    "politicas/sla.md#0": "Cada ticket tiene prioridad; el ticket urgente se atiende pronto.",
    "runbooks/caida-pagos.md#0": "El T-1042 tiene un doble cargo a la tarjeta del cliente.",
}


def test_tokenize_keeps_ids_emails_and_numbers():
    texto = "¿Qué pasó con el T-1042? Escribe a mesa@nubehost.example: 99.5%."
    assert tokenize(texto) == ["paso", "t-1042", "escribe", "mesa@nubehost.example", "99.5"]


def test_rare_term_beats_a_repeated_common_term():
    hits = BM25Index(CORPUS).search("¿Qué dice el ticket T-1042?", k=4)

    assert hits[0].chunk_id == "runbooks/caida-pagos.md#0"
    assert [h.puntaje for h in hits] == sorted((h.puntaje for h in hits), reverse=True)


def test_chunks_without_terms_in_common_are_not_returned():
    assert BM25Index(CORPUS).search("vacaciones") == []


@pytest.mark.parametrize(
    ("where", "esperado"),
    [
        (None, True),
        ({"tipo_doc": "contrato"}, True),
        ({"tipo_doc": "politica"}, False),
        ({"$and": [{"tipo_doc": "contrato"}, {"proveedor": "PagosRed"}]}, True),
        ({"$and": [{"tipo_doc": "contrato"}, {"proveedor": "NubeHost"}]}, False),
    ],
)
def test_matches_understands_build_where(where, esperado):
    assert matches({"tipo_doc": "contrato", "proveedor": "PagosRed"}, where) is esperado


def test_from_store_reads_chunks_title_and_section(tmp_path):
    store = open_store(FakeEmbedder(dim=64), path=str(tmp_path / "chroma"))
    store.index_document(
        *document("politicas/escalamiento.md", ["Escalar a Guardia N2."], seccion="Prioridad alta")
    )
    store.index_document(
        *document("faq/accesos.txt", ["Restablece tu contraseña."], tipo_doc="faq")
    )

    index = BM25Index.from_store(store)

    assert index.search("prioridad alta")[0].chunk_id == "politicas/escalamiento.md#0"
    assert index.search("contraseña", where={"tipo_doc": "politica"}) == []
